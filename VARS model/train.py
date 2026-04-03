import logging
import os
import torch
import gc
from config.classes import INVERSE_EVENT_DICTIONARY
import json
from SoccerNet.Evaluation.MV_FoulRecognition import evaluate
from tqdm import tqdm

def _decode_predictions(preds_sev, preds_act, actions, action_ids):
    """Wspólna logika dekodowania predykcji dla batcha."""
    for i in range(len(action_ids)):
        values = {}
        values["Action class"] = INVERSE_EVENT_DICTIONARY["action_class"][preds_act[i].item()]
        sev = preds_sev[i].item()
        if sev == 0:
            values["Offence"] = "No offence"
            values["Severity"] = ""
        elif sev == 1:
            values["Offence"] = "Offence"
            values["Severity"] = "1.0"
        elif sev == 2:
            values["Offence"] = "Offence"
            values["Severity"] = "3.0"
        elif sev == 3:
            values["Offence"] = "Offence"
            values["Severity"] = "5.0"
        actions[action_ids[i]] = values


def trainer(train_loader,
            val_loader2,
            test_loader2,
            model,
            optimizer,
            scheduler,
            criterion,
            best_model_path,
            epoch_start,
            model_name,
            path_dataset,
            max_epochs=1000
            ):

    logging.info("start training")
    counter = 0
    best_val = 0.0
    for epoch in range(epoch_start, max_epochs):

        print(f"Epoch {epoch+1}/{max_epochs}")

        pbar = tqdm(total=len(train_loader), desc="Training", position=0, leave=True)

        ###################### TRAINING ###################
        prediction_file, loss_action, loss_offence_severity = train(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch + 1,
            model_name,
            train=True,
            set_name="train",
            pbar=pbar,
        )

        results = evaluate(os.path.join(path_dataset, "Train", "annotations.json"), prediction_file)
        print("TRAINING")
        print(results)

        ###################### VALIDATION ###################
        prediction_file, loss_action, loss_offence_severity = train(
            val_loader2,
            model,
            criterion,
            optimizer,
            epoch + 1,
            model_name,
            train=False,
            set_name="valid"
        )

        results = evaluate(os.path.join(path_dataset, "Valid", "annotations.json"), prediction_file)
        print("VALIDATION")
        print(results)

        # Po ewaluacji validacji:
        val_leaderboard = results.get('leaderboard_value', 0)
        if val_leaderboard > best_val:
            best_val = val_leaderboard
            best_state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            torch.save(best_state, os.path.join(best_model_path, "best_model.pth.tar"))

        ###################### TEST ###################
        prediction_file, loss_action, loss_offence_severity = train(
            test_loader2,
            model,
            criterion,
            optimizer,
            epoch + 1,
            model_name,
            train=False,
            set_name="test",
        )

        results = evaluate(os.path.join(path_dataset, "Test", "annotations.json"), prediction_file)
        print("TEST")
        print(results)

        scheduler.step()

        counter += 1

        if counter >= 1:
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            path_aux = os.path.join(best_model_path, str(epoch + 1) + "_model.pth.tar")
            torch.save(state, path_aux)

    pbar.close()
    return


def train(dataloader,
          model,
          criterion,
          optimizer,
          epoch,
          model_name,
          train=False,
          set_name="train",
          pbar=None,
          ):

    if train:
        model.train()
    else:
        model.eval()

    loss_total_action = 0
    loss_total_offence_severity = 0
    total_loss = 0

    if not os.path.isdir(model_name):
        os.mkdir(model_name)

    prediction_file = "predicitions_" + set_name + "_epoch_" + str(epoch) + ".json"
    data = {}
    data["Set"] = set_name
    actions = {}

    for targets_offence_severity, targets_action, mvclips, action in dataloader:

        targets_offence_severity = targets_offence_severity.cuda()
        targets_action = targets_action.cuda()
        mvclips = mvclips.cuda().float()

        if pbar is not None:
            pbar.update()

        outputs_offence_severity, outputs_action, _ = model(mvclips)

        # Ujednolicona ścieżka dla batch_size=1 i >1
        preds_sev = torch.argmax(outputs_offence_severity.detach().cpu(), dim=1)
        preds_act = torch.argmax(outputs_action.detach().cpu(), dim=1)
        _decode_predictions(preds_sev, preds_act, actions, action)

        # Zabezpieczenie przed outputem 1D przy batch_size=1
        if outputs_offence_severity.dim() == 1:
            outputs_offence_severity = outputs_offence_severity.unsqueeze(0)
        if outputs_action.dim() == 1:
            outputs_action = outputs_action.unsqueeze(0)

        loss_offence_severity = criterion[0](outputs_offence_severity, targets_offence_severity)
        loss_action = criterion[1](outputs_action, targets_action)
        loss = loss_offence_severity + loss_action

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_total_action += float(loss_action)
        loss_total_offence_severity += float(loss_offence_severity)
        total_loss += 1

    gc.collect()
    torch.cuda.empty_cache()

    data["Actions"] = actions
    with open(os.path.join(model_name, prediction_file), "w") as outfile:
        json.dump(data, outfile)

    return (
        os.path.join(model_name, prediction_file),
        loss_total_action / total_loss,
        loss_total_offence_severity / total_loss
    )


def evaluation(dataloader,
               model,
               set_name="test",
               ):

    model.eval()

    prediction_file = "predicitions_" + set_name + ".json"
    data = {}
    data["Set"] = set_name
    actions = {}

    for _, _, mvclips, action in dataloader:

        mvclips = mvclips.cuda().float()
        outputs_offence_severity, outputs_action, _ = model(mvclips)

        # Ujednolicona ścieżka identyczna jak w train()
        preds_sev = torch.argmax(outputs_offence_severity.detach().cpu(), dim=1)
        preds_act = torch.argmax(outputs_action.detach().cpu(), dim=1)
        _decode_predictions(preds_sev, preds_act, actions, action)

    gc.collect()
    torch.cuda.empty_cache()

    data["Actions"] = actions
    with open(prediction_file, "w") as outfile:
        json.dump(data, outfile)

    return prediction_file
