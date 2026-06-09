import os
import json
import logging
import time
import tempfile
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from dataset import MultiViewDataset
from model import MVNetwork, EarlyFusionNetwork
from train import EMA, _run_with_tta, ordinal_to_probs
from SoccerNet.Evaluation.MV_FoulRecognition import evaluate
from main import checkArguments, VideoMAETransform, TAdaFormerTransform, HieraTransform, VJEPA21Transform, InternVideo2Transform
from config.classes import INVERSE_EVENT_DICTIONARY
from torchvision.models.video import (
    R3D_18_Weights,
    MC3_18_Weights,
    R2Plus1D_18_Weights,
    S3D_Weights,
    MViT_V2_S_Weights,
    MViT_V1_B_Weights,
)
from model import HF_VIDEOMAE_REGISTRY

_TORCHVISION_MODELS = {
    "r3d_18",
    "mc3_18",
    "r2plus1d_18",
    "s3d",
    "mvit_v2_s",
    "mvit_v1_b",
    "tadaformer_b16",
    "aim_vitb16",
    "vjepa21_vitb",
    "intern_video2_dist_b",
}
HIERA_MODELS = {
    "hiera_base_16x224",
    "hiera_base_plus_16x224",
    "hiera_large_16x224",
    "hiera_huge_16x224",
}

def collect_logits(dataloader, model, use_tta=True):
    model.eval()
    all_sev_logits = []
    all_act_logits = []
    all_action_ids = []
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 8:
                mvclips = batch[6]
                action_ids = batch[7]
                text_feat = None
            elif len(batch) == 9:
                mvclips = batch[6]
                action_ids = batch[7]
                text_feat = batch[8]
            else:
                raise ValueError(f"Unexpected batch size: {len(batch)}")

            mvclips = mvclips.cuda().float()
            if text_feat is not None:
                text_feat = text_feat.cuda().float()

            if use_tta:
                out_sev, out_act, *_ = _run_with_tta(
                    model, mvclips, text_features=text_feat
                )
            else:
                out = model(mvclips, text_features=text_feat)
                out_sev, out_act = out[0], out[1]
            
            all_sev_logits.append(out_sev.cpu())
            all_act_logits.append(out_act.cpu())
            all_action_ids.extend(action_ids)
            
    return torch.cat(all_sev_logits, dim=0), torch.cat(all_act_logits, dim=0), all_action_ids

def decode_predictions_adjusted(preds_sev, preds_act, action_ids, split_name):
    actions = {}
    preds_sev = preds_sev.reshape(-1)
    preds_act = preds_act.reshape(-1)
    for i in range(len(action_ids)):
        values = {}
        values["Action class"] = INVERSE_EVENT_DICTIONARY["action_class"][
            preds_act[i].item()
        ]
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
    return {"Set": split_name, "Actions": actions}

def evaluate_adjusted(preds_sev, preds_act, action_ids, ann_path, split_name):
    pred_dict = decode_predictions_adjusted(preds_sev, preds_act, action_ids, split_name)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(pred_dict, f)
        temp_path = f.name
    try:
        results = evaluate(ann_path, temp_path)
    finally:
        os.remove(temp_path)
    return results

def main(args):
    # Setup model and data transforms (copied from main.py)
    start_frame = args.start_frame
    end_frame = args.end_frame
    fps = args.fps
    pre_model = args.pre_model
    num_views = args.num_views
    path = args.path
    pooling_type = args.pooling_type
    max_num_worker = args.max_num_worker
    path_to_model_weights = args.path_to_model_weights
    ema_decay = args.ema_decay
    use_tta = args.use_tta
    fusion_mode = args.fusion_mode
    graph_topology = args.graph_topology
    cascade_severity = args.cascade_severity
    use_text_bridge = args.use_text_bridge
    descriptions_path = args.descriptions_path
    use_mffm = args.use_mffm
    use_decoder = args.use_decoder

    number_of_frames = int(
        (end_frame - start_frame) / (((end_frame - start_frame) / 25) * fps)
    )

    _videomae_transform = VideoMAETransform(size=224)
    _transform_map = {
        "r3d_18": R3D_18_Weights.KINETICS400_V1.transforms(),
        "mc3_18": MC3_18_Weights.KINETICS400_V1.transforms(),
        "r2plus1d_18": R2Plus1D_18_Weights.KINETICS400_V1.transforms(),
        "s3d": S3D_Weights.KINETICS400_V1.transforms(),
        "mvit_v2_s": MViT_V2_S_Weights.KINETICS400_V1.transforms(),
        "mvit_v1_b": MViT_V1_B_Weights.KINETICS400_V1.transforms(),
        **{k: _videomae_transform for k in HF_VIDEOMAE_REGISTRY},
        "tadaformer_b16": TAdaFormerTransform(size=224),
        "aim_vitb16": TAdaFormerTransform(size=224),
        **{k: HieraTransform(size=224) for k in HIERA_MODELS},
        "vjepa21_vitb": VJEPA21Transform(size=384),
        "intern_video2_dist_b": InternVideo2Transform(size=224),
    }

    transforms_model = (
        MViT_V2_S_Weights.KINETICS400_V1.transforms()
        if fusion_mode
        else _transform_map.get(
            pre_model, R2Plus1D_18_Weights.KINETICS400_V1.transforms()
        )
    )

    dataset_kwargs = dict(
        path=path,
        start=start_frame,
        end=end_frame,
        fps=fps,
        transform_model=transforms_model,
        fusion_mode=fusion_mode,
        backbone_type=pre_model,
        descriptions_path=descriptions_path,
    )

    # Dataloaders for Valid and Test
    dataset_Valid = MultiViewDataset(
        **dataset_kwargs, split="Valid", num_views=num_views
    )
    dataset_Test = MultiViewDataset(
        **dataset_kwargs, split="Test", num_views=num_views
    )
    
    val_loader = torch.utils.data.DataLoader(
        dataset_Valid,
        batch_size=1,
        shuffle=False,
        num_workers=max_num_worker,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_Test,
        batch_size=1,
        shuffle=False,
        num_workers=max_num_worker,
        pin_memory=True,
    )

    # Initialize model
    if fusion_mode:
        model = EarlyFusionNetwork(
            num_views=num_views,
            T_per_view=number_of_frames,
            cascade_severity=cascade_severity,
        ).cuda()
    else:
        model = MVNetwork(
            net_name=pre_model,
            agr_type=pooling_type,
            graph_topology=graph_topology,
            cascade_severity=cascade_severity,
            use_text_bridge=use_text_bridge,
            use_mffm=use_mffm,
            use_decoder=use_decoder,
        ).cuda()

    # Load weights
    if path_to_model_weights == "":
        raise ValueError("Must provide --path_to_model_weights to evaluate a checkpoint.")
    
    load = torch.load(path_to_model_weights, map_location="cpu")
    model.load_state_dict(load["state_dict"])
    print(f"Loaded model weights from {path_to_model_weights}")

    # Set up EMA
    ema = EMA(model, decay=ema_decay)
    if "ema" in load:
        ema.load_state_dict(load["ema"])
        ema.apply_shadow()
        print("Applied EMA shadow weights.")
    else:
        print("No EMA state found in checkpoint; evaluating using standard model weights.")

    # Paths to annotations
    val_ann_path = os.path.join(path, "Valid", "annotations.json")
    test_ann_path = os.path.join(path, "Test", "annotations.json")

    # 1. Inference on validation set to collect logits
    print("Running inference on validation set...")
    val_sev_logits, val_act_logits, val_action_ids = collect_logits(val_loader, model, use_tta=use_tta)
    print(f"Collected validation logits for {len(val_action_ids)} samples.")

    # 2. Sweep tau values
    tau_range = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    
    # Class priors
    action_counts = torch.tensor([1559.0, 576.0, 475.0, 457.0, 218.0, 130.0, 108.0, 34.0])
    prior_action = action_counts / action_counts.sum()

    severity_counts = torch.tensor([431.0, 1751.0, 859.0, 35.0])
    prior_severity = severity_counts / severity_counts.sum()

    best_lb = -1.0
    best_tau_act = 0.0
    best_tau_sev = 0.0
    best_results = None

    print("\n--- Sweeping tau values on validation set ---")
    print(f"{'tau_act':<8} | {'tau_sev':<8} | {'BAact (%)':<10} | {'BAsev (%)':<10} | {'Leaderboard (%)':<16}")
    print("-" * 62)

    for tau_act in tau_range:
        for tau_sev in tau_range:
            # Action head adjustment
            adjusted_act_logits = val_act_logits - tau_act * torch.log(prior_action + 1e-8)
            preds_act = torch.argmax(adjusted_act_logits, dim=-1)

            # Severity head adjustment
            class_probs = ordinal_to_probs(val_sev_logits)
            log_probs = torch.log(class_probs)
            adjusted_log_probs = log_probs - tau_sev * torch.log(prior_severity + 1e-8)
            preds_sev = torch.argmax(adjusted_log_probs, dim=-1)

            # Evaluate
            res = evaluate_adjusted(preds_sev, preds_act, val_action_ids, val_ann_path, "Valid")
            
            ba_act = res["balanced_accuracy_action"]
            ba_sev = res["balanced_accuracy_offence_severity"]
            lb_val = res["leaderboard_value"]

            print(f"{tau_act:<8.2f} | {tau_sev:<8.2f} | {ba_act:<10.2f} | {ba_sev:<10.2f} | {lb_val:<16.2f}")

            if lb_val > best_lb:
                best_lb = lb_val
                best_tau_act = tau_act
                best_tau_sev = tau_sev
                best_results = res

    print("\n" + "=" * 62)
    print("Optimal τ pair found on validation set:")
    print(f"  τ_action = {best_tau_act:.2f}")
    print(f"  τ_severity = {best_tau_sev:.2f}")
    print(f"  Validation Leaderboard value = {best_lb:.2f}%")
    print(f"  Validation BAact = {best_results['balanced_accuracy_action']:.2f}%")
    print(f"  Validation BAsev = {best_results['balanced_accuracy_offence_severity']:.2f}%")
    print("=" * 62 + "\n")

    # 3. Inference on test set
    print("Running inference on test set...")
    test_sev_logits, test_act_logits, test_action_ids = collect_logits(test_loader, model, use_tta=use_tta)
    print(f"Collected test logits for {len(test_action_ids)} samples.")

    # Apply optimal tau values on test set
    adjusted_test_act_logits = test_act_logits - best_tau_act * torch.log(prior_action + 1e-8)
    test_preds_act = torch.argmax(adjusted_test_act_logits, dim=-1)

    test_class_probs = ordinal_to_probs(test_sev_logits)
    test_log_probs = torch.log(test_class_probs)
    adjusted_test_log_probs = test_log_probs - best_tau_sev * torch.log(prior_severity + 1e-8)
    test_preds_sev = torch.argmax(adjusted_test_log_probs, dim=-1)

    # Evaluate on test set
    test_pred_dict = decode_predictions_adjusted(test_preds_sev, test_preds_act, test_action_ids, "Test")
    test_prediction_file = f"predictions_test_adjusted_tau_{best_tau_act}_{best_tau_sev}.json"
    with open(test_prediction_file, "w") as f:
        json.dump(test_pred_dict, f)
        
    print(f"Saved test predictions to {test_prediction_file}")
    
    test_res = evaluate(test_ann_path, test_prediction_file)
    print("\n" + "=" * 62)
    print("Test set results with optimal τ values:")
    print(f"  τ_action = {best_tau_act:.2f}")
    print(f"  τ_severity = {best_tau_sev:.2f}")
    print(f"  Test Leaderboard value = {test_res['leaderboard_value']:.2f}%")
    print(f"  Test BAact = {test_res['balanced_accuracy_action']:.2f}%")
    print(f"  Test BAsev = {test_res['balanced_accuracy_offence_severity']:.2f}%")
    print("=" * 62 + "\n")

    if "ema" in load:
        ema.restore()

if __name__ == "__main__":
    parser = ArgumentParser(
        description="VARS Standalone Logit Adjustment Sweep and Evaluation",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    # Data
    parser.add_argument("--path", required=True, type=str)
    parser.add_argument("--start_frame", default=0, type=int)
    parser.add_argument("--end_frame", default=125, type=int)
    parser.add_argument("--fps", default=25, type=int)
    parser.add_argument("--num_views", default=5, type=int)
    parser.add_argument("--data_aug", default="Yes", type=str)
    parser.add_argument("--aug_preset", default="default", type=str)

    # Model
    parser.add_argument("--fusion_mode", action="store_true", default=False)
    parser.add_argument("--pre_model", default="mvit_v2_s", type=str)
    parser.add_argument("--pooling_type", default="transformer", type=str)
    parser.add_argument("--use_text_bridge", action="store_true", default=False)
    parser.add_argument("--use_mffm", action="store_true", default=False)
    parser.add_argument("--use_decoder", action="store_true", default=False)
    parser.add_argument("--descriptions_path", default=None, type=str)
    parser.add_argument("--graph_topology", default="structured", type=str)

    # Advanced options
    parser.add_argument("--cascade_severity", action="store_true", default=False)
    parser.add_argument("--ema_decay", default=0.999, type=float)
    parser.add_argument("--use_tta", action="store_true", default=True)

    # Infra
    parser.add_argument("--GPU", default=0, type=int)
    parser.add_argument("--max_num_worker", default=4, type=int)
    parser.add_argument("--path_to_model_weights", required=True, type=str)

    args = parser.parse_args()
    
    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    main(args)
