import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import copy
from config.classes import INVERSE_EVENT_DICTIONARY
import json
import io, contextlib
from SoccerNet.Evaluation.MV_FoulRecognition import evaluate as _sn_evaluate
from clip_severity_loss import CLIPSeverityLoss


def evaluate(ann_path, pred_file):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = _sn_evaluate(ann_path, pred_file)
    return result


from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ordinal severity helpers
# ---------------------------------------------------------------------------


def ordinal_targets(labels_int, num_thresholds=3, device="cpu", smoothing=0.0):
    targets = torch.stack(
        [(labels_int > k).float() for k in range(num_thresholds)], dim=1
    )
    if smoothing > 0:
        # Smooth: push 1s down, 0s up
        targets = targets * (1 - smoothing) + smoothing * 0.5
    return targets.to(device)


def ordinal_loss(logits, labels_int, pos_weight=None):
    """
    Cumulative BCE loss for ordinal regression.
    logits : (B, 3)
    labels_int : (B,) integer 0-3
    """
    smoothing = 0.0 if pos_weight is not None else 0.1
    targets = ordinal_targets(
        labels_int,
        num_thresholds=logits.shape[1],
        device=logits.device,
        smoothing=smoothing,
    )
    if pos_weight is not None:
        pos_weight = pos_weight.to(logits.device)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def temporal_entropy_loss(temporal_weights):
    """
    Encourages temporal attention weights to be peaked (low entropy)
    rather than uniform. This pushes the model to focus on one moment
    rather than averaging across all frames.

    temporal_weights : [B, V, T'] or None
    Returns scalar loss (0.0 if weights is None or all-padded).
    """
    if temporal_weights is None:
        return torch.tensor(0.0)

    # temporal_weights: [B, V, T'], values sum to 1 over T' per view
    # Entropy = -sum(p * log(p)), lower = more peaked = better localization
    # Clamp to avoid log(0)
    w = temporal_weights.clamp(min=1e-8)
    entropy = -(w * w.log()).sum(dim=-1)  # [B, V]

    # Only include non-padded views (padded views have nan weights → already 0)
    entropy = torch.nan_to_num(entropy, nan=0.0)
    return entropy.mean()


def ordinal_predict(logits):
    """
    Decode ordinal logits to integer class predictions.
    Count how many cumulative thresholds are exceeded (>= 0.5).
    Returns (B,) integer tensor.
    """
    return (torch.sigmoid(logits) >= 0.5).sum(dim=1)


def ordinal_to_probs(logits):
    """
    Convert ordinal logits (B, K-1) to per-class probabilities (B, K).
    """
    p = torch.sigmoid(logits)
    B = p.shape[0]
    ones = torch.ones(B, 1, device=p.device)
    zeros = torch.zeros(B, 1, device=p.device)
    cum = torch.cat([ones, p, zeros], dim=1)
    class_probs = (cum[:, :-1] - cum[:, 1:]).clamp(min=1e-8)
    return class_probs / class_probs.sum(dim=1, keepdim=True)


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average) weight tracker
# ---------------------------------------------------------------------------


class EMA:
    """
    Maintains a shadow copy of model weights updated as:
        shadow = decay * shadow + (1 - decay) * param
    Call apply_shadow() before evaluation, restore() after.
    """

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register()

    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                )

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, d):
        self.shadow = d["shadow"]
        self.decay = d["decay"]

    def register_new(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name not in self.shadow:
                self.shadow[name] = param.data.clone()


# ---------------------------------------------------------------------------
# Uncertainty-weighted multi-task loss (Kendall & Gal, 2018)
# ---------------------------------------------------------------------------


class UncertaintyWeighting(nn.Module):
    """
    Learnable homoscedastic uncertainty weighting for N tasks.

    Optimises log-variance scalars s_i = log(σ_i²):
        L_total = Σ_i  exp(-s_i) · L_i  +  s_i

    For two tasks (severity + action):
        L = exp(-s0)·L_sev + s0 + exp(-s1)·L_act + s1

    Low-loss tasks are naturally up-weighted; high-variance tasks are
    down-weighted.  s_i are initialised at 0 (equal weighting at start).
    """

    def __init__(self, num_tasks: int = 2):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        return sum(
            torch.exp(-self.log_vars[i]) * losses[i] + self.log_vars[i]
            for i in range(len(losses))
        )


# ---------------------------------------------------------------------------
# Prediction decoding
# ---------------------------------------------------------------------------


def _decode_predictions(preds_sev, preds_act, actions, action_ids):
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


# ---------------------------------------------------------------------------
# TTA helper
# ---------------------------------------------------------------------------


def _run_with_tta(model, mvclips, text_features=None):
    o1 = model(mvclips, text_features=text_features)
    o2 = model(mvclips.flip(-1), text_features=text_features)
    sev = (o1[0] + o2[0]) / 2
    act = (o1[1] + o2[1]) / 2
    return sev, act, o1[2], o1[3], o1[4], o1[5]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def _find_vjepa_backbone(model):
    for module in model.modules():
        if hasattr(module, "get_layerwise_lr_groups"):
            return module
    return None


def trainer(
    train_loader,
    val_loader,
    test_loader,
    model,
    optimizer,
    scheduler,
    criterion,
    ema,
    best_model_path,
    epoch_start,
    model_name,
    path_dataset,
    max_epochs=40,
    patience=8,
    aux_weight=0.2,
    use_tta=True,
    accum_steps=1,
    backbone_prefix="aggregation_model.model.",
    freeze_epoch=8,
    uncertainty_weighter=None,
    val_ann_path=None,
    train_all_but_test=False,
    clip_weight=0.0,
    clip_embeddings_path="",
):
    logging.info("start training")
    best_val = 0.0
    no_improve = 0

    if val_ann_path is None:
        val_ann_path = os.path.join(path_dataset, "Valid", "annotations.json")

    clip_loss_fn = None
    clip_warmup_epochs = 3
    if clip_weight > 0 and not clip_embeddings_path:
        logging.warning(
            "clip_weight > 0 but clip_embeddings_path is empty; CLIP loss disabled."
        )
    if clip_weight > 0 and clip_embeddings_path:
        try:
            clip_loss_fn = CLIPSeverityLoss(clip_embeddings_path).cuda().eval()
            logging.info(f"Loaded CLIP severity embeddings from {clip_embeddings_path}")
        except Exception as exc:
            logging.warning(
                f"Failed to load CLIP embeddings from {clip_embeddings_path}: {exc}"
            )
            clip_loss_fn = None

    for epoch in range(epoch_start, max_epochs):

        # Unfreeze backbone at freeze_epoch with discriminative fine-tuning
        if epoch == freeze_epoch:
            vjepa_backbone = _find_vjepa_backbone(model)
            if vjepa_backbone is not None and hasattr(vjepa_backbone, "encoder"):
                for p in vjepa_backbone.encoder.parameters():
                    p.requires_grad = True
                for group in vjepa_backbone.get_layerwise_lr_groups(
                    base_lr=1e-5, decay=0.65
                ):
                    optimizer.add_param_group(group)
                ema.register_new()
                logging.info(
                    "Backbone unfrozen with layer-wise LR decay (base_lr=1e-5, decay=0.65)"
                )
            else:
                backbone_params = [
                    p
                    for n, p in model.named_parameters()
                    if backbone_prefix in n and not p.requires_grad
                ]
                for p in backbone_params:
                    p.requires_grad = True
                if backbone_params:
                    optimizer.add_param_group({"params": backbone_params, "lr": 1e-5})
                ema.register_new()
                logging.info(
                    f"Backbone unfrozen at epoch {freeze_epoch} (prefix='{backbone_prefix}') — "
                    f"{len(backbone_params)} param groups added at LR=1e-5"
                )

        print(f"\nEpoch {epoch + 1}/{max_epochs}")
        pbar = tqdm(total=len(train_loader), desc="Training", leave=True)

        clip_weight_epoch = clip_weight
        if clip_weight > 0 and (epoch + 1) <= clip_warmup_epochs:
            clip_weight_epoch = 0.0

        # --- Train ---
        pred_file, loss_act, loss_sev = _train_epoch(
            train_loader,
            model,
            optimizer,
            criterion,
            ema,
            epoch + 1,
            model_name,
            train=True,
            set_name="train",
            aux_weight=aux_weight,
            pbar=pbar,
            accum_steps=accum_steps,
            uncertainty_weighter=uncertainty_weighter,
            clip_loss_fn=clip_loss_fn,
            clip_weight=clip_weight_epoch,
        )
        if not train_all_but_test:
            results = evaluate(
                os.path.join(path_dataset, "Train", "annotations.json"), pred_file
            )
            print("TRAINING RESULTS:", results)

        # --- Validation (with EMA weights) ---
        ema.apply_shadow()
        pred_file, _, _ = _train_epoch(
            val_loader,
            model,
            optimizer,
            criterion,
            ema,
            epoch + 1,
            model_name,
            train=False,
            set_name="valid",
            aux_weight=aux_weight,
            use_tta=use_tta,
            uncertainty_weighter=uncertainty_weighter,
            clip_loss_fn=None,
            clip_weight=0.0,
        )
        ema.restore()

        results = evaluate(val_ann_path, pred_file)
        print("VALIDATION RESULTS:", results)

        val_lb = results.get("leaderboard_value", 0)
        if val_lb > best_val:
            best_val = val_lb
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "uw": (
                        uncertainty_weighter.state_dict()
                        if uncertainty_weighter is not None
                        else None
                    ),
                },
                os.path.join(best_model_path, "best_model.pth.tar"),
            )
            logging.info(f"New best val LB: {best_val:.4f} at epoch {epoch + 1}")
        else:
            no_improve += 1
            if no_improve >= patience:
                logging.info(
                    f"Early stopping at epoch {epoch + 1} (patience={patience}), best LB={best_val:.4f}"
                )
                break

        # --- Test (with EMA weights) ---
        ema.apply_shadow()
        pred_file, _, _ = _train_epoch(
            test_loader,
            model,
            optimizer,
            criterion,
            ema,
            epoch + 1,
            model_name,
            train=False,
            set_name="test",
            aux_weight=aux_weight,
            use_tta=use_tta,
            uncertainty_weighter=uncertainty_weighter,
            clip_loss_fn=None,
            clip_weight=0.0,
        )
        ema.restore()

        results = evaluate(
            os.path.join(path_dataset, "Test", "annotations.json"), pred_file
        )
        print("TEST RESULTS:", results)

        scheduler.step()

        torch.save(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "uw": (
                    uncertainty_weighter.state_dict()
                    if uncertainty_weighter is not None
                    else None
                ),
            },
            os.path.join(best_model_path, f"{epoch + 1}_model.pth.tar"),
        )

    if "pbar" in locals():
        pbar.close()


# ---------------------------------------------------------------------------
# Single-epoch loop
# ---------------------------------------------------------------------------


def _train_epoch(
    dataloader,
    model,
    optimizer,
    criterion,
    ema,
    epoch,
    model_name,
    train=False,
    set_name="train",
    aux_weight=0.2,
    use_tta=False,
    pbar=None,
    accum_steps=1,
    uncertainty_weighter=None,
    clip_loss_fn=None,
    clip_weight=0.0,
):
    if train:
        model.train()
    else:
        model.eval()

    os.makedirs(model_name, exist_ok=True)
    prediction_file = os.path.join(
        model_name, f"predictions_{set_name}_epoch_{epoch}.json"
    )
    actions = {}
    loss_total_act = 0.0
    loss_total_sev = 0.0
    loss_total_clip = 0.0
    clip_batches = 0
    n_batches = 0

    criterion_action = criterion["action"]
    criterion_bce = criterion["bce"]

    ctx = torch.no_grad() if not train else torch.enable_grad()

    with ctx:
        for batch in dataloader:
            if len(batch) == 8:
                (
                    targets_sev,
                    targets_act,
                    targets_contact,
                    targets_bodypart,
                    targets_try_to_play,
                    targets_handball,
                    mvclips,
                    action_ids,
                ) = batch
                text_feat = None
            elif len(batch) == 9:
                (
                    targets_sev,
                    targets_act,
                    targets_contact,
                    targets_bodypart,
                    targets_try_to_play,
                    targets_handball,
                    mvclips,
                    action_ids,
                    text_feat,
                ) = batch
            else:
                raise ValueError(f"Unexpected batch size: {len(batch)}")

            targets_sev = targets_sev.cuda()
            targets_act = targets_act.cuda()
            targets_contact = targets_contact.cuda()
            targets_bodypart = targets_bodypart.cuda()
            targets_try_to_play = targets_try_to_play.cuda()
            targets_handball = targets_handball.cuda()
            mvclips = mvclips.cuda().float()
            if text_feat is not None:
                text_feat = text_feat.cuda().float()

            if pbar is not None:
                pbar.update()

            # --- forward ---
            full_out = None
            clip_proj = None
            if not train and use_tta:
                (
                    out_sev,
                    out_act,
                    out_contact,
                    out_bodypart,
                    out_try_to_play,
                    out_handball,
                ) = _run_with_tta(model, mvclips, text_features=text_feat)
                attention = None
            else:
                full_out = model(mvclips, text_features=text_feat)
                out_sev, out_act = full_out[0], full_out[1]
                out_contact, out_bodypart = full_out[2], full_out[3]
                out_try_to_play, out_handball = full_out[4], full_out[5]
                attention = full_out[6]
                clip_proj = full_out[7] if len(full_out) > 7 else None

            if train and n_batches % 100 == 0:
                mvnet = getattr(model, "mvnetwork", None)
                if mvnet is not None:
                    agg = getattr(mvnet, "aggregation_model", None)
                    text_bridge = getattr(agg, "text_bridge", None)
                    if text_bridge is not None:
                        gate_val = torch.sigmoid(text_bridge.gate).item()
                        logging.info(f"  text_bridge gate: {gate_val:.4f}")

            if out_sev.dim() == 1:
                out_sev = out_sev.unsqueeze(0)
            if out_act.dim() == 1:
                out_act = out_act.unsqueeze(0)

            # --- decode predictions ---
            preds_sev = ordinal_predict(out_sev.detach().cpu())
            preds_act = torch.argmax(out_act.detach().cpu(), dim=1)

            preds_sev = ordinal_predict(out_sev.detach().cpu())  # [B]
            preds_act = torch.argmax(out_act.detach().cpu(), dim=-1).reshape(-1)  # [B]

            assert (
                preds_act.max().item() <= 7
            ), f"Invalid action class {preds_act.max().item()}, out_act shape: {out_act.shape}"

            _decode_predictions(preds_sev, preds_act, actions, action_ids)

            # dim guard for batch_size=1
            if out_sev.dim() == 1:
                out_sev = out_sev.unsqueeze(0)
            if out_act.dim() == 1:
                out_act = out_act.unsqueeze(0)

            # --- losses ---
            labels_int = targets_sev.argmax(dim=1)
            pos_weight = None
            if isinstance(criterion, dict):
                pos_weight = criterion.get("ordinal_pos_weight")
            loss_sev = ordinal_loss(out_sev, labels_int, pos_weight=pos_weight)
            loss_act = criterion_action(out_act, targets_act)

            loss_aux = (
                criterion_bce(out_contact, targets_contact)
                + criterion_bce(out_bodypart, targets_bodypart)
                + criterion_bce(out_try_to_play, targets_try_to_play)
                + criterion_bce(out_handball, targets_handball)
            ) / 4.0

            # Temporal entropy regularization — only during training,
            # only if the aggregator returns temporal weights (TransformerAggregate)
            attention = full_out[6] if full_out is not None else None
            loss_temporal = (
                temporal_entropy_loss(attention) if train else torch.tensor(0.0)
            )

            loss_clip = None
            if (
                train
                and clip_loss_fn is not None
                and clip_weight > 0
                and clip_proj is not None
            ):
                loss_clip = clip_loss_fn(clip_proj, labels_int)

            if uncertainty_weighter is not None:
                total_loss = (
                    uncertainty_weighter([loss_sev, loss_act])
                    + aux_weight * loss_aux
                    + 0.01 * loss_temporal
                )
            else:
                total_loss = (
                    loss_sev + loss_act + aux_weight * loss_aux + 0.01 * loss_temporal
                )

            if loss_clip is not None:
                total_loss = total_loss + clip_weight * loss_clip

            if train:
                (total_loss / accum_steps).backward()
                if (n_batches + 1) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    ema.update()

            loss_total_sev += loss_sev.item()
            loss_total_act += loss_act.item()
            if loss_clip is not None:
                loss_total_clip += loss_clip.item()
                clip_batches += 1
            n_batches += 1

    gc.collect()
    torch.cuda.empty_cache()

    data = {"Set": set_name, "Actions": actions}
    with open(prediction_file, "w") as f:
        json.dump(data, f)

    avg_sev = loss_total_sev / max(n_batches, 1)
    avg_act = loss_total_act / max(n_batches, 1)
    if train and clip_batches > 0:
        avg_clip = loss_total_clip / clip_batches
        logging.info(f"  clip_loss={avg_clip:.4f} (weight={clip_weight:.3f})")
    return prediction_file, avg_act, avg_sev


# ---------------------------------------------------------------------------
# Inference-only evaluation (no labels required)
# ---------------------------------------------------------------------------


def evaluation(dataloader, model, ema=None, set_name="test", use_tta=True):
    if ema is not None:
        ema.apply_shadow()

    model.eval()
    actions = {}

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

            preds_sev = ordinal_predict(out_sev.cpu())
            preds_act = torch.argmax(out_act.cpu(), dim=1)
            _decode_predictions(preds_sev, preds_act, actions, action_ids)

    if ema is not None:
        ema.restore()

    gc.collect()
    torch.cuda.empty_cache()

    prediction_file = f"predictions_{set_name}.json"
    with open(prediction_file, "w") as f:
        json.dump({"Set": set_name, "Actions": actions}, f)
    return prediction_file
