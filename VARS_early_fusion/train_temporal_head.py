#!/usr/bin/env python3
"""
train_temporal_head.py
======================
Stage-2 style training using TAdaFormer temporal CLS tokens instead of
pooled features.

Pipeline:
  1. Load frozen TAdaFormer (epoch-16 checkpoint)
  2. Extract [N, 8, 768] token sequences for Train/Valid/Test
  3. Train a tiny temporal transformer head on cached tokens
  4. Evaluate severity BA — expect improvement over pooled-vector baseline

The backbone never runs after extraction. Head training is essentially free.

Usage:
  python train_temporal_head.py \
      --checkpoint models/VARS_tadaformer_b16/.../16_model.pth.tar \
      --path /net/tscratch/people/plgaszos/SoccerNet_Data \
      --feature_bank temporal_features.h5 \
      --mode full

Modes:
  extract      — run backbone, save tokens to HDF5
  train_heads  — train head on cached tokens
  full         — both
"""

import gc
import json
import logging
import os
import time
import warnings
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TADACONV_ROOT = "/net/tscratch/people/plgaszos/sn-mvfoul/TAdaConv"
VARS_ROOT = "/net/tscratch/people/plgaszos/sn-mvfoul/VARS_early_fusion"
HDF5_ROOT = "/net/tscratch/people/plgaszos/SoccerNet_HDF5"

import sys
sys.path.insert(0, VARS_ROOT)
sys.path.insert(0, TADACONV_ROOT)

from data_loader import label2vectormerge, clips2vectormerge
from config.classes import INVERSE_EVENT_DICTIONARY
from train import ordinal_loss, ordinal_predict, EMA, UncertaintyWeighting, _decode_predictions, evaluate


# ---------------------------------------------------------------------------
# Token dataset (reads from HDF5 cache)
# ---------------------------------------------------------------------------

class TokenDataset(Dataset):
    """
    Loads pre-extracted [8, 768] token sequences from HDF5.
    Each sample: tokens [8, 768], sev label [4], act label scalar
    """

    def __init__(self, hdf5_path: str, split: str):
        with h5py.File(hdf5_path, "r") as f:
            g = f[split]
            self.tokens         = torch.from_numpy(g["tokens"][:])          # [N, 8, 768]
            self.targets_sev    = torch.from_numpy(g["targets_sev"][:])     # [N, 4]
            targets_act_raw = torch.from_numpy(g["targets_act"][:])
            if targets_act_raw.dim() == 2:
                self.targets_act = targets_act_raw.argmax(dim=1).long()  # [N]
            else:
                self.targets_act = targets_act_raw.long()  # [N]
            self.targets_contact    = torch.from_numpy(g["targets_contact"][:]).float()
            self.targets_bodypart   = torch.from_numpy(g["targets_bodypart"][:]).float()
            self.targets_ttp        = torch.from_numpy(g["targets_try_to_play"][:]).float()
            self.targets_handball   = torch.from_numpy(g["targets_handball"][:]).float()
            self.action_ids     = [s.decode() for s in g["action_ids"][:]]

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return (
            self.tokens[idx],           # [8, 768]
            self.targets_sev[idx],      # [4]
            self.targets_act[idx],      # scalar
            self.targets_contact[idx],
            self.targets_bodypart[idx],
            self.targets_ttp[idx],
            self.targets_handball[idx],
            self.action_ids[idx],
        )


# ---------------------------------------------------------------------------
# Temporal head architectures
# ---------------------------------------------------------------------------

class TemporalTransformerHead(nn.Module):
    """
    Tiny transformer that operates on [B, 8, 768] token sequences.

    Three outputs:
      - severity  (ordinal, 3 BCE logits → 4 classes)
      - action    (8-class CE)
      - auxiliary (contact, bodypart, try_to_play, handball)

    Architecture:
      LayerNorm → 1-layer TransformerEncoder (4 heads, 512 FFN) →
      CLS-style readout → task heads

    ~2M parameters total. Trains in seconds per epoch.
    """

    def __init__(
        self,
        feat_dim: int = 768,
        num_heads: int = 4,
        ffn_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        cascade_severity: bool = True,
    ):
        super().__init__()
        self.cascade_severity = cascade_severity

        # Input norm
        self.input_norm = nn.LayerNorm(feat_dim)

        # Learnable [CLS] token for readout
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feat_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Learnable positional embeddings for 8 temporal steps + 1 CLS
        self.pos_embed = nn.Parameter(torch.zeros(1, 9, feat_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Shared projection
        self.inter = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, feat_dim),
        )

        # Severity head (ordinal)
        sev_in = feat_dim + 8 if cascade_severity else feat_dim
        self.fc_severity = nn.Sequential(
            nn.LayerNorm(sev_in),
            nn.Dropout(0.3),
            nn.Linear(sev_in, feat_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(feat_dim // 2, 3),
        )

        # Action head
        self.fc_action = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(0.4),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(feat_dim // 2, 8),
        )

        # Auxiliary heads
        self.fc_contact    = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 1))
        self.fc_bodypart   = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 1))
        self.fc_ttp        = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 1))
        self.fc_handball   = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 1))

    def forward(self, tokens: torch.Tensor):
        """
        tokens : [B, 8, 768]
        Returns 7-tuple matching MVNetwork output convention.
        """
        B, T, D = tokens.shape

        # Normalize input tokens
        x = self.input_norm(tokens)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)   # [B, 1, 768]
        x = torch.cat([cls, x], dim=1)            # [B, 9, 768]
        x = x + self.pos_embed                    # add positional embeddings

        # Transformer
        x = self.transformer(x)                   # [B, 9, 768]

        # Readout from CLS position
        cls_out = x[:, 0]                         # [B, 768]
        inter = self.inter(cls_out)               # [B, 768]

        # Task heads
        pred_action = self.fc_action(inter)       # [B, 8]

        if self.cascade_severity:
            sev_in = torch.cat([inter, pred_action.detach()], dim=-1)
            pred_sev = self.fc_severity(sev_in)
        else:
            pred_sev = self.fc_severity(inter)

        return (
            pred_sev,
            pred_action,
            self.fc_contact(inter).squeeze(-1),
            self.fc_bodypart(inter).squeeze(-1),
            self.fc_ttp(inter).squeeze(-1),
            self.fc_handball(inter).squeeze(-1),
            None,
        )


class LinearProbeHead(nn.Module):
    """
    Simple ablation: mean-pool tokens then linear classification.
    No temporal reasoning — equivalent to what pooled-feature Stage-2 does.
    Use as baseline to verify transformer head adds value.
    """

    def __init__(self, feat_dim: int = 768, cascade_severity: bool = True):
        super().__init__()
        self.cascade_severity = cascade_severity

        sev_in = feat_dim + 8 if cascade_severity else feat_dim
        self.fc_severity = nn.Sequential(
            nn.LayerNorm(sev_in),
            nn.Dropout(0.3),
            nn.Linear(sev_in, feat_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(feat_dim // 2, 3),
        )
        self.fc_action = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(0.4),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(feat_dim // 2, 8),
        )
        self.fc_contact  = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 1))
        self.fc_bodypart = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 1))
        self.fc_ttp      = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 1))
        self.fc_handball = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 1))

    def forward(self, tokens: torch.Tensor):
        x = tokens.mean(dim=1)   # [B, 768] — mean pool over time
        pred_action = self.fc_action(x)
        if self.cascade_severity:
            sev_in = torch.cat([x, pred_action.detach()], dim=-1)
            pred_sev = self.fc_severity(sev_in)
        else:
            pred_sev = self.fc_severity(x)
        return (
            pred_sev,
            pred_action,
            self.fc_contact(x).squeeze(-1),
            self.fc_bodypart(x).squeeze(-1),
            self.fc_ttp(x).squeeze(-1),
            self.fc_handball(x).squeeze(-1),
            None,
        )


class MaxTokenHead(nn.Module):
    """
    Ablation: take the max-norm token (peak moment) instead of mean.
    Tests whether contact-moment token is more informative than average.
    """

    def __init__(self, feat_dim: int = 768, cascade_severity: bool = True):
        super().__init__()
        self.cascade_severity = cascade_severity

        sev_in = feat_dim + 8 if cascade_severity else feat_dim
        self.fc_severity = nn.Sequential(
            nn.LayerNorm(sev_in),
            nn.Dropout(0.3),
            nn.Linear(sev_in, feat_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(feat_dim // 2, 3),
        )
        self.fc_action = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(0.4),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(feat_dim // 2, 8),
        )
        self.fc_contact  = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 1))
        self.fc_bodypart = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 1))
        self.fc_ttp      = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 1))
        self.fc_handball = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 1))

    def forward(self, tokens: torch.Tensor):
        # Take token with maximum L2 norm — the "most activated" temporal step
        norms = tokens.norm(dim=-1)          # [B, 8]
        idx = norms.argmax(dim=-1)           # [B]
        x = tokens[torch.arange(tokens.size(0)), idx]   # [B, 768]
        pred_action = self.fc_action(x)
        if self.cascade_severity:
            sev_in = torch.cat([x, pred_action.detach()], dim=-1)
            pred_sev = self.fc_severity(sev_in)
        else:
            pred_sev = self.fc_severity(x)
        return (
            pred_sev,
            pred_action,
            self.fc_contact(x).squeeze(-1),
            self.fc_bodypart(x).squeeze(-1),
            self.fc_ttp(x).squeeze(-1),
            self.fc_handball(x).squeeze(-1),
            None,
        )


# ---------------------------------------------------------------------------
# Token extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_tokens(checkpoint_path, dataset_path, hdf5_path, n_passes,
                   fps, start_frame, end_frame, num_views, max_workers):
    """
    Run frozen TAdaFormer on all splits and save [N, 8, 768] tokens.
    Train split is run n_passes times with augmentation for diversity.
    """
    import torchvision.transforms as transforms
    from dataset import MultiViewDataset
    from tadaformer_backbone import TAdaFormerBackbone
    from main import TAdaFormerTransform

    logging.info("Loading TAdaFormer backbone...")
    backbone = TAdaFormerBackbone(
        checkpoint_path=checkpoint_path,
        num_frames=16,
        apply_renormalize=False,
    ).cuda()
    backbone.eval()

    # Load epoch from checkpoint for logging
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    epoch = ckpt.get("epoch", "?")
    logging.info(f"Loaded checkpoint epoch {epoch}")

    # Backbone weights already loaded via checkpoint_path in TAdaFormerBackbone.__init__
    # The MVNetwork state_dict keys don't map cleanly to bare backbone — skip.

    transform_model = TAdaFormerTransform(size=224)

    augmentation = transforms.Compose([
        transforms.RandomAffine(degrees=(0, 0), translate=(0.1, 0.1), scale=(0.9, 1)),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.5, saturation=0.5, contrast=0.5),
        transforms.RandomHorizontalFlip(),
    ])

    ds_kwargs = dict(
        path=dataset_path,
        start=start_frame,
        end=end_frame,
        fps=fps,
        transform_model=transform_model,
        fusion_mode=False,
    )

    def run_loader(loader):
        all_tokens, all_sevs, all_acts = [], [], []
        all_contacts, all_bodyparts, all_ttps, all_handballs, all_ids = [], [], [], [], []

        for batch in tqdm(loader, leave=False):
            (t_sev, t_act, t_contact, t_bodypart, t_ttp,
             t_handball, mvclips, action_ids) = batch

            B, V = mvclips.shape[:2]
            mvclips = mvclips.cuda().float()

            # Flatten views into batch dim for backbone
            clips_flat = mvclips.reshape(B * V, *mvclips.shape[2:])  # [B*V, T, C, H, W]
            tokens_flat = backbone(clips_flat, return_tokens=True)    # [B*V, 8, 768]
            T_out, D = tokens_flat.shape[1], tokens_flat.shape[2]
            tokens_per_sample = tokens_flat.reshape(B, V, T_out, D)  # [B, V, 8, 768]

            # Mean pool across valid views → [B, 8, 768]
            view_mask = mvclips.abs().sum(dim=tuple(range(2, mvclips.dim()))) == 0  # [B, V]
            mask_4d = view_mask.unsqueeze(-1).unsqueeze(-1)           # [B, V, 1, 1]
            tokens_masked = tokens_per_sample.masked_fill(mask_4d, 0.0)
            valid_counts = (~view_mask).float().sum(dim=1)            # [B]
            valid_counts = valid_counts.clamp(min=1).view(B, 1, 1)   # [B, 1, 1]
            sample_tokens = tokens_masked.sum(dim=1) / valid_counts   # [B, T', D]

            all_tokens.append(sample_tokens.cpu().numpy().astype(np.float32))
            all_sevs.append(t_sev.reshape(-1, 4).numpy().astype(np.float32))
            all_acts.append(t_act.reshape(-1, 8).numpy().astype(np.float32))
            all_contacts.append(t_contact.numpy().astype(np.float32))
            all_bodyparts.append(t_bodypart.numpy().astype(np.float32))
            all_ttps.append(t_ttp.numpy().astype(np.float32))
            all_handballs.append(t_handball.numpy().astype(np.float32))
            all_ids.extend(list(action_ids))

        return (np.concatenate(all_tokens), np.concatenate(all_sevs),
                np.concatenate(all_acts), np.concatenate(all_contacts),
                np.concatenate(all_bodyparts), np.concatenate(all_ttps),
                np.concatenate(all_handballs), all_ids)

    def save_group(hf, name, tokens, sevs, acts, contacts, bodyparts, ttps, handballs, ids):
        str_dt = h5py.special_dtype(vlen=bytes)
        g = hf.create_group(name)
        g.create_dataset("tokens", data=tokens, compression="gzip", compression_opts=4)
        g.create_dataset("targets_sev", data=sevs)
        g.create_dataset("targets_act", data=acts)
        g.create_dataset("targets_contact", data=contacts)
        g.create_dataset("targets_bodypart", data=bodyparts)
        g.create_dataset("targets_try_to_play", data=ttps)
        g.create_dataset("targets_handball", data=handballs)
        ids_enc = np.array([s.encode() for s in ids], dtype=object)
        g.create_dataset("action_ids", data=ids_enc, dtype=str_dt)
        logging.info(f"  [{name}] {len(tokens)} samples, token shape={tokens.shape[1:]}")

    with h5py.File(hdf5_path, "w") as hf:
        for split in ["Valid", "Test"]:
            logging.info(f"Extracting {split} (no augmentation)...")
            ds = MultiViewDataset(**ds_kwargs, split=split, num_views=num_views)
            loader = DataLoader(ds, batch_size=16, shuffle=False,
                                num_workers=max_workers, pin_memory=True)
            data = run_loader(loader)
            save_group(hf, split.lower(), *data)
            del ds, loader
            gc.collect(); torch.cuda.empty_cache()

        logging.info(f"Extracting Train ({n_passes} augmented passes)...")
        all_tokens, all_sevs, all_acts = [], [], []
        all_contacts, all_bodyparts, all_ttps, all_handballs, all_ids = [], [], [], [], []

        for i in range(n_passes):
            logging.info(f"  Pass {i+1}/{n_passes}")
            ds = MultiViewDataset(**ds_kwargs, split="Train",
                                  num_views=num_views, transform=augmentation)
            loader = DataLoader(ds, batch_size=16, shuffle=True,
                                num_workers=max_workers, pin_memory=True)
            tokens, sevs, acts, contacts, bodyparts, ttps, handballs, ids = run_loader(loader)
            all_tokens.append(tokens); all_sevs.append(sevs)
            all_acts.append(acts); all_contacts.append(contacts)
            all_bodyparts.append(bodyparts); all_ttps.append(ttps)
            all_handballs.append(handballs); all_ids.extend(ids)
            del ds, loader
            gc.collect(); torch.cuda.empty_cache()

        save_group(hf, "train",
                   np.concatenate(all_tokens), np.concatenate(all_sevs),
                   np.concatenate(all_acts), np.concatenate(all_contacts),
                   np.concatenate(all_bodyparts), np.concatenate(all_ttps),
                   np.concatenate(all_handballs), all_ids)

    logging.info(f"Token bank saved to {hdf5_path}")


# ---------------------------------------------------------------------------
# Head training loop
# ---------------------------------------------------------------------------

def train_one_epoch(head, loader, optimizer, criterion, ema,
                    model_name, set_name, epoch, aux_weight,
                    uncertainty_weighter, accum_steps, train):
    head.train() if train else head.eval()
    os.makedirs(model_name, exist_ok=True)
    pred_file = os.path.join(model_name, f"predictions_{set_name}_epoch_{epoch}.json")
    actions = {}
    loss_sev_total = loss_act_total = 0.0
    n = 0

    criterion_action = criterion["action"]
    criterion_bce = criterion["bce"]

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for i, batch in enumerate(tqdm(loader, desc=set_name, leave=False)):
            (tokens, t_sev, t_act, t_contact, t_bodypart,
             t_ttp, t_handball, action_ids) = batch

            tokens     = tokens.cuda().float()
            t_sev      = t_sev.cuda()
            t_act      = t_act.cuda()
            t_contact  = t_contact.cuda()
            t_bodypart = t_bodypart.cuda()
            t_ttp      = t_ttp.cuda()
            t_handball = t_handball.cuda()

            out_sev, out_act, out_contact, out_bodypart, out_ttp, out_handball, _ = head(tokens)

            if out_sev.dim() == 1: out_sev = out_sev.unsqueeze(0)
            if out_act.dim() == 1: out_act = out_act.unsqueeze(0)

            preds_sev = ordinal_predict(out_sev.detach().cpu())
            preds_act = torch.argmax(out_act.detach().cpu(), dim=-1)
            _decode_predictions(preds_sev, preds_act, actions, action_ids)

            labels_int = t_sev.argmax(dim=1)
            loss_sev = ordinal_loss(out_sev, labels_int)
            loss_act = criterion_action(out_act, t_act)
            loss_aux = (
                criterion_bce(out_contact, t_contact) +
                criterion_bce(out_bodypart, t_bodypart) +
                criterion_bce(out_ttp, t_ttp) +
                criterion_bce(out_handball, t_handball)
            ) / 4.0

            if uncertainty_weighter is not None:
                total_loss = uncertainty_weighter([loss_sev, loss_act]) + aux_weight * loss_aux
            else:
                total_loss = loss_sev + loss_act + aux_weight * loss_aux

            if train:
                (total_loss / accum_steps).backward()
                if (i + 1) % accum_steps == 0:
                    nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    ema.update()

            loss_sev_total += loss_sev.item()
            loss_act_total += loss_act.item()
            n += 1

    with open(pred_file, "w") as f:
        json.dump({"Set": set_name, "Actions": actions}, f)

    return pred_file, loss_act_total / max(n, 1), loss_sev_total / max(n, 1)


def train_head(head, train_dl, val_dl, test_dl, optimizer, scheduler,
               criterion, ema, best_model_path, model_name, dataset_path,
               max_epochs, patience, aux_weight, uncertainty_weighter,
               accum_steps, head_name):

    best_val = 0.0
    no_improve = 0

    for epoch in range(max_epochs):
        print(f"\n[{head_name}] Epoch {epoch+1}/{max_epochs}")

        pred_file, _, _ = train_one_epoch(
            head, train_dl, optimizer, criterion, ema,
            model_name, "train", epoch+1, aux_weight,
            uncertainty_weighter, accum_steps, train=True)
        results = evaluate(
            os.path.join(dataset_path, "Train", "annotations.json"), pred_file)
        print(f"  TRAIN: {results}")

        ema.apply_shadow()
        pred_file, _, _ = train_one_epoch(
            head, val_dl, optimizer, criterion, ema,
            model_name, "valid", epoch+1, aux_weight,
            uncertainty_weighter, accum_steps=1, train=False)
        ema.restore()
        results = evaluate(
            os.path.join(dataset_path, "Valid", "annotations.json"), pred_file)
        print(f"  VALID: {results}")

        val_lb = results.get("leaderboard_value", 0)
        if val_lb > best_val:
            best_val = val_lb
            no_improve = 0
            torch.save({
                "epoch": epoch + 1,
                "state_dict": head.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "uw": uncertainty_weighter.state_dict() if uncertainty_weighter else None,
            }, os.path.join(best_model_path, f"best_{head_name}.pth.tar"))
            logging.info(f"[{head_name}] New best val LB: {best_val:.4f} (epoch {epoch+1})")
        else:
            no_improve += 1
            if no_improve >= patience:
                logging.info(f"[{head_name}] Early stopping at epoch {epoch+1}")
                break

        ema.apply_shadow()
        pred_file, _, _ = train_one_epoch(
            head, test_dl, optimizer, criterion, ema,
            model_name, "test", epoch+1, aux_weight,
            uncertainty_weighter, accum_steps=1, train=False)
        ema.restore()
        results = evaluate(
            os.path.join(dataset_path, "Test", "annotations.json"), pred_file)
        print(f"  TEST:  {results}")

        scheduler.step()

    logging.info(f"[{head_name}] Best val LB: {best_val:.4f}")
    return best_val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(args.model_name, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.model_name, "temporal_head.log")),
            logging.StreamHandler(),
        ],
    )

    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    # ---- Stage 1: extract tokens ----------------------------------------
    if args.mode in ("extract", "full"):
        extract_tokens(
            checkpoint_path=args.checkpoint,
            dataset_path=args.path,
            hdf5_path=args.feature_bank,
            n_passes=args.n_passes,
            fps=args.fps,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            num_views=args.num_views,
            max_workers=args.max_num_worker,
        )

    # ---- Stage 2: train heads -------------------------------------------
    if args.mode in ("train_heads", "full"):
        train_ds = TokenDataset(args.feature_bank, "train")
        val_ds   = TokenDataset(args.feature_bank, "valid")
        test_ds  = TokenDataset(args.feature_bank, "test")

        logging.info(f"Token bank: train={len(train_ds)}, "
                     f"val={len(val_ds)}, test={len(test_ds)}")
        logging.info(f"Token shape: {train_ds.tokens.shape}")

        # Class-balanced sampler for severity
        from torch.utils.data import WeightedRandomSampler
        sev_classes = train_ds.targets_sev.float().argmax(dim=1)
        counts = torch.bincount(sev_classes)
        weights = 1.0 / counts.float().clamp(min=1)
        sample_weights = weights[sev_classes]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True)
        val_dl   = DataLoader(val_ds, batch_size=256, shuffle=False,
                              num_workers=2, pin_memory=True)
        test_dl  = DataLoader(test_ds, batch_size=256, shuffle=False,
                              num_workers=2, pin_memory=True)

        # Action class weights
        act_counts = torch.bincount(train_ds.targets_act.long(), minlength=8).float()
        act_weights = (1.0 / act_counts.clamp(min=1))
        act_weights = (act_weights / act_weights.sum() * 8).cuda()

        criterion = {
            "action": nn.CrossEntropyLoss(weight=act_weights, label_smoothing=0.1),
            "bce":    nn.BCEWithLogitsLoss(),
        }

        # Three architectures to compare
        heads_to_train = {
            "transformer": TemporalTransformerHead(
                feat_dim=768, num_heads=4, ffn_dim=512,
                num_layers=args.num_layers, dropout=0.1,
                cascade_severity=args.cascade_severity,
            ),
            "linear_probe": LinearProbeHead(
                feat_dim=768, cascade_severity=args.cascade_severity,
            ),
            "max_token": MaxTokenHead(
                feat_dim=768, cascade_severity=args.cascade_severity,
            ),
        }

        results_summary = {}

        for head_name, head in heads_to_train.items():
            logging.info(f"\n{'='*60}")
            logging.info(f"Training head: {head_name}")
            n_params = sum(p.numel() for p in head.parameters()) / 1e6
            logging.info(f"Parameters: {n_params:.2f}M")

            head = head.cuda()
            head_params = list(head.parameters())

            uncertainty_weighter = None
            if args.uncertainty_weighting:
                uncertainty_weighter = UncertaintyWeighting(num_tasks=2).cuda()
                head_params = head_params + list(uncertainty_weighter.parameters())

            optimizer = torch.optim.AdamW(
                head_params, lr=args.LR,
                weight_decay=args.weight_decay,
                betas=(0.9, 0.999), eps=1e-7,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.max_epochs, eta_min=1e-6,
            )
            ema = EMA(head, decay=args.ema_decay)

            best_lb = train_head(
                head=head,
                train_dl=train_dl,
                val_dl=val_dl,
                test_dl=test_dl,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                ema=ema,
                best_model_path=args.model_name,
                model_name=args.model_name,
                dataset_path=args.path,
                max_epochs=args.max_epochs,
                patience=args.patience,
                aux_weight=args.aux_weight,
                uncertainty_weighter=uncertainty_weighter,
                accum_steps=args.accum_steps,
                head_name=head_name,
            )
            results_summary[head_name] = best_lb

            del head, optimizer, scheduler, ema
            gc.collect(); torch.cuda.empty_cache()

        logging.info("\n" + "="*60)
        logging.info("FINAL SUMMARY")
        for name, lb in results_summary.items():
            logging.info(f"  {name:20s}: {lb:.4f} LB")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Temporal token head training on TAdaFormer features",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--mode", default="full",
                        choices=["extract", "train_heads", "full"])
    parser.add_argument("--checkpoint", required=True,
                        help="Path to fine-tuned TAdaFormer checkpoint (e.g. 16_model.pth.tar)")
    parser.add_argument("--feature_bank", default="temporal_tokens.h5")
    parser.add_argument("--n_passes", default=10, type=int,
                        help="Augmented passes over Train for extraction")

    # Dataset
    parser.add_argument("--path",        required=True)
    parser.add_argument("--start_frame", default=63,  type=int)
    parser.add_argument("--end_frame",   default=87,  type=int)
    parser.add_argument("--fps",         default=17,  type=int)
    parser.add_argument("--num_views",   default=5,   type=int)

    # Head architecture
    parser.add_argument("--num_layers",       default=2,    type=int)
    parser.add_argument("--cascade_severity", action="store_true", default=True)

    # Training
    parser.add_argument("--LR",               default=1e-3,  type=float)
    parser.add_argument("--weight_decay",     default=1e-4,  type=float)
    parser.add_argument("--max_epochs",       default=50,    type=int)
    parser.add_argument("--patience",         default=10,    type=int)
    parser.add_argument("--batch_size",       default=128,   type=int)
    parser.add_argument("--accum_steps",      default=1,     type=int)
    parser.add_argument("--aux_weight",       default=0.2,   type=float)
    parser.add_argument("--ema_decay",        default=0.999, type=float)
    parser.add_argument("--uncertainty_weighting", action="store_true", default=False)

    # Infra
    parser.add_argument("--model_name",     default="VARS_temporal_head")
    parser.add_argument("--GPU",            default=0,  type=int)
    parser.add_argument("--max_num_worker", default=8,  type=int)

    args = parser.parse_args()
    t0 = time.time()
    main(args)
    logging.info(f"Total time: {time.time()-t0:.1f}s")
