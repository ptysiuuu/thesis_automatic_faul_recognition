"""
train_dejavid_severity.py
=========================
DejaVid-inspired severity classification head operating on TAdaFormer
temporal token sequences [N, 8, 768] extracted from the token bank.

Key idea (from DejaVid, CVPR 2025):
  - Represent each sample as a Temporal Sequence of Embeddings (TSE)
  - Learn per-class centroid TSEs via gradient descent
  - Learn per-timestep, per-feature weights for each class
  - Classify by soft-min DTW distance to class centroids

Adaptation for this project:
  - Severity only (4 classes: No offence, No card, Yellow card, Red card)
  - T=8 tokens (trivially fast, no CUDA kernel needed)
  - Multi-view: tokens are mean-pooled across 5 views → [8, 768]
  - Token bank already exists: temporal_tokens_newparams.h5

Usage:
    python train_dejavid_severity.py \
        --feature_bank temporal_tokens_newparams.h5 \
        --model_name VARS_dejavid_severity \
        --max_epochs 50 \
        --patience 10 \
        --LR 1e-3 \
        --GPU 0

Output:
    - best_dejavid_severity.pth  (best centroid + weights checkpoint)
    - predictions_valid.json     (for ensemble with action model)
    - predictions_test.json
"""

import os
import gc
import json
import logging
import time
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

from config.classes import INVERSE_EVENT_DICTIONARY

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEVERITY_CLASSES = ["No offence", "No card", "Yellow card", "Red card"]
N_SEVERITY = 4
T_TOKENS = 8
FEAT_DIM = 768

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TokenBankDataset(Dataset):
    """
    Loads pre-extracted [8, 768] temporal token sequences from HDF5.
    Targets are severity class integers (0-3).
    """

    def __init__(self, hdf5_path: str, split: str):
        with h5py.File(hdf5_path, "r") as f:
            g = f[split]
            # tokens: [N, 8, 768]
            self.tokens = torch.from_numpy(g["tokens"][:]).float()
            # targets_sev: [N, 4] one-hot → convert to integer
            targets_sev_raw = torch.from_numpy(g["targets_sev"][:]).float()
            self.targets_sev = targets_sev_raw.argmax(dim=1).long()  # [N]
            self.action_ids = [s.decode() for s in g["action_ids"][:]]

        print(f"[TokenBankDataset] {split}: {len(self.tokens)} samples, "
              f"token shape={tuple(self.tokens.shape[1:])}")
        counts = torch.bincount(self.targets_sev, minlength=N_SEVERITY)
        for i, name in enumerate(SEVERITY_CLASSES):
            print(f"  {name}: {counts[i].item()}")

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx], self.targets_sev[idx], self.action_ids[idx]

    def get_balanced_sampler(self):
        counts = torch.bincount(self.targets_sev, minlength=N_SEVERITY).float()
        weights = 1.0 / counts.clamp(min=1)
        sample_weights = weights[self.targets_sev]
        return WeightedRandomSampler(sample_weights, len(sample_weights),
                                     replacement=True)


# ---------------------------------------------------------------------------
# Pure PyTorch DTW (efficient for T=8)
# ---------------------------------------------------------------------------


def dtw_distance(
    weights: torch.Tensor,   # [T_c, D] per-timestep per-feature weights (positive)
    centroid: torch.Tensor,  # [T_c, D] centroid TSE
    sample: torch.Tensor,    # [T_s, D] sample TSE
) -> torch.Tensor:
    """
    Time-weighted DTW distance between centroid and sample.
    No diagonal transitions (DejaVid modification for stability).

    For T=8, the grid is 8x8=64 cells — trivially fast in Python loops.

    Returns scalar distance.
    """
    T_c = centroid.shape[0]
    T_s = sample.shape[0]

    # D[i,j] = min warping cost to align centroid[:i+1] with sample[:j+1]
    INF = torch.tensor(1e9, device=centroid.device)
    D = torch.full((T_c, T_s), float('inf'), device=centroid.device)

    # Pointwise weighted Manhattan distance: sum(w * |c - s|)
    # Precompute all pairwise distances: [T_c, T_s]
    # weights: [T_c, D], centroid: [T_c, D], sample: [T_s, D]
    w = F.softplus(weights)  # ensure positive
    # Expand for broadcasting: [T_c, 1, D] * |[T_c, 1, D] - [1, T_s, D]|
    dist_matrix = (w.unsqueeze(1) * (centroid.unsqueeze(1) - sample.unsqueeze(0)).abs()).sum(-1)
    # dist_matrix: [T_c, T_s]

    # Fill DTW table (no diagonal transitions)
    D[0, 0] = dist_matrix[0, 0]
    for j in range(1, T_s):
        D[0, j] = D[0, j-1] + dist_matrix[0, j]
    for i in range(1, T_c):
        D[i, 0] = D[i-1, 0] + dist_matrix[i, 0]
    for i in range(1, T_c):
        for j in range(1, T_s):
            D[i, j] = torch.minimum(D[i-1, j], D[i, j-1]) + dist_matrix[i, j]

    return D[T_c-1, T_s-1]


def batch_dtw_distances(
    log_weights: torch.Tensor,   # [N_classes, T_c, D]
    centroids: torch.Tensor,     # [N_classes, T_c, D]
    samples: torch.Tensor,       # [B, T_s, D]
) -> torch.Tensor:
    """
    Compute DTW distances from each sample to each class centroid.
    Returns [B, N_classes] distance matrix.

    For T=8 and 4 classes this is 4 * B * 64 operations — very fast.
    """
    B = samples.shape[0]
    N_classes = centroids.shape[0]
    distances = torch.zeros(B, N_classes, device=samples.device)

    for c in range(N_classes):
        for b in range(B):
            distances[b, c] = dtw_distance(
                log_weights[c], centroids[c], samples[b]
            )
    return distances


# ---------------------------------------------------------------------------
# Vectorized DTW (faster alternative using precomputed distance matrix)
# ---------------------------------------------------------------------------


def vectorized_dtw_distance(
    log_weights: torch.Tensor,  # [T_c, D]
    centroid: torch.Tensor,     # [T_c, D]
    samples: torch.Tensor,      # [B, T_s, D]
) -> torch.Tensor:
    """
    Vectorized DTW: compute distances from all B samples to one centroid.
    Returns [B] distances.

    Uses sequential DP over T_c * T_s = 64 steps — fast even without CUDA kernel.
    """
    B, T_s, D = samples.shape
    T_c = centroid.shape[0]

    w = F.softplus(log_weights)  # [T_c, D] positive weights

    # Precompute all pairwise distances: [B, T_c, T_s]
    # centroid: [T_c, D] → [1, T_c, 1, D]
    # samples:  [B, T_s, D] → [B, 1, T_s, D]
    # w: [T_c, D] → [1, T_c, 1, D]
    dist_matrix = (
        w.unsqueeze(0).unsqueeze(2) *
        (centroid.unsqueeze(0).unsqueeze(2) - samples.unsqueeze(1)).abs()
    ).sum(-1)  # [B, T_c, T_s]

    # DTW DP: D_table[B, T_c, T_s]
    INF = 1e9
    D_table = torch.full((B, T_c, T_s), INF, device=samples.device)

    # Initialize borders
    D_table[:, 0, 0] = dist_matrix[:, 0, 0]
    for j in range(1, T_s):
        D_table[:, 0, j] = D_table[:, 0, j-1] + dist_matrix[:, 0, j]
    for i in range(1, T_c):
        D_table[:, i, 0] = D_table[:, i-1, 0] + dist_matrix[:, i, 0]

    # Fill table
    for i in range(1, T_c):
        for j in range(1, T_s):
            D_table[:, i, j] = (
                torch.minimum(D_table[:, i-1, j], D_table[:, i, j-1])
                + dist_matrix[:, i, j]
            )

    return D_table[:, T_c-1, T_s-1]  # [B]


# ---------------------------------------------------------------------------
# DejaVid Severity Head
# ---------------------------------------------------------------------------


class DejaVidSeverityHead(nn.Module):
    """
    DejaVid head for severity classification.

    Learnable parameters:
    - centroids: [N_severity, T_c, D]  — one centroid TSE per class
    - log_weights: [N_severity, T_c, D] — log of temporal feature weights

    Forward: compute soft-min DTW distances → class probabilities.

    T_c = T_s = 8 for our token bank.
    """

    def __init__(
        self,
        n_classes: int = N_SEVERITY,
        T_c: int = T_TOKENS,
        feat_dim: int = FEAT_DIM,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.T_c = T_c
        self.feat_dim = feat_dim
        self.temperature = nn.Parameter(torch.tensor(temperature))

        # Centroids initialized to zero — will be set from training data
        self.centroids = nn.Parameter(
            torch.zeros(n_classes, T_c, feat_dim)
        )
        # Log-weights initialized to zero → softplus(0) = log(2) ≈ 0.693
        # (near-uniform weighting at initialization)
        self.log_weights = nn.Parameter(
            torch.zeros(n_classes, T_c, feat_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T_s, D] token sequences
        Returns: [B, N_classes] logits (negative distances for soft-min)
        """
        B = x.shape[0]
        distances = torch.zeros(B, self.n_classes, device=x.device)

        for c in range(self.n_classes):
            distances[:, c] = vectorized_dtw_distance(
                self.log_weights[c],  # [T_c, D]
                self.centroids[c],    # [T_c, D]
                x,                    # [B, T_s, D]
            )

        # Soft-min: use negative distances as logits, temperature-scaled
        # Lower distance = higher probability
        logits = -distances / self.temperature.abs().clamp(min=0.1)
        return logits

    @torch.no_grad()
    def initialize_centroids(self, train_dataset: TokenBankDataset):
        """
        Initialize centroids as class-mean of training tokens.
        Simple mean is sufficient for T=8 (no need for DBA algorithm).
        """
        print("Initializing centroids from training data...")
        for c in range(self.n_classes):
            mask = train_dataset.targets_sev == c
            if mask.sum() == 0:
                print(f"  WARNING: No training samples for class {c}")
                continue
            class_tokens = train_dataset.tokens[mask]  # [N_c, 8, 768]
            centroid = class_tokens.mean(dim=0)         # [8, 768]
            self.centroids.data[c] = centroid
            print(f"  Class {c} ({SEVERITY_CLASSES[c]}): "
                  f"{mask.sum().item()} samples → centroid initialized")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_epoch(
    model: DejaVidSeverityHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    train: bool,
) -> tuple:
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_targets, all_ids = [], [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for tokens, targets, action_ids in tqdm(loader, leave=False):
            tokens = tokens.to(device)
            targets = targets.to(device)

            logits = model(tokens)  # [B, 4]
            loss = F.cross_entropy(logits, targets)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            preds = logits.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.cpu().tolist())
            all_ids.extend(list(action_ids))
            total_loss += loss.item()

    return total_loss / len(loader), all_preds, all_targets, all_ids


def compute_balanced_accuracy(preds, targets, n_classes=N_SEVERITY):
    preds = np.array(preds)
    targets = np.array(targets)
    per_class = []
    for c in range(n_classes):
        mask = targets == c
        if mask.sum() == 0:
            continue
        acc = (preds[mask] == c).mean()
        per_class.append(acc)
    return np.mean(per_class) * 100.0


def save_predictions(preds, targets, action_ids, split, output_dir):
    """Save predictions in format compatible with SoccerNet evaluator."""
    actions = {}
    for pred, target, aid in zip(preds, targets, action_ids):
        sev_name = SEVERITY_CLASSES[pred]
        if pred == 0:
            offence = "No offence"
            severity = ""
        elif pred == 1:
            offence = "Offence"
            severity = "1.0"
        elif pred == 2:
            offence = "Offence"
            severity = "3.0"
        else:
            offence = "Offence"
            severity = "5.0"
        actions[str(aid)] = {
            "Offence": offence,
            "Severity": severity,
            "Action class": "Tackling",  # placeholder — use action model for this
        }
    pred_file = os.path.join(output_dir, f"predictions_dejavid_{split}.json")
    with open(pred_file, "w") as f:
        json.dump({"Set": split, "Actions": actions}, f)
    return pred_file


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.model_name, "dejavid.log")),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    os.makedirs(args.model_name, exist_ok=True)

    device = f"cuda:{args.GPU}" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = TokenBankDataset(args.feature_bank, "train")
    val_ds   = TokenBankDataset(args.feature_bank, "valid")
    test_ds  = TokenBankDataset(args.feature_bank, "test")

    sampler = train_ds.get_balanced_sampler()
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          sampler=sampler, num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=256, shuffle=False,
                          num_workers=2, pin_memory=True)
    test_dl  = DataLoader(test_ds, batch_size=256, shuffle=False,
                          num_workers=2, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DejaVidSeverityHead(
        n_classes=N_SEVERITY,
        T_c=T_TOKENS,
        feat_dim=FEAT_DIM,
        temperature=args.temperature,
    ).to(device)

    # Initialize centroids from training data mean
    model.initialize_centroids(train_ds)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"DejaVid severity head: {n_params:.3f}M parameters")
    logger.info(f"  Centroids: {model.centroids.shape}")
    logger.info(f"  Log-weights: {model.log_weights.shape}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # Centroids and weights have different optimal learning rates per DejaVid paper
    optimizer = torch.optim.AdamW([
        {"params": [model.centroids],    "lr": args.LR / 3},   # 1/3 of weight LR
        {"params": [model.log_weights],  "lr": args.LR},
        {"params": [model.temperature],  "lr": args.LR / 10},
    ], betas=(0.9, 0.999), weight_decay=0.0)  # no weight decay per DejaVid

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=1e-6
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_ba = 0.0
    no_improve = 0

    for epoch in range(args.max_epochs):
        print(f"\nEpoch {epoch+1}/{args.max_epochs}")

        # Train
        train_loss, train_preds, train_targets, _ = train_epoch(
            model, train_dl, optimizer, device, train=True
        )
        train_ba = compute_balanced_accuracy(train_preds, train_targets)
        logger.info(f"  TRAIN  loss={train_loss:.4f}  sev_BA={train_ba:.2f}%")

        # Validate
        val_loss, val_preds, val_targets, val_ids = train_epoch(
            model, val_dl, optimizer, device, train=False
        )
        val_ba = compute_balanced_accuracy(val_preds, val_targets)
        logger.info(f"  VALID  loss={val_loss:.4f}  sev_BA={val_ba:.2f}%")

        # Save best
        if val_ba > best_val_ba:
            best_val_ba = val_ba
            no_improve = 0
            torch.save({
                "epoch": epoch + 1,
                "centroids": model.centroids.data,
                "log_weights": model.log_weights.data,
                "temperature": model.temperature.data,
                "val_ba": val_ba,
            }, os.path.join(args.model_name, "best_dejavid_severity.pth"))
            logger.info(f"  ✓ New best val severity BA: {best_val_ba:.2f}%")

            # Save valid predictions
            save_predictions(val_preds, val_targets, val_ids,
                             "valid", args.model_name)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Test (every epoch with best weights for tracking)
        best_ckpt = torch.load(
            os.path.join(args.model_name, "best_dejavid_severity.pth")
        )
        model.centroids.data = best_ckpt["centroids"].to(device)
        model.log_weights.data = best_ckpt["log_weights"].to(device)
        model.temperature.data = best_ckpt["temperature"].to(device)

        _, test_preds, test_targets, test_ids = train_epoch(
            model, test_dl, optimizer, device, train=False
        )
        test_ba = compute_balanced_accuracy(test_preds, test_targets)
        logger.info(f"  TEST   sev_BA={test_ba:.2f}%")
        save_predictions(test_preds, test_targets, test_ids,
                         "test", args.model_name)

        scheduler.step()

    logger.info(f"\nBest valid severity BA: {best_val_ba:.2f}%")
    logger.info("Done. Use predictions_dejavid_test.json with your action "
                "model predictions to compute ensemble LB.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DejaVid severity head on TAdaFormer temporal tokens"
    )
    parser.add_argument("--feature_bank", required=True,
                        help="Path to temporal_tokens_newparams.h5")
    parser.add_argument("--model_name", default="VARS_dejavid_severity")
    parser.add_argument("--LR", default=1e-3, type=float)
    parser.add_argument("--max_epochs", default=50, type=int)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--GPU", default=0, type=int)
    args = parser.parse_args()
    main(args)
