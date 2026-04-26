#!/usr/bin/env python3
"""
Stage-2 feature-bank training for VARS models.

Step 1  (--mode extract):
  Load a checkpoint, freeze the backbone+aggregator, run the Train split
  N times with augmentation and Valid/Test once without augmentation, and
  save all pooled feature vectors to an HDF5 file.

Step 2  (--mode train_heads):
  Load the HDF5 bank, train only the inter + fc_* classification heads on
  the cached features.  The backbone never runs during head training, so
  each epoch is ~100x cheaper than full training.

Step 3  (--mode full):
  Both steps sequentially (extract then train_heads).

Example
-------
  # Extract (10 augmented passes over Train):
  python train_stage2.py --mode extract \\
      --checkpoint models/.../best_model.pth.tar \\
      --path /data/SoccerNet --feature_bank feats.h5 --n_passes 10 \\
      --pre_model mvit_v2_s --pooling_type gat --graph_topology fully_connected \\
      --num_views 5 --fps 17 --start_frame 63 --end_frame 87

  # Train heads:
  python train_stage2.py --mode train_heads \\
      --checkpoint models/.../best_model.pth.tar \\
      --path /data/SoccerNet --feature_bank feats.h5 \\
      --LR 1e-3 --max_epochs 30 --patience 6 --model_name VARS_stage2
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
import torchvision.transforms as transforms
from tqdm import tqdm
from SoccerNet.Evaluation.MV_FoulRecognition import evaluate as sn_evaluate

from dataset import MultiViewDataset
from model import MVNetwork, EarlyFusionNetwork, HF_VIDEOMAE_REGISTRY
from train import ordinal_loss, ordinal_predict, EMA, UncertaintyWeighting, _decode_predictions
from torchvision.models.video import (
    MViT_V2_S_Weights, R2Plus1D_18_Weights, R3D_18_Weights,
    MC3_18_Weights, S3D_Weights, MViT_V1_B_Weights,
)

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

_TRANSFORM_MAP = {
    "r3d_18":      R3D_18_Weights.KINETICS400_V1.transforms(),
    "mc3_18":      MC3_18_Weights.KINETICS400_V1.transforms(),
    "r2plus1d_18": R2Plus1D_18_Weights.KINETICS400_V1.transforms(),
    "s3d":         S3D_Weights.KINETICS400_V1.transforms(),
    "mvit_v2_s":   MViT_V2_S_Weights.KINETICS400_V1.transforms(),
    "mvit_v1_b":   MViT_V1_B_Weights.KINETICS400_V1.transforms(),
    **{k: MViT_V2_S_Weights.KINETICS400_V1.transforms() for k in HF_VIDEOMAE_REGISTRY},
}


# ---------------------------------------------------------------------------
# HeadsOnly: classification heads extracted from a full model
# ---------------------------------------------------------------------------

class HeadsOnly(nn.Module):
    """
    Wraps the inter projection + all fc_* heads from a loaded model.
    Source is EarlyFusionNetwork (fusion_mode=True) or MVAggregate
    (accessed via full_model.mvnetwork for MVNetwork).
    """

    def __init__(self, full_model, fusion_mode: bool = False):
        super().__init__()
        src = full_model if fusion_mode else full_model.mvnetwork
        self.inter               = src.inter
        self.fc_ordinal_severity = src.fc_ordinal_severity
        self.fc_action           = src.fc_action
        self.fc_contact          = src.fc_contact
        self.fc_bodypart         = src.fc_bodypart
        self.fc_try_to_play      = src.fc_try_to_play
        self.fc_handball         = src.fc_handball
        self.cascade_severity    = getattr(src, 'cascade_severity', False)

    def forward(self, feat: torch.Tensor):
        inter = self.inter(feat)
        pred_action = self.fc_action(inter)
        if self.cascade_severity:
            pred_sev = self.fc_ordinal_severity(torch.cat([inter, pred_action], dim=-1))
        else:
            pred_sev = self.fc_ordinal_severity(inter)
        return (
            pred_sev,
            pred_action,
            self.fc_contact(inter).squeeze(-1),
            self.fc_bodypart(inter).squeeze(-1),
            self.fc_try_to_play(inter).squeeze(-1),
            self.fc_handball(inter).squeeze(-1),
            None,
        )


# ---------------------------------------------------------------------------
# FeatureBank dataset
# ---------------------------------------------------------------------------

class FeatureBank(Dataset):
    """Load pre-extracted features from HDF5 into memory."""

    def __init__(self, hdf5_path: str, split: str):
        with h5py.File(hdf5_path, 'r') as f:
            g = f[split]
            self.features         = torch.from_numpy(g['features'][:])
            self.targets_sev      = torch.from_numpy(g['targets_sev'][:])
            self.targets_act      = torch.from_numpy(g['targets_act'][:]).long()
            self.targets_contact  = torch.from_numpy(g['targets_contact'][:]).float()
            self.targets_bodypart = torch.from_numpy(g['targets_bodypart'][:]).float()
            self.targets_ttp      = torch.from_numpy(g['targets_try_to_play'][:]).float()
            self.targets_handball = torch.from_numpy(g['targets_handball'][:]).float()
            self.action_ids       = [s.decode() for s in g['action_ids'][:]]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.targets_sev[idx],
            self.targets_act[idx],
            self.targets_contact[idx],
            self.targets_bodypart[idx],
            self.targets_ttp[idx],
            self.targets_handball[idx],
            self.action_ids[idx],
        )


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _pooled(model, mvclips: torch.Tensor, fusion_mode: bool) -> torch.Tensor:
    """Forward through backbone+aggregator only; return [B, D] pooled vector."""
    if fusion_mode:
        return model.backbone(mvclips)
    else:
        pooled, _ = model.mvnetwork.aggregation_model(mvclips)
        return pooled


def _run_loader(model, loader, fusion_mode: bool):
    feats, sevs, acts = [], [], []
    contacts, bodyparts, ttps, handballs, ids = [], [], [], [], []
    for batch in tqdm(loader, leave=False):
        t_sev, t_act, t_contact, t_bodypart, t_ttp, t_handball, mvclips, action_ids = batch
        mvclips = mvclips.cuda().float()
        pooled = _pooled(model, mvclips, fusion_mode)
        feats.append(pooled.cpu().numpy().astype(np.float32))
        sevs.append(t_sev.numpy().astype(np.float32))
        acts.append(t_act.numpy().astype(np.int32))
        contacts.append(t_contact.numpy().astype(np.float32))
        bodyparts.append(t_bodypart.numpy().astype(np.float32))
        ttps.append(t_ttp.numpy().astype(np.float32))
        handballs.append(t_handball.numpy().astype(np.float32))
        ids.extend(list(action_ids))
    return (
        np.concatenate(feats), np.concatenate(sevs), np.concatenate(acts),
        np.concatenate(contacts), np.concatenate(bodyparts),
        np.concatenate(ttps), np.concatenate(handballs), ids,
    )


def _save_group(hf, name, feats, sevs, acts, contacts, bodyparts, ttps, handballs, ids):
    str_dt = h5py.special_dtype(vlen=bytes)
    g = hf.create_group(name)
    g.create_dataset('features',            data=feats,     compression='gzip', compression_opts=4)
    g.create_dataset('targets_sev',         data=sevs)
    g.create_dataset('targets_act',         data=acts)
    g.create_dataset('targets_contact',     data=contacts)
    g.create_dataset('targets_bodypart',    data=bodyparts)
    g.create_dataset('targets_try_to_play', data=ttps)
    g.create_dataset('targets_handball',    data=handballs)
    ids_enc = np.array([s.encode() for s in ids], dtype=object)
    g.create_dataset('action_ids', data=ids_enc, dtype=str_dt)
    logging.info(f"  [{name}] {len(feats)} samples, feature dim={feats.shape[1]}")


# ---------------------------------------------------------------------------
# Stage 1: extract
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract(model, train_loader_aug, val_loader, test_loader,
            hdf5_path: str, fusion_mode: bool, n_passes: int):
    """
    Run backbone+aggregator on every split, save to hdf5_path.
    Train split is run n_passes times (different random augmentations each pass).
    Val and Test are run once without augmentation.
    """
    model.eval()

    with h5py.File(hdf5_path, 'w') as hf:
        for split, loader in [('valid', val_loader), ('test', test_loader)]:
            logging.info(f"Extracting {split} (1 pass, no augmentation)...")
            data = _run_loader(model, loader, fusion_mode)
            _save_group(hf, split, *data)
            gc.collect(); torch.cuda.empty_cache()

        logging.info(f"Extracting train ({n_passes} augmented passes)...")
        all_feats, all_sevs, all_acts = [], [], []
        all_contacts, all_bodyparts, all_ttps, all_handballs, all_ids = [], [], [], [], []
        for i in range(n_passes):
            logging.info(f"  Pass {i + 1}/{n_passes}")
            feats, sevs, acts, contacts, bodyparts, ttps, handballs, ids = \
                _run_loader(model, train_loader_aug, fusion_mode)
            all_feats.append(feats);   all_sevs.append(sevs)
            all_acts.append(acts);     all_contacts.append(contacts)
            all_bodyparts.append(bodyparts); all_ttps.append(ttps)
            all_handballs.append(handballs); all_ids.extend(ids)
            gc.collect(); torch.cuda.empty_cache()

        _save_group(hf, 'train',
                    np.concatenate(all_feats), np.concatenate(all_sevs), np.concatenate(all_acts),
                    np.concatenate(all_contacts), np.concatenate(all_bodyparts),
                    np.concatenate(all_ttps), np.concatenate(all_handballs), all_ids)

    logging.info(f"Feature bank saved to {hdf5_path}")


# ---------------------------------------------------------------------------
# Stage 2: train heads
# ---------------------------------------------------------------------------

def _epoch_heads(heads, loader, optimizer, criterion, ema, train: bool,
                 aux_weight, model_name, set_name, epoch,
                 uncertainty_weighter, accum_steps=1):
    heads.train() if train else heads.eval()
    os.makedirs(model_name, exist_ok=True)
    pred_file = os.path.join(model_name, f"predictions_{set_name}_epoch_{epoch}.json")
    actions = {}
    loss_sev_total = loss_act_total = 0.0
    n = 0

    criterion_action = criterion['action']
    criterion_bce    = criterion['bce']

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for i, batch in enumerate(tqdm(loader, desc=set_name, leave=False)):
            (feat, t_sev, t_act, t_contact, t_bodypart,
             t_ttp, t_handball, action_ids) = batch

            feat       = feat.cuda().float()
            t_sev      = t_sev.cuda()
            t_act      = t_act.cuda()
            t_contact  = t_contact.cuda()
            t_bodypart = t_bodypart.cuda()
            t_ttp      = t_ttp.cuda()
            t_handball = t_handball.cuda()

            out_sev, out_act, out_contact, out_bodypart, out_ttp, out_handball, _ = heads(feat)

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
                    nn.utils.clip_grad_norm_(heads.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    ema.update()

            loss_sev_total += loss_sev.item()
            loss_act_total += loss_act.item()
            n += 1

    with open(pred_file, 'w') as f:
        json.dump({"Set": set_name, "Actions": actions}, f)

    return pred_file, loss_act_total / max(n, 1), loss_sev_total / max(n, 1)


def train_heads(heads, train_dl, val_dl, test_dl,
                optimizer, scheduler, criterion, ema,
                best_model_path, model_name, path_dataset,
                max_epochs, patience, aux_weight, uncertainty_weighter, accum_steps):

    best_val = 0.0
    no_improve = 0

    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch + 1}/{max_epochs}")

        pred_file, _, _ = _epoch_heads(
            heads, train_dl, optimizer, criterion, ema, train=True,
            aux_weight=aux_weight, model_name=model_name,
            set_name='train', epoch=epoch + 1,
            uncertainty_weighter=uncertainty_weighter, accum_steps=accum_steps,
        )
        results = sn_evaluate(os.path.join(path_dataset, 'Train', 'annotations.json'), pred_file)
        print("TRAIN:", results)

        ema.apply_shadow()
        pred_file, _, _ = _epoch_heads(
            heads, val_dl, optimizer, criterion, ema, train=False,
            aux_weight=aux_weight, model_name=model_name,
            set_name='valid', epoch=epoch + 1,
            uncertainty_weighter=uncertainty_weighter,
        )
        ema.restore()
        results = sn_evaluate(os.path.join(path_dataset, 'Valid', 'annotations.json'), pred_file)
        print("VALID:", results)

        val_lb = results.get('leaderboard_value', 0)
        if val_lb > best_val:
            best_val = val_lb
            no_improve = 0
            torch.save({
                'epoch': epoch + 1,
                'state_dict': heads.state_dict(),
                'ema': ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'uw': uncertainty_weighter.state_dict() if uncertainty_weighter else None,
            }, os.path.join(best_model_path, 'best_model_stage2.pth.tar'))
            logging.info(f"New best val LB: {best_val:.4f} (epoch {epoch + 1})")
        else:
            no_improve += 1
            if no_improve >= patience:
                logging.info(f"Early stopping at epoch {epoch + 1} (patience={patience})")
                break

        ema.apply_shadow()
        pred_file, _, _ = _epoch_heads(
            heads, test_dl, optimizer, criterion, ema, train=False,
            aux_weight=aux_weight, model_name=model_name,
            set_name='test', epoch=epoch + 1,
            uncertainty_weighter=uncertainty_weighter,
        )
        ema.restore()
        results = sn_evaluate(os.path.join(path_dataset, 'Test', 'annotations.json'), pred_file)
        print("TEST:", results)

        scheduler.step()
        torch.save({
            'epoch': epoch + 1,
            'state_dict': heads.state_dict(),
            'ema': ema.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'uw': uncertainty_weighter.state_dict() if uncertainty_weighter else None,
        }, os.path.join(best_model_path, f'{epoch + 1}_model_stage2.pth.tar'))

    logging.info(f"Stage-2 complete. Best val LB: {best_val:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(args.model_name, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.model_name, 'stage2.log')),
            logging.StreamHandler(),
        ],
    )

    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"]   = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    # -----------------------------------------------------------------------
    # Compute clip length
    # -----------------------------------------------------------------------
    number_of_frames = int(
        (args.end_frame - args.start_frame) /
        (((args.end_frame - args.start_frame) / 25) * args.fps)
    )

    # -----------------------------------------------------------------------
    # Load model from checkpoint
    # -----------------------------------------------------------------------
    transforms_model = (
        MViT_V2_S_Weights.KINETICS400_V1.transforms()
        if args.fusion_mode
        else _TRANSFORM_MAP.get(args.pre_model, R2Plus1D_18_Weights.KINETICS400_V1.transforms())
    )

    if args.fusion_mode:
        model = EarlyFusionNetwork(
            num_views=args.num_views, T_per_view=number_of_frames,
            cascade_severity=args.cascade_severity,
        ).cuda()
    else:
        model = MVNetwork(
            net_name=args.pre_model, agr_type=args.pooling_type,
            graph_topology=args.graph_topology, cascade_severity=args.cascade_severity,
        ).cuda()

    load = torch.load(args.checkpoint)
    model.load_state_dict(load['state_dict'])
    logging.info(f"Loaded checkpoint from {args.checkpoint} (epoch {load.get('epoch', '?')})")

    # -----------------------------------------------------------------------
    # Stage 1: extract
    # -----------------------------------------------------------------------
    if args.mode in ('extract', 'full'):
        transformAug = transforms.Compose([
            transforms.RandomAffine(degrees=(0, 0), translate=(0.1, 0.1), scale=(0.9, 1)),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.5, saturation=0.5, contrast=0.5),
            transforms.RandomHorizontalFlip(),
        ])

        ds_kwargs = dict(
            path=args.path, start=args.start_frame, end=args.end_frame,
            fps=args.fps, transform_model=transforms_model, fusion_mode=args.fusion_mode,
        )
        ds_train_aug = MultiViewDataset(**ds_kwargs, split='Train',
                                        num_views=args.num_views, transform=transformAug)
        ds_val       = MultiViewDataset(**ds_kwargs, split='Valid', num_views=args.num_views)
        ds_test      = MultiViewDataset(**ds_kwargs, split='Test',  num_views=args.num_views)

        loader_aug = DataLoader(ds_train_aug, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.max_num_worker, pin_memory=True)
        loader_val = DataLoader(ds_val,       batch_size=1, shuffle=False,
                                num_workers=args.max_num_worker, pin_memory=True)
        loader_test= DataLoader(ds_test,      batch_size=1, shuffle=False,
                                num_workers=args.max_num_worker, pin_memory=True)

        extract(model, loader_aug, loader_val, loader_test,
                args.feature_bank, args.fusion_mode, args.n_passes)

        del ds_train_aug, ds_val, ds_test, loader_aug, loader_val, loader_test
        gc.collect(); torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Stage 2: train heads
    # -----------------------------------------------------------------------
    if args.mode in ('train_heads', 'full'):
        heads = HeadsOnly(model, fusion_mode=args.fusion_mode).cuda()

        train_ds = FeatureBank(args.feature_bank, 'train')
        val_ds   = FeatureBank(args.feature_bank, 'valid')
        test_ds  = FeatureBank(args.feature_bank, 'test')

        train_dl = DataLoader(train_ds, batch_size=args.head_batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
        val_dl   = DataLoader(val_ds,   batch_size=256, shuffle=False,
                              num_workers=2, pin_memory=True)
        test_dl  = DataLoader(test_ds,  batch_size=256, shuffle=False,
                              num_workers=2, pin_memory=True)

        logging.info(f"Feature bank: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

        # Inverse-frequency action weights from bank
        act_counts = np.bincount(train_ds.targets_act.numpy(), minlength=8).astype(np.float32)
        act_weights = 1.0 / (act_counts + 1.0)
        act_weights = torch.from_numpy(act_weights / act_weights.sum() * 8).cuda()

        criterion = {
            'action': nn.CrossEntropyLoss(weight=act_weights, label_smoothing=0.1),
            'bce':    nn.BCEWithLogitsLoss(),
        }

        uncertainty_weighter = None
        head_params = list(heads.parameters())
        if args.uncertainty_weighting:
            uncertainty_weighter = UncertaintyWeighting(num_tasks=2).cuda()
            head_params = head_params + list(uncertainty_weighter.parameters())
            logging.info("Uncertainty weighting enabled")

        optimizer = torch.optim.AdamW(
            head_params, lr=args.LR, weight_decay=args.weight_decay,
            betas=(0.9, 0.999), eps=1e-7,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.max_epochs, eta_min=1e-6,
        )
        ema = EMA(heads, decay=args.ema_decay)

        best_model_path = args.model_name
        os.makedirs(best_model_path, exist_ok=True)

        train_heads(
            heads, train_dl, val_dl, test_dl,
            optimizer, scheduler, criterion, ema,
            best_model_path, args.model_name, args.path,
            args.max_epochs, args.patience,
            args.aux_weight, uncertainty_weighter, args.accum_steps,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = ArgumentParser(
        description='VARS Stage-2 feature-bank training',
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--mode', default='full',
                        choices=['extract', 'train_heads', 'full'],
                        help='extract: save features | train_heads: train heads from bank | full: both')

    # Checkpoint & bank
    parser.add_argument('--checkpoint',    required=True, type=str,
                        help='Path to best_model.pth.tar from Stage-1 training')
    parser.add_argument('--feature_bank',  default='features.h5', type=str,
                        help='HDF5 file for storing/loading extracted features')
    parser.add_argument('--n_passes',      default=10, type=int,
                        help='Number of augmented passes over Train for feature extraction')

    # Dataset
    parser.add_argument('--path',          required=True, type=str)
    parser.add_argument('--start_frame',   default=63,   type=int)
    parser.add_argument('--end_frame',     default=87,   type=int)
    parser.add_argument('--fps',           default=17,   type=int)
    parser.add_argument('--num_views',     default=5,    type=int)

    # Model architecture (must match the checkpoint)
    parser.add_argument('--fusion_mode',      action='store_true', default=False)
    parser.add_argument('--pre_model',        default='mvit_v2_s', type=str)
    parser.add_argument('--pooling_type',     default='transformer', type=str)
    parser.add_argument('--graph_topology',   default='structured', type=str)
    parser.add_argument('--cascade_severity', action='store_true', default=False)

    # Head training
    parser.add_argument('--LR',               default=1e-3,  type=float)
    parser.add_argument('--weight_decay',     default=1e-4,  type=float)
    parser.add_argument('--max_epochs',       default=30,    type=int)
    parser.add_argument('--patience',         default=6,     type=int)
    parser.add_argument('--batch_size',       default=16,    type=int,
                        help='Batch size for feature extraction (backbone forward)')
    parser.add_argument('--head_batch_size',  default=256,   type=int,
                        help='Batch size for head training (feature forward only)')
    parser.add_argument('--accum_steps',      default=1,     type=int)
    parser.add_argument('--aux_weight',       default=0.2,   type=float)
    parser.add_argument('--ema_decay',        default=0.999, type=float)
    parser.add_argument('--uncertainty_weighting', action='store_true', default=False)

    # Infra
    parser.add_argument('--model_name',      default='VARS_stage2', type=str)
    parser.add_argument('--GPU',             default=0,    type=int)
    parser.add_argument('--max_num_worker',  default=4,    type=int)

    args = parser.parse_args()
    start = time.time()
    main(args)
    logging.info(f'Total time: {time.time() - start:.1f}s')
