"""
visualize_temporal.py
=====================
After training, shows which frames the temporal localizer focuses on
for each foul type. Validates that the model learned meaningful
contact-moment localization without any temporal supervision.

Usage:
  python visualize_temporal.py \
    --checkpoint models/VARS_transformer_sev/.../best_model.pth.tar \
    --hdf5_path /net/tscratch/people/plgaszos/SoccerNet_HDF5/Valid.hdf5 \
    --annotations /net/tscratch/people/plgaszos/SoccerNet_Data/Valid/annotations.json \
    --output_dir temporal_viz \
    --n_samples 5
"""

import os
import json
import torch
import h5py
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from model import MVNetwork
from dataset import MultiViewDataset
from torch.utils.data import DataLoader
from torchvision.models.video import MViT_V2_S_Weights
from config.classes import INVERSE_EVENT_DICTIONARY

ACTION_NAMES = [
    "Tackling",
    "Standing tackling",
    "High leg",
    "Holding",
    "Pushing",
    "Elbowing",
    "Challenge",
    "Dive",
]
SEVERITY_NAMES = ["No offence", "No card", "Yellow card", "Red card"]


def extract_temporal_weights(model, mvclips):
    """
    Run forward pass and extract temporal attention weights.
    TransformerAggregate now returns weights as attention[6].
    """
    model.eval()
    with torch.no_grad():
        out = model(mvclips)
    # out[6] = attention = temporal_weights [B, V, T']
    temporal_weights = out[6]
    pred_sev = out[0]
    pred_act = out[1]
    return temporal_weights, pred_sev, pred_act


def plot_action_temporal_patterns(
    weights_by_action: dict,
    output_dir: str,
):
    """
    For each action class, plot the mean temporal attention profile
    averaged across all samples of that class.

    This shows whether different foul types focus on different moments.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for action_idx, action_name in enumerate(ACTION_NAMES):
        ax = axes[action_idx]
        if (
            action_name not in weights_by_action
            or len(weights_by_action[action_name]) == 0
        ):
            ax.set_title(f"{action_name}\n(no samples)")
            ax.set_visible(False)
            continue

        # Stack all weights for this action: list of [V, T'] → [N, V, T']
        all_weights = np.stack(weights_by_action[action_name])

        # Mean over samples and views: [T']
        mean_profile = all_weights.mean(axis=(0, 1))
        std_profile = all_weights.std(axis=(0, 1))
        T = len(mean_profile)

        ax.bar(range(T), mean_profile, alpha=0.7, color=f"C{action_idx}")
        ax.errorbar(
            range(T),
            mean_profile,
            yerr=std_profile,
            fmt="none",
            color="black",
            alpha=0.5,
        )
        ax.axhline(1.0 / T, color="red", linestyle="--", alpha=0.5, label="uniform")
        ax.set_title(f"{action_name}\n(n={len(weights_by_action[action_name])})")
        ax.set_xlabel("Frame index (T'=8)")
        ax.set_ylabel("Attention weight")
        ax.set_xticks(range(T))
        ax.set_xticklabels([f"t{i}" for i in range(T)])

    fig.suptitle(
        "Mean temporal attention per foul type\n"
        "(red dashed = uniform baseline, peak = contact moment)",
        fontsize=14,
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "temporal_profiles_by_action.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_sample_visualization(
    frames_np: np.ndarray,
    weights: np.ndarray,
    pred_action: int,
    true_action: int,
    pred_severity: int,
    true_severity: int,
    output_path: str,
):
    """
    For a single sample, show frames with temporal attention overlaid.
    frames_np : [V, T, H, W, C] uint8
    weights   : [V, T'] float
    """
    V, T_feat = weights.shape
    T_frames = frames_np.shape[1]

    # Map T' attention steps to T frame indices
    frame_indices = np.linspace(0, T_frames - 1, T_feat, dtype=int)

    fig = plt.figure(figsize=(T_feat * 2, V * 2.5))
    gs = gridspec.GridSpec(V, T_feat, figure=fig)

    for v in range(V):
        for t_feat in range(T_feat):
            ax = fig.add_subplot(gs[v, t_feat])
            t_frame = frame_indices[t_feat]
            frame = frames_np[v, t_frame]  # [H, W, C]
            ax.imshow(frame)

            # Overlay attention weight as border color and title
            w = weights[v, t_feat]
            border_color = plt.cm.hot(w)
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color[:3])
                spine.set_linewidth(4)

            if v == 0:
                label = "Live" if v == 0 else f"Replay {v}"
                ax.set_title(f"t{t_feat}\nw={w:.2f}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])

        # View label on left
        ax = fig.add_subplot(gs[v, 0])
        view_label = "Live" if v == 0 else f"Replay {v}"
        ax.set_ylabel(view_label, fontsize=9)

    correct_a = "✓" if pred_action == true_action else "✗"
    correct_s = "✓" if pred_severity == true_severity else "✗"
    fig.suptitle(
        f"Action: {ACTION_NAMES[pred_action]} {correct_a} "
        f"(true: {ACTION_NAMES[true_action]})  |  "
        f"Severity: {SEVERITY_NAMES[pred_severity]} {correct_s} "
        f"(true: {SEVERITY_NAMES[true_severity]})",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    transforms_model = MViT_V2_S_Weights.KINETICS400_V1.transforms()
    model = MVNetwork(
        net_name="mvit_v2_s",
        agr_type="transformer",
        cascade_severity=False,
    ).cuda()

    load = torch.load(args.checkpoint)
    model.load_state_dict(load["state_dict"], strict=False)
    model.eval()
    print(f"Loaded checkpoint from {args.checkpoint}")

    # Dataset
    dataset = MultiViewDataset(
        path=args.annotations.replace("/annotations.json", "/.."),
        start=63,
        end=87,
        fps=17,
        split="Valid",
        num_views=5,
        transform_model=transforms_model,
        fusion_mode=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # Collect temporal weights per action class
    weights_by_action = {name: [] for name in ACTION_NAMES}
    n_visualized = 0

    with h5py.File(args.hdf5_path, "r", swmr=True) as hdf5:
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.max_batches:
                break

            targets_sev, targets_act, _, _, _, _, mvclips, action_ids = batch
            mvclips = mvclips.cuda().float()

            temporal_weights, pred_sev_logits, pred_act_logits = (
                extract_temporal_weights(model, mvclips)
            )

            if temporal_weights is None:
                print(
                    "Warning: model returned no temporal weights. "
                    "Make sure TransformerAggregate was modified correctly."
                )
                break

            # Decode predictions
            from train import ordinal_predict

            pred_sev = ordinal_predict(pred_sev_logits.cpu()).item()
            pred_act = pred_act_logits.cpu().argmax(dim=1).item()
            true_act = targets_act[0].argmax().item()
            true_sev = targets_sev[0].argmax().item()

            action_name = ACTION_NAMES[true_act]
            w_np = temporal_weights[0].cpu().numpy()  # [V, T']
            weights_by_action[action_name].append(w_np)

            # Save frame-level visualization for first N_SAMPLES
            if n_visualized < args.n_samples:
                action_id = action_ids[0]
                action_key = f"action_{action_id}"

                # Load raw frames for visualization
                frames_list = []
                for clip_name in [f"clip_{i}" for i in range(5)]:
                    key = f"{action_key}/{clip_name}"
                    if key in hdf5:
                        frames = hdf5[key][:]  # [T, H, W, C]
                        frames_list.append(frames)

                if frames_list:
                    # Pad to 5 views
                    while len(frames_list) < 5:
                        frames_list.append(np.zeros_like(frames_list[0]))
                    frames_np = np.stack(frames_list[:5])  # [V, T, H, W, C]

                    out_path = str(
                        output_dir / f"sample_{batch_idx:04d}_{action_name}.png"
                    )
                    plot_sample_visualization(
                        frames_np=frames_np,
                        weights=w_np[: frames_np.shape[0]],
                        pred_action=pred_act,
                        true_action=true_act,
                        pred_severity=pred_sev,
                        true_severity=true_sev,
                        output_path=out_path,
                    )
                    print(f"  Saved sample viz: {out_path}")
                    n_visualized += 1

    # Plot aggregate profiles
    plot_action_temporal_patterns(weights_by_action, str(output_dir))
    print(f"\nAll visualizations saved to {output_dir}/")
    print("Look for peaked attention profiles (high weight on 1-2 frames)")
    print("vs uniform profiles (equal weight across all frames).")
    print("Peaked = model learned to localize the contact moment.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hdf5_path", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output_dir", default="temporal_viz")
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="Number of individual sample visualizations to save",
    )
    parser.add_argument(
        "--max_batches", type=int, default=321, help="Max validation batches to process"
    )
    args = parser.parse_args()
    main(args)
