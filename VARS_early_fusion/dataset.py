from torch.utils.data import Dataset, WeightedRandomSampler
import torch
import random
from data_loader import label2vectormerge, clips2vectormerge
from torchvision.io.video import read_video
import warnings
import logging
import h5py
import os
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("torchvision").setLevel(logging.ERROR)

HDF5_ROOT = "/net/tscratch/people/plgaszos/SoccerNet_HDF5"


class MultiViewDataset(Dataset):
    def __init__(
        self,
        path,
        start,
        end,
        fps,
        split,
        num_views,
        transform=None,
        transform_model=None,
        fusion_mode=False,
        center_weighted=False,
        backbone_type=None,
        descriptions_path: str = None,
    ):
        self.split = split
        self.start = start
        self.end = end
        self.transform = transform
        self.transform_model = transform_model
        self.num_views = num_views
        self.fusion_mode = fusion_mode
        self.factor = (end - start) / (((end - start) / 25) * fps)
        self.center_weighted = center_weighted
        self.backbone_type = backbone_type
        self.vjepa_num_frames = 8 if backbone_type == "intern_video2_dist_b" else 16
        self.text_embeddings = None
        self._missing_text_embeddings = set()

        if split != "Chall":
            (
                self.labels_offence_severity,
                self.labels_action,
                self.labels_contact,
                self.labels_bodypart,
                self.labels_try_to_play,
                self.labels_handball,
                self.distribution_offence_severity,
                self.distribution_action,
                not_taking,
                self.number_of_actions,
            ) = label2vectormerge(path, split, num_views)

            self.clips = clips2vectormerge(path, split, num_views, not_taking)

            n = len(self.labels_offence_severity)
            self.distribution_offence_severity = torch.div(
                self.distribution_offence_severity, n
            )
            self.distribution_action = torch.div(self.distribution_action, n)

            self.weights_offence_severity = torch.div(
                1, torch.sqrt(self.distribution_offence_severity)
            )
            self.weights_action = torch.div(1, torch.sqrt(self.distribution_action))
            self.weights_offence_severity = (
                self.weights_offence_severity / self.weights_offence_severity.mean()
            )
            self.weights_action = self.weights_action / self.weights_action.mean()

        else:
            self.clips = clips2vectormerge(path, split, num_views, [])

        self.length = len(self.clips)

        hdf5_path = os.path.join(HDF5_ROOT, f"{split}.hdf5")
        self._hdf5_path = hdf5_path if os.path.exists(hdf5_path) else None
        self._hdf5 = None

        if descriptions_path:
            if not os.path.exists(descriptions_path):
                logging.warning(
                    f"Text embeddings not found: {descriptions_path}"
                )
            else:
                self.text_embeddings = {}
                with h5py.File(descriptions_path, "r") as f:
                    for key in f.keys():
                        self.text_embeddings[key] = np.asarray(
                            f[key], dtype=np.float32
                        )
                logging.info(
                    f"Loaded {len(self.text_embeddings)} text embeddings from {descriptions_path}"
                )

        print(
            f"Loaded {self.length} actions for {split} "
            f"({'HDF5' if self._hdf5_path else 'fallback mp4'}, "
            f"{'early fusion' if fusion_mode else 'multi-view'})"
        )

    def _get_hdf5(self):
        if self._hdf5_path is None:
            return None
        if self._hdf5 is None:
            self._hdf5 = h5py.File(self._hdf5_path, "r", swmr=True)
        return self._hdf5

    def _sample_indices_center_weighted(self, total_frames: int, n_frames: int) -> list:
        """
        Sample n_frames indices with gaussian weight around center.
        Contact moment is typically in the middle of SoccerNet clips.
        """
        import numpy as np

        center = total_frames / 2.0
        sigma = total_frames / 4.0
        positions = np.arange(total_frames, dtype=float)
        weights = np.exp(-0.5 * ((positions - center) / sigma) ** 2)
        weights /= weights.sum()
        indices = np.random.choice(
            total_frames, size=n_frames, replace=False, p=weights
        )
        return sorted(indices.tolist())

    def _sample_indices_stride(self, total_frames: int, stride: int = 2) -> list:
        """
        Stride-2 sampling — matches VJEPA pretraining distribution.
        Takes every stride-th frame, centered on clip.
        """
        indices = list(range(0, total_frames, stride))
        # Center the window if we have more than needed
        return indices

    def _sample_vjepa_frames(self, frames: torch.Tensor) -> torch.Tensor:
        total = frames.shape[0]
        if total <= 0:
            return None
        if total == self.vjepa_num_frames:
            return frames
        idx = torch.linspace(0, total - 1, steps=self.vjepa_num_frames)
        idx = idx.round().long().clamp(0, total - 1)
        return frames[idx]

    def getDistribution(self):
        return self.distribution_offence_severity, self.distribution_action

    def getWeights(self):
        return self.weights_offence_severity, self.weights_action

    def get_severity_classes(self):
        """Return integer severity class per sample — used for balanced sampler."""
        return [torch.argmax(l[0]).item() for l in self.labels_offence_severity]

    def get_balanced_sampler(self):
        """WeightedRandomSampler that equalises severity-class frequencies."""
        classes = torch.tensor(self.get_severity_classes())
        counts = torch.bincount(classes)
        class_weights = 1.0 / counts.float().clamp(min=1)
        sample_weights = class_weights[classes]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    def _process_clip(self, clip_path):
        hdf5 = self._get_hdf5()
        if hdf5 is not None:
            return self._process_clip_hdf5(clip_path, hdf5)
        return self._process_clip_video(clip_path)

    def _process_clip_hdf5(self, clip_path, hdf5):

        parts = clip_path.split(os.sep)
        action = parts[-2]
        clip_key = parts[-1].replace(".mp4", "")
        key = f"{action}/{clip_key}"

        if key not in hdf5:
            return self._process_clip_video(clip_path)

        frames = torch.from_numpy(hdf5[key][:])  # [T, H, W, C] uint8

        if frames.shape[0] < 2:
            return None

        total = frames.shape[0]

        if self.backbone_type in ("vjepa21_vitb", "intern_video2_dist_b"):
            final_frames = self._sample_vjepa_frames(frames)
        # Center-weighted sampling — focuses on contact moment
        elif self.center_weighted and total >= 8:
            import numpy as np

            center = total / 2.0
            sigma = total / 4.0
            positions = np.arange(total, dtype=float)
            weights = np.exp(-0.5 * ((positions - center) / sigma) ** 2)
            weights /= weights.sum()
            n_frames = max(1, int(total / self.factor))
            n_frames = min(n_frames, total)
            chosen = np.random.choice(total, size=n_frames, replace=False, p=weights)
            chosen = sorted(chosen.tolist())
            final_frames = frames[chosen]
        else:
            # Original uniform stride sampling
            chosen = [j for j in range(total) if j % self.factor < 1]
            if not chosen:
                return None
            final_frames = frames[chosen]

        if final_frames is None or final_frames.shape[0] == 0:
            return None

        final_frames = final_frames.permute(0, 3, 1, 2).float() / 255.0

        if self.transform is not None:
            final_frames = self.transform(final_frames)

        final_frames = self.transform_model(final_frames)
        return final_frames.permute(1, 0, 2, 3)  # [C, T, H, W]

    def _process_clip_video(self, clip_path):
        try:
            video, _, _ = read_video(clip_path, pts_unit="sec", output_format="THWC")
        except Exception as e:
            print(f"Read error {clip_path}: {e}")
            return None

        if video.shape[0] < 2:
            return None

        frames = video[self.start : self.end]
        final_frames = None

        if self.backbone_type in ("vjepa21_vitb", "intern_video2_dist_b"):
            final_frames = self._sample_vjepa_frames(frames)
        else:
            for j in range(len(frames)):
                if j % self.factor < 1:
                    f = frames[j].unsqueeze(0)
                    final_frames = (
                        f if final_frames is None else torch.cat((final_frames, f), 0)
                    )

        if final_frames is None or final_frames.shape[0] == 0:
            return None

        final_frames = final_frames.permute(0, 3, 1, 2)

        if self.transform is not None:
            final_frames = self.transform(final_frames)

        final_frames = self.transform_model(final_frames)
        return final_frames.permute(1, 0, 2, 3)

    def __getitem__(self, index):
        available_clips = self.clips[index]
        num_available = len(available_clips)

        indices = (
            random.sample(range(num_available), num_available)
            if self.split == "Train"
            else list(range(num_available))
        )

        processed_views = []
        for idx in indices:
            clip = self._process_clip(available_clips[idx])
            if clip is not None:
                processed_views.append(clip)

        if len(processed_views) == 0:
            return self.__getitem__(random.randint(0, self.length - 1))

        # Random view dropout during training (view-level regularisation)
        if self.split == "Train" and len(processed_views) > 2:
            processed_views = random.sample(processed_views, 2)

        videos = torch.stack(processed_views, dim=0)

        if videos.shape[0] < self.num_views:
            # Duplicate real views cyclically instead of
            # zero-padding — zero views waste aggregator capacity on empty tokens.
            # Views 1+ (replays) are duplicated; view 0 (live) is never the sole
            # padding source so the aggregator sees diverse replay angles.
            n_have = videos.shape[0]
            n_need = self.num_views - n_have
            # Cycle through views 1..n_have-1 (replays) for padding; fall back
            # to view 0 only if there is truly just one view.
            replay_pool = videos[1:] if n_have > 1 else videos
            indices = [i % len(replay_pool) for i in range(n_need)]
            padding = replay_pool[indices]
            videos = torch.cat((videos, padding), dim=0)

        videos = videos[: self.num_views]
        # Keep canonical layout: [V, C, T, H, W]

        if self.fusion_mode:
            # Interleave views along time: [V, C, T, H, W] -> [C, T*V, H, W]
            # At each timestep t, all V views appear consecutively.
            V_, C_, T_, H_, W_ = videos.shape
            fused = videos.permute(2, 0, 1, 3, 4)  # [T, V, C, H, W]
            fused = fused.reshape(T_ * V_, C_, H_, W_)
            fused = fused.permute(1, 0, 2, 3)
            clip_out = fused
        else:
            clip_out = videos

        text_feature = None
        if self.text_embeddings is not None:
            action_id = available_clips[0].split(os.sep)[-2]
            key = action_id if self.split == "Train" else f"{self.split}_{action_id}"
            emb = self.text_embeddings.get(key)
            if emb is None:
                if action_id not in self._missing_text_embeddings:
                    logging.warning(
                        f"Missing text embedding for {action_id}; using zeros"
                    )
                    self._missing_text_embeddings.add(action_id)
                emb = np.zeros((768,), dtype=np.float32)
            text_feature = torch.from_numpy(emb).float()

        if self.split != "Chall":
            output = (
                self.labels_offence_severity[index][0],  # (4,) one-hot
                self.labels_action[index][0],  # (8,) one-hot
                torch.tensor(self.labels_contact[index], dtype=torch.float),
                torch.tensor(self.labels_bodypart[index], dtype=torch.float),
                torch.tensor(self.labels_try_to_play[index], dtype=torch.float),
                torch.tensor(self.labels_handball[index], dtype=torch.float),
                clip_out.clone(),
                self.number_of_actions[index],
            )
            if text_feature is not None:
                output = output + (text_feature,)
            return output
        else:
            dummy = torch.tensor(0.0)
            output = (
                dummy,
                dummy,
                dummy,
                dummy,
                dummy,
                dummy,
                clip_out.clone(),
                str(index),
            )
            if text_feature is not None:
                output = output + (text_feature,)
            return output

    def __len__(self):
        return self.length

    def __del__(self):
        if hasattr(self, "_hdf5") and self._hdf5 is not None:
            self._hdf5.close()
