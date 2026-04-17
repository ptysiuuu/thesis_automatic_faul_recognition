from torch.utils.data import Dataset
import torch
import random
from data_loader import label2vectormerge, clips2vectormerge
from torchvision.io.video import read_video
import warnings
import logging
import h5py
import os

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("torchvision").setLevel(logging.ERROR)

HDF5_ROOT = "/net/tscratch/people/plgaszos/SoccerNet_HDF5"
TARGET_FRAMES = 16

class MultiViewDataset(Dataset):
    def __init__(self, path, start, end, fps, split, num_views, transform=None, transform_model=None):

        if split != 'Chall':
            self.labels_offence_severity, self.labels_action, self.distribution_offence_severity, self.distribution_action, not_taking, self.number_of_actions = label2vectormerge(path, split, num_views)
            self.clips = clips2vectormerge(path, split, num_views, not_taking)
            self.distribution_offence_severity = torch.div(self.distribution_offence_severity, len(self.labels_offence_severity))
            self.distribution_action = torch.div(self.distribution_action, len(self.labels_action))

            self.weights_offence_severity = torch.div(1, torch.sqrt(self.distribution_offence_severity))
            self.weights_action = torch.div(1, torch.sqrt(self.distribution_action))
            self.weights_offence_severity = self.weights_offence_severity / self.weights_offence_severity.mean()
            self.weights_action = self.weights_action / self.weights_action.mean()
        else:
            self.clips = clips2vectormerge(path, split, num_views, [])

        self.split = split
        self.start = start
        self.end = end
        self.transform = transform
        self.transform_model = transform_model
        self.num_views = num_views
        self.factor = (end - start) / (((end - start) / 25) * fps)
        self.length = len(self.clips)

        # Ścieżka do HDF5 — NIE otwieramy tutaj, tylko zapamiętujemy ścieżkę
        hdf5_path = os.path.join(HDF5_ROOT, f"{split}.hdf5")
        self._hdf5_path = hdf5_path if os.path.exists(hdf5_path) else None
        self._hdf5 = None  # lazy init per worker

        print(f"Loaded {self.length} actions for {split} "
              f"({'HDF5' if self._hdf5_path else 'fallback mp4'})")

    def _get_hdf5(self):
        """Lazy init — każdy worker DataLoadera otworzy własną kopię pliku."""
        if self._hdf5_path is None:
            return None
        if self._hdf5 is None:
            self._hdf5 = h5py.File(self._hdf5_path, "r", swmr=True)
        return self._hdf5

    def getDistribution(self):
        return self.distribution_offence_severity, self.distribution_action

    def getWeights(self):
        return self.weights_offence_severity, self.weights_action

    def _process_clip(self, clip_path):
        hdf5 = self._get_hdf5()

        if hdf5 is not None:
            return self._process_clip_hdf5(clip_path, hdf5)
        else:
            return self._process_clip_video(clip_path)

    def _process_clip_hdf5(self, clip_path, hdf5):
        parts    = clip_path.split(os.sep)
        action   = parts[-2]
        clip_key = parts[-1].replace(".mp4", "")
        key      = f"{action}/{clip_key}"

        if key not in hdf5:
            # Fallback do mp4 jeśli klip nie trafił do HDF5 (uszkodzony podczas ekstrakcji)
            return self._process_clip_video(clip_path)

        frames = torch.from_numpy(hdf5[key][:])  # [T, H, W, C] uint8

        if frames.shape[0] < 2:
            return None

        final_frames = None
        for j in range(len(frames)):
            if j % self.factor < 1:
                f = frames[j].unsqueeze(0)
                final_frames = f if final_frames is None else torch.cat((final_frames, f), 0)

        if final_frames is None:
            return None

        final_frames = final_frames.permute(0, 3, 1, 2).float() / 255.0  # [T, C, H, W]

        if self.transform is not None:
            final_frames = self.transform(final_frames)

        final_frames = self.transform_model(final_frames)
        final_frames = final_frames.permute(1, 0, 2, 3)  # [C, T, H, W]

        C, T, H, W = final_frames.shape
        if T != TARGET_FRAMES:
            final_frames = final_frames.unsqueeze(0)  # [1, C, T, H, W]
            final_frames = torch.nn.functional.interpolate(
                final_frames, size=(TARGET_FRAMES, H, W),
                mode='trilinear', align_corners=False
            ).squeeze(0)  # [C, TARGET_FRAMES, H, W]

        return final_frames

    def _process_clip_video(self, clip_path):
        """Fallback — stary kod z read_video."""
        try:
            video, _, _ = read_video(clip_path, pts_unit='sec', output_format="THWC")
        except Exception as e:
            print(f"Błąd odczytu {clip_path}: {e}")
            return None

        if video.shape[0] < 2:
            print(f"Ostrzeżenie: Plik uszkodzony/pusty, pomijam: {clip_path}")
            return None

        frames = video[self.start:self.end, :, :, :]
        final_frames = None

        for j in range(len(frames)):
            if j % self.factor < 1:
                f = frames[j, :, :, :].unsqueeze(0)
                final_frames = f if final_frames is None else torch.cat((final_frames, f), 0)

        if final_frames is None:
            return None

        final_frames = final_frames.permute(0, 3, 1, 2).float() / 255.0

        if self.transform is not None:
            final_frames = self.transform(final_frames)

        final_frames = self.transform_model(final_frames)
        final_frames = final_frames.permute(1, 0, 2, 3)

        C, T, H, W = final_frames.shape
        if T != TARGET_FRAMES:
            final_frames = final_frames.unsqueeze(0)
            final_frames = torch.nn.functional.interpolate(
                final_frames, size=(TARGET_FRAMES, H, W),
                mode='trilinear', align_corners=False
            ).squeeze(0)

        return final_frames

    def __getitem__(self, index):
        available_clips = self.clips[index]
        num_available = len(available_clips)

        if self.split == 'Train':
            indices = random.sample(range(num_available), num_available)
        else:
            indices = list(range(num_available))

        processed_views = []
        for idx in indices:
            clip = self._process_clip(available_clips[idx])
            if clip is not None:
                processed_views.append(clip)

        if len(processed_views) == 0:
            return self.__getitem__(random.randint(0, self.length - 1))

        if self.split == 'Train' and len(processed_views) > 1:
            surviving = [v for v in processed_views if random.random() > 0.2]
            if len(surviving) > 0:
                processed_views = surviving

        videos = torch.stack(processed_views, dim=0)

        if videos.shape[0] < self.num_views:
            pad_shape = list(videos.shape)
            pad_shape[0] = self.num_views - videos.shape[0]
            padding = torch.zeros(pad_shape, dtype=videos.dtype)
            videos = torch.cat((videos, padding), dim=0)

        videos = videos[:self.num_views]

        if self.split != 'Chall':
            return (
                self.labels_offence_severity[index][0],
                self.labels_action[index][0],
                videos.clone(),
                self.number_of_actions[index]
            )
        else:
            return -1, -1, videos.clone(), str(index)

    def __len__(self):
        return self.length

    def __del__(self):
        if hasattr(self, '_hdf5') and self._hdf5 is not None:
            self._hdf5.close()