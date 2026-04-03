from torch.utils.data import Dataset
import torch
import random
from data_loader import label2vectormerge, clips2vectormerge
from torchvision.io.video import read_video
import warnings
import logging

# Uciszenie ostrzeżeń o jednostkach PTS i logów torchvision
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("torchvision").setLevel(logging.ERROR)


class MultiViewDataset(Dataset):
    def __init__(self, path, start, end, fps, split, num_views, transform=None, transform_model=None):

        if split != 'Chall':
            self.labels_offence_severity, self.labels_action, self.distribution_offence_severity, self.distribution_action, not_taking, self.number_of_actions = label2vectormerge(path, split, num_views)
            self.clips = clips2vectormerge(path, split, num_views, not_taking)
            self.distribution_offence_severity = torch.div(self.distribution_offence_severity, len(self.labels_offence_severity))
            self.distribution_action = torch.div(self.distribution_action, len(self.labels_action))

            self.weights_offence_severity = torch.div(1, self.distribution_offence_severity)
            self.weights_action = torch.div(1, self.distribution_action)
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
        print(f"Loaded {self.length} actions for {split}")

    def getDistribution(self):
        return self.distribution_offence_severity, self.distribution_action

    def getWeights(self):
        return self.weights_offence_severity, self.weights_action

    def _process_clip(self, clip_path):
        """
        Wczytuje i przetwarza pojedynczy klip wideo.
        Zwraca tensor [C, T, H, W] lub None jeśli klip jest uszkodzony.
        """
        try:
            # Używamy domyślnego wywołania, ale warningi są wyciszone globalnie
            video, _, _ = read_video(clip_path, output_format="THWC")
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
                if final_frames is None:
                    final_frames = frames[j, :, :, :].unsqueeze(0)
                else:
                    final_frames = torch.cat((final_frames, frames[j, :, :, :].unsqueeze(0)), 0)

        if final_frames is None:
            return None

        final_frames = final_frames.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]

        if self.transform is not None:
            final_frames = self.transform(final_frames)

        final_frames = self.transform_model(final_frames)
        final_frames = final_frames.permute(1, 0, 2, 3)  # [T, C, H, W] -> [C, T, H, W]

        return final_frames

    def __getitem__(self, index):
        available_clips = self.clips[index]
        num_available = len(available_clips)

        # Dla Train: losowa kolejność widoków (augmentacja ujęć)
        if self.split == 'Train':
            indices = random.sample(range(num_available), num_available)
        else:
            indices = list(range(num_available))

        processed_views = []
        for idx in indices:
            clip = self._process_clip(available_clips[idx])
            if clip is not None:
                processed_views.append(clip)

        # Jeśli żaden widok w akcji nie jest poprawny, losuj inną akcję (bezpiecznik)
        if len(processed_views) == 0:
            return self.__getitem__(random.randint(0, self.length - 1))

        # Stack dostępnych widoków: [V, C, T, H, W]
        videos = torch.stack(processed_views, dim=0)

        # Padding do MAX_VIEWS (4) przez duplikację ostatniego sprawnego widoku
        # Dzięki temu Transformer zawsze dostanie sekwencję o stałej długości 4
        while videos.shape[0] self.num_views:
            videos = torch.cat((videos, videos[-1:].clone()), dim=0)

        # Przycięcie na wypadek nadmiarowych plików
        videos = videos[:self.num_views]

        # Konwersja do formatu [V, T, C, H, W] oczekiwanego przez model
        videos = videos.permute(0, 2, 1, 3, 4)

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
