import os
import cv2
import h5py
import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import multiprocessing as mp
from functools import partial

# Konfiguracja
BALL_CLASS_ID = (
    32  # W modelu COCO (używanym przez standardowe YOLO) piłka sportowa to klasa 32
)


def process_single_video(video_path, pose_model, ball_model, output_shape=(18, 3)):
    """
    Przetwarza pojedynczy plik wideo, wyciąga 17 stawów COCO + 1 punkt piłki.
    Zwraca tensor numpy o kształcie [T, 18, 3].
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    center_x = frame_width / 2
    center_y = frame_height / 2

    frames_data = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Inicjalizacja pustego szkieletu klatki (17 stawów + piłka)
        # Zera oznaczają, że obiekt nie został wykryty w danej klatce
        current_frame_data = np.zeros(output_shape, dtype=np.float32)

        # --- 1. DETEKCJA SZKIELETÓW (POSE) ---
        # conf=0.3 -> Ignorujemy słabe detekcje, persist=True -> włącza tracking
        pose_results = pose_model.track(
            frame, persist=True, tracker="bytetrack.yaml", conf=0.3, verbose=False
        )

        if (
            pose_results
            and pose_results[0].keypoints is not None
            and len(pose_results[0].keypoints) > 0
        ):
            # Szukamy gracza najbliżej środka ekranu (faul zazwyczaj jest centrum akcji)
            boxes = (
                pose_results[0].boxes.xywh.cpu().numpy()
            )  # [N, 4] -> [x_center, y_center, width, height]
            keypoints = pose_results[0].keypoints.data.cpu().numpy()  # [N, 17, 3]

            best_dist = float("inf")
            best_idx = 0

            for i, box in enumerate(boxes):
                dist = np.sqrt((box[0] - center_x) ** 2 + (box[1] - center_y) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            # Kopiujemy 17 stawów najlepszego (najbliższego środka) gracza
            current_frame_data[:17, :] = keypoints[best_idx]

        # --- 2. DETEKCJA PIŁKI ---
        # classes=[32] wymusza szukanie tylko 'sports ball'
        ball_results = ball_model(
            frame, classes=[BALL_CLASS_ID], conf=0.15, verbose=False
        )

        if (
            ball_results
            and ball_results[0].boxes is not None
            and len(ball_results[0].boxes) > 0
        ):
            # Bierzemy piłkę z najwyższym confidence
            ball_boxes = ball_results[0].boxes
            best_ball = ball_boxes[ball_boxes.conf.argmax()]

            b_x, b_y = best_ball.xywh[0][:2].cpu().numpy()
            b_conf = best_ball.conf[0].cpu().numpy()

            # Punkt 18 to piłka (indeks 17 w tablicy)
            current_frame_data[17] = [b_x, b_y, b_conf]

        frames_data.append(current_frame_data)

    cap.release()

    if len(frames_data) == 0:
        return None

    return np.array(frames_data)  # Ostateczny kształt: [T, 18, 3]


def extract_split(split_name, data_dir, output_hdf5_path):
    """
    Przeszukuje folder danego splitu (np. Train), znajduje wszystkie mp4
    i zapisuje je do pliku HDF5.
    """
    split_dir = Path(data_dir) / split_name
    if not split_dir.exists():
        print(f"Katalog {split_dir} nie istnieje. Pomijam.")
        return

    # Znajdź wszystkie pliki clip_X.mp4 w podfolderach action_Y
    video_files = list(split_dir.rglob("clip_*.mp4"))
    print(f"Znaleziono {len(video_files)} klipów wideo w splicie {split_name}.")

    if len(video_files) == 0:
        return

    # Upewnij się, że katalog docelowy HDF5 istnieje
    Path(output_hdf5_path).parent.mkdir(parents=True, exist_ok=True)

    # Inicjalizacja modeli w głównym procesie (będą współdzielone)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Używam urządzenia: {device}")

    # YOLOv8x-pose to najdokładniejszy model do szkieletów
    pose_model = YOLO("yolov8x-pose.pt").to(device)
    # YOLOv8m to dobry kompromis między szybkością a dokładnością dla małej piłki
    ball_model = YOLO("yolov8m.pt").to(device)

    # Otwieramy plik HDF5 do zapisu
    with h5py.File(output_hdf5_path, "w") as hdf5_file:

        # Pętla po wszystkich wideo z paskiem postępu
        for video_path in tqdm(video_files, desc=f"Przetwarzanie {split_name}"):
            try:
                # Wypakuj action_id i clip_id ze ścieżki (np. action_42/clip_0.mp4)
                action_folder = video_path.parent.name
                clip_name = video_path.stem
                dataset_key = f"{action_folder}/{clip_name}"  # np. "action_42/clip_0"

                # Pomiń, jeśli już wyodrębniono (przydatne przy wznawianiu przerwanego joba)
                if dataset_key in hdf5_file:
                    continue

                # Przetwarzanie
                tensor_data = process_single_video(video_path, pose_model, ball_model)

                if tensor_data is not None:
                    # Zapisz do HDF5 z kompresją (oszczędza bardzo dużo miejsca)
                    hdf5_file.create_dataset(
                        dataset_key,
                        data=tensor_data,
                        compression="gzip",
                        compression_opts=4,
                    )

            except Exception as e:
                print(f"Błąd podczas przetwarzania {video_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ekstrakcja szkieletów COCO17 + Piłka z SoccerNet"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Katalog z danymi wideo SoccerNet (np. /net/tscratch/.../SoccerNet_Data)",
    )
    parser.add_argument(
        "--out_dir", type=str, required=True, help="Katalog wyjściowy na pliki HDF5"
    )

    args = parser.parse_args()

    # Automatycznie pobierze wagi YOLO przy pierwszym uruchomieniu
    splits = ["Train", "Valid", "Test", "Chall"]

    for split in splits:
        out_path = os.path.join(args.out_dir, f"{split}.hdf5")
        extract_split(split, args.data_dir, out_path)

    print("Ekstrakcja całkowicie zakończona!")
