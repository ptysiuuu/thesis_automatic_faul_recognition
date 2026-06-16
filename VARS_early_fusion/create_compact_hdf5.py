"""
create_compact_hdf5.py
Converts SoccerNet MP4 clips to a compact uint8 HDF5.

One HDF5 per split: {output_dir}/{split}.hdf5
Key layout: {action_id}/{view_id}  (e.g. action_0/clip_1)
Root attributes: start_frame, end_frame, num_frames, img_size
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import h5py
import numpy as np
from tqdm import tqdm


def _decode_clip(args):
    mp4_path, start_frame, end_frame, num_frames, img_size = args
    try:
        from decord import VideoReader, cpu

        vr = VideoReader(mp4_path, ctx=cpu(0))
        total = len(vr)
        sf = min(start_frame, total - 1)
        ef = min(end_frame, total - 1)
        indices = np.linspace(sf, ef, num_frames).round().astype(int)
        indices = np.clip(indices, 0, total - 1)
        frames = vr.get_batch(indices).asnumpy()  # [T, H, W, 3]
        resized = np.stack(
            [
                cv2.resize(f, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
                for f in frames
            ]
        )
        return resized.astype(np.uint8)
    except Exception:
        return None


def process_split(
    data_dir, output_dir, split, start_frame, end_frame, num_frames, img_size, num_workers
):
    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        print(f"[{split}] directory not found: {split_dir}")
        return

    out_path = os.path.join(output_dir, f"{split}.hdf5")
    os.makedirs(output_dir, exist_ok=True)

    # Collect all (key, mp4_path) pairs
    tasks = []
    for action_name in sorted(os.listdir(split_dir)):
        action_dir = os.path.join(split_dir, action_name)
        if not os.path.isdir(action_dir):
            continue
        for fname in sorted(os.listdir(action_dir)):
            if not fname.endswith(".mp4"):
                continue
            view_id = fname[:-4]
            key = f"{action_name}/{view_id}"
            tasks.append((key, os.path.join(action_dir, fname)))

    with h5py.File(out_path, "a") as h5f:
        h5f.attrs["start_frame"] = start_frame
        h5f.attrs["end_frame"] = end_frame
        h5f.attrs["num_frames"] = num_frames
        h5f.attrs["img_size"] = img_size

        pending = [(key, mp4) for key, mp4 in tasks if key not in h5f]
        already_done = len(tasks) - len(pending)
        if already_done:
            print(f"[{split}] Skipping {already_done} already-processed clips")
        if not pending:
            print(f"[{split}] All {len(tasks)} clips already processed.")
            return

        decode_args = [
            (mp4, start_frame, end_frame, num_frames, img_size) for _, mp4 in pending
        ]
        key_map = {i: key for i, (key, _) in enumerate(pending)}

        n_ok = n_err = 0
        futures = {}
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for i, arg in enumerate(decode_args):
                fut = executor.submit(_decode_clip, arg)
                futures[fut] = key_map[i]

            with tqdm(total=len(pending), desc=split) as pbar:
                for fut in as_completed(futures):
                    key = futures[fut]
                    frames = fut.result()
                    if frames is not None:
                        if key not in h5f:
                            h5f.create_dataset(
                                key,
                                data=frames,
                                compression="gzip",
                                compression_opts=4,
                            )
                        n_ok += 1
                    else:
                        n_err += 1
                    pbar.update(1)

    print(f"[{split}] Done — {n_ok} written, {n_err} errors → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert SoccerNet MP4 clips to compact uint8 HDF5"
    )
    parser.add_argument("--data_dir", required=True, help="SoccerNet_Data root with MP4s")
    parser.add_argument("--output_dir", required=True, help="Where to write HDF5 files")
    parser.add_argument("--start_frame", type=int, default=58)
    parser.add_argument("--end_frame", type=int, default=92)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--splits", nargs="+", default=["Train", "Valid", "Test"])
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    for split in args.splits:
        process_split(
            args.data_dir,
            args.output_dir,
            split,
            args.start_frame,
            args.end_frame,
            args.num_frames,
            args.img_size,
            args.num_workers,
        )


if __name__ == "__main__":
    main()
