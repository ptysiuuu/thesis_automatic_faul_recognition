"""
retrieval/medoid_cache.py
=========================
Builds and loads a cache of medoid training examples per (action × severity) class.
Each medoid is the most prototypical training clip for its class, stored as
base64 JPEG frames for injection into VLM prompts.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from io import BytesIO
from typing import List, Tuple
from PIL import Image

from ..utils.annotations import load_annotations
from ..utils.constants import ACTION_CLASSES, SEVERITY_CLASSES, CONFUSABLE_ACTIONS

SEVERITY_DESCRIPTIONS = {
    "Challenge|No card": (
        "The player makes a late or careless challenge with limited speed, low body control, and a foot that arrives a bit behind the ball. "
        "Under IFAB Law 12 this is careless contact: a foul, but not reckless enough to merit a card."
    ),
    "Challenge|No offence": (
        "The challenger reaches the ball first or there is only very slight contact with no clear disruption, and the opponent stays upright. "
        "Under IFAB Law 12 there is no meaningful foul contact or simulation, so no offence is given."
    ),
    "Challenge|Yellow card": (
        "The player lunges with noticeable speed, leaves the ground or arrives out of control, and the challenge clips the opponent with force. "
        "Under IFAB Law 12 that reckless disregard for the opponent's safety requires a yellow card."
    ),
    "Dive|No card": (
        "The attacker throws the body down or collapses after minimal or no contact, but the fall is exaggerated and clearly not caused by a tackle. "
        "Under IFAB Law 12 this is simulation rather than a genuine foul, so no card is shown for the incident itself."
    ),
    "Dive|No offence": (
        "The player goes to ground with no visible contact and no defender touching the body, often with flailing arms or a delayed collapse. "
        "Under IFAB Law 12 there is no foul by the opponent, so the correct decision is no offence."
    ),
    "Elbowing|No card": (
        "The upper arm rises during a shoulder-to-shoulder duel with only glancing contact to the opponent's torso and no violent follow-through. "
        "Under IFAB Law 12 this is careless upper-body contact, so it is a foul but not a cardable offense."
    ),
    "Elbowing|No offence": (
        "The arm is near the opponent but there is no clear strike, no forceful swing, and the opponent does not react to impact. "
        "Under IFAB Law 12 there is no significant contact or intent to strike, so no offence is given."
    ),
    "Elbowing|Red card": (
        "The elbow swings high and hard toward the opponent's face or head, with the body turning aggressively and the opponent recoiling or falling. "
        "Under IFAB Law 12 that violent conduct uses excessive force and endangers safety, so it requires a red card."
    ),
    "Elbowing|Yellow card": (
        "The elbow is lifted above shoulder height and makes firm contact with the opponent's head or upper body, but without the full violence of a strike to the face. "
        "Under IFAB Law 12 this is reckless upper-body contact and merits a yellow card."
    ),
    "High leg|No card": (
        "The foot is raised high in a challenge but the player keeps control, the leg stays clear of the opponent, and the ball is played first or nearly first. "
        "Under IFAB Law 12 this is careless positioning rather than reckless danger, so it is a foul without a card."
    ),
    "High leg|No offence": (
        "The raised leg never reaches the opponent and the movement is contained, with no visible contact and no player sent off balance. "
        "Under IFAB Law 12 there is no meaningful contact or danger to penalize, so no offence is given."
    ),
    "High leg|Red card": (
        "The boot comes up dangerously high, often near the opponent's head or chest, with a fast and uncontrolled motion that makes the opponent flinch or fall. "
        "Under IFAB Law 12 that excessive force endangers safety and requires a red card."
    ),
    "High leg|Yellow card": (
        "The foot is raised above a safe level and clips the opponent during the challenge, showing clear disregard for the opponent's position. "
        "Under IFAB Law 12 that reckless high-leg contact warrants a yellow card."
    ),
    "Holding|No card": (
        "The defender briefly grabs a shirt or arm while tracking the opponent, but the grip is light and the opponent keeps moving with little resistance. "
        "Under IFAB Law 12 this is careless holding: a foul, but not enough for a card."
    ),
    "Holding|No offence": (
        "The hands brush the opponent without any visible grip or pulling, and the attacker is not impeded. "
        "Under IFAB Law 12 there is no actual holding action, so no offence is given."
    ),
    "Holding|Yellow card": (
        "The player clearly clamps onto the shirt or arm, tugs the opponent backward, and breaks the opponent's stride. "
        "Under IFAB Law 12 that reckless denial of space or movement requires a yellow card."
    ),
    "Pushing|No card": (
        "The hands make brief contact with the opponent's body and create only a small loss of balance, with no hard shove or aggressive follow-through. "
        "Under IFAB Law 12 this is careless pushing and is a foul without a card."
    ),
    "Pushing|No offence": (
        "The player reaches toward the opponent but there is no visible shove and the opponent remains balanced. "
        "Under IFAB Law 12 there is no meaningful contact, so no offence is given."
    ),
    "Pushing|Yellow card": (
        "The player extends both arms and forcefully drives the opponent sideways or backward, often making the opponent stumble or fall. "
        "Under IFAB Law 12 that reckless push deserves a yellow card."
    ),
    "Standing tackling|No card": (
        "The defender steps in upright with a controlled foot movement to the ball, but arrives a little late and clips the opponent lightly. "
        "Under IFAB Law 12 this is a careless standing tackle, so it is a foul without a card."
    ),
    "Standing tackling|No offence": (
        "The player stretches the foot toward the ball and avoids the opponent, with clean contact on the ball and no visible impact on the attacker. "
        "Under IFAB Law 12 there is no foul contact, so no offence is given."
    ),
    "Standing tackling|Red card": (
        "The standing tackle arrives with a high, forceful leg and the defender drives through the opponent with a hard collision that knocks the opponent down. "
        "Under IFAB Law 12 that excessive force endangers the opponent and requires a red card."
    ),
    "Standing tackling|Yellow card": (
        "The defender commits to an upright challenge with a late, forceful clip that sends the opponent off balance and shows clear disregard for safety. "
        "Under IFAB Law 12 this reckless standing tackle deserves a yellow card."
    ),
    "Tackling|No card": (
        "The player slides or steps in with moderate speed, wins or reaches the ball, but leaves some late contact on the opponent's leg or foot. "
        "Under IFAB Law 12 this is careless tackling: a foul, but not reckless enough for a card."
    ),
    "Tackling|No offence": (
        "The tackler gets the ball cleanly or misses the opponent entirely, and the challenge does not disturb the opponent's movement. "
        "Under IFAB Law 12 there is no foul contact, so no offence is given."
    ),
    "Tackling|Red card": (
        "The tackle is a fast, forceful slide or lunge through the opponent's legs, with studs or full-body contact that makes the opponent fall hard. "
        "Under IFAB Law 12 this is excessive force and serious foul play, so it requires a red card."
    ),
    "Tackling|Yellow card": (
        "The tackler arrives late or out of control, clips the opponent with a strong challenge, and clearly ignores the opponent's safety. "
        "Under IFAB Law 12 that reckless tackle merits a yellow card."
    ),
}


def build_medoid_cache(
    train_hdf5_path: str,
    train_annotations_path: str,
    faiss_index_path: str,
    faiss_meta_path: str,
    output_path: str,
    n_frames_per_example: int = 2,
):
    """
    For each (action, severity) class combination, find the training sample
    whose MViT feature is closest to the class centroid (medoid).
    Saves N frames as base64 JPEGs.
    """
    import faiss, h5py

    FEAT_DIM = 768
    print("[MedoidCache] Loading FAISS index and metadata...")
    index = faiss.read_index(faiss_index_path)
    with open(faiss_meta_path) as f:
        metadata = json.load(f)

    # Group FAISS indices by (action, severity)
    groups = defaultdict(list)
    for idx_str, meta in metadata.items():
        key = f"{meta['action']}|{meta['severity']}"
        groups[key].append(int(idx_str))

    print(f"[MedoidCache] {len(groups)} (action, severity) groups:")
    for k, v in sorted(groups.items()):
        print(f"  {k}: {len(v)} samples")

    # Reconstruct feature vectors
    all_feats = np.zeros((index.ntotal, FEAT_DIM), dtype=np.float32)
    for i in range(index.ntotal):
        faiss.downcast_index(index).reconstruct(i, all_feats[i])

    # Find medoid per group
    medoid_faiss_indices = {}
    for key, indices in groups.items():
        if len(indices) == 1:
            medoid_faiss_indices[key] = indices[0]
            continue
        feats = all_feats[indices]
        dists = np.sum((feats[:, None] - feats[None, :]) ** 2, axis=-1)
        medoid_faiss_indices[key] = indices[np.argmin(dists.sum(axis=1))]

    # Load frames for each medoid
    train_samples = load_annotations(train_annotations_path)
    faiss_to_action = {
        int(idx_str): meta["action_id"] for idx_str, meta in metadata.items()
    }

    cache = {}
    with h5py.File(train_hdf5_path, "r", swmr=True) as hdf5:
        for key, faiss_idx in medoid_faiss_indices.items():
            action_id = faiss_to_action.get(faiss_idx)
            if not action_id or action_id not in train_samples:
                continue
            sample = train_samples[action_id]

            frames_b64 = []
            for clip_raw in sample["clips"][:1]:
                clip_key = clip_raw.replace(".mp4", "")
                hdf5_key = f"action_{action_id}/{clip_key}"
                if hdf5_key not in hdf5:
                    continue
                frames_np = hdf5[hdf5_key][:]
                T = len(frames_np)
                if T < 2:
                    continue
                indices = np.linspace(0, T - 1, n_frames_per_example, dtype=int)
                for idx in indices:
                    img = Image.fromarray(frames_np[idx])
                    buf = BytesIO()
                    img.save(buf, format="JPEG", quality=85)
                    import base64

                    frames_b64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

            if frames_b64:
                action_str, severity_str = key.split("|")
                cache[key] = {
                    "action": action_str,
                    "severity": severity_str,
                    "action_id": action_id,
                    "frames_b64": frames_b64,
                }
                print(f"  ✓ {key} → action_id={action_id}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(cache, f)
    print(f"[MedoidCache] Saved {len(cache)} entries → {output_path}")
    return cache


def load_medoid_cache(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_examples_text(
    medoid_cache: dict, n_per_class: int = 1
) -> Tuple[str, List[Image.Image]]:
    """All medoid examples as text + PIL images."""
    import base64

    examples_text = ""
    example_images = []
    for i, (key, entry) in enumerate(sorted(medoid_cache.items()), 1):
        action, severity = entry["action"], entry["severity"]
        examples_text += (
            f"EXAMPLE {i} — {action} / {severity}:\n"
            f'  Decision: {{"action": "{action}", "severity": "{severity}"}}\n\n'
        )
        for b64 in entry["frames_b64"][:n_per_class]:
            img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
            example_images.append(img)
    return examples_text.strip(), example_images


def build_targeted_examples(
    medoid_cache: dict,
    predicted_action: str,
    n_same_action: int = 2,
    n_confusable: int = 1,
) -> Tuple[str, List[Image.Image]]:
    """
    Build targeted examples for a predicted action:
    - n_same_action examples of the predicted action (different severities)
    - n_confusable examples from the most easily confused action

    This is the key fix for data_driven: instead of all examples every time,
    show only contextually relevant ones.
    """
    import base64

    selected = []

    # Same action, different severities
    for key, entry in sorted(medoid_cache.items()):
        if entry["action"] == predicted_action and len(selected) < n_same_action:
            selected.append(entry)

    # Confusable action
    confusable = CONFUSABLE_ACTIONS.get(predicted_action, [])
    for conf_action in confusable:
        for key, entry in sorted(medoid_cache.items()):
            if (
                entry["action"] == conf_action
                and len(selected) < n_same_action + n_confusable
            ):
                selected.append(entry)
                break

    examples_text = ""
    example_images = []
    for i, entry in enumerate(selected, 1):
        action, severity = entry["action"], entry["severity"]
        examples_text += (
            f"EXAMPLE {i} — {action} / {severity}:\n"
            f'  Decision: {{"action": "{action}", "severity": "{severity}"}}\n\n'
        )
        for b64 in entry["frames_b64"][:1]:
            img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
            example_images.append(img)

    return examples_text.strip(), example_images


def build_severity_examples(
    medoid_cache: dict, action: str
) -> Tuple[str, List[Image.Image]]:
    """Examples for a specific action, covering all severity levels."""
    import base64

    examples_text = ""
    example_images = []
    rank = 1
    for key, entry in sorted(medoid_cache.items()):
        if entry["action"] != action:
            continue
        severity = entry["severity"]
        description = SEVERITY_DESCRIPTIONS.get(key)
        examples_text += f"EXAMPLE {rank} — {action} / {severity}:\n"
        if description:
            examples_text += f"  Incident: {description}\n"
        examples_text += (
            f'  Decision: {{"action": "{action}", "severity": "{severity}"}}\n\n'
        )
        for b64 in entry["frames_b64"][:1]:
            img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
            example_images.append(img)
        rank += 1
    return examples_text.strip(), example_images
