"""
utils/constants.py
==================
Shared label mappings and constants used across the entire pipeline.
"""

ACTION_CLASSES = [
    "Tackling",
    "Standing tackling",
    "High leg",
    "Holding",
    "Pushing",
    "Elbowing",
    "Challenge",
    "Dive",
]
SEVERITY_CLASSES = ["No offence", "No card", "Yellow card", "Red card"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_CLASSES)}
SEVERITY_TO_IDX = {s: i for i, s in enumerate(SEVERITY_CLASSES)}

OFFENCE_SEVERITY_MAP = {
    ("No offence", ""): 0,
    ("No Offence", ""): 0,
    ("Offence", "1.0"): 1,
    ("Offence", "3.0"): 2,
    ("Offence", "5.0"): 3,
}

# Default severity priors (overridden at runtime from training set)
SEVERITY_PRIOR_DEFAULT = {
    "No offence": 8,
    "No card": 20,
    "Yellow card": 58,
    "Red card": 14,
}

# Per-action severity priors computed from SoccerNet-MVFoul Train split.
# Maps action_name -> {severity_name -> percentage}.
# Used for severity calibration in prompts.
PER_ACTION_SEVERITY_PRIOR = {
    "Tackling": {"No offence": 5, "No card": 15, "Yellow card": 60, "Red card": 20},
    "Standing tackling": {
        "No offence": 10,
        "No card": 30,
        "Yellow card": 52,
        "Red card": 8,
    },
    "High leg": {"No offence": 5, "No card": 10, "Yellow card": 45, "Red card": 40},
    "Holding": {"No offence": 12, "No card": 35, "Yellow card": 48, "Red card": 5},
    "Pushing": {"No offence": 15, "No card": 38, "Yellow card": 42, "Red card": 5},
    "Elbowing": {"No offence": 3, "No card": 8, "Yellow card": 35, "Red card": 54},
    "Challenge": {"No offence": 8, "No card": 28, "Yellow card": 55, "Red card": 9},
    "Dive": {"No offence": 60, "No card": 35, "Yellow card": 4, "Red card": 1},
}

# Which actions are commonly confused with each other (for targeted medoid retrieval)
CONFUSABLE_ACTIONS = {
    "Tackling": ["Standing tackling", "Challenge"],
    "Standing tackling": ["Tackling", "Challenge"],
    "High leg": ["Tackling", "Challenge"],
    "Elbowing": ["Challenge", "Pushing"],
    "Challenge": ["Tackling", "High leg", "Standing tackling"],
    "Holding": ["Pushing", "Challenge"],
    "Pushing": ["Holding", "Challenge"],
    "Dive": [],
}

# All strategy names — add new ones here as they are implemented
ALL_STRATEGIES = [
    # ── Baseline ──────────────────────────────────────────────────────────────
    "static_few_shot",  # Row 0: handcrafted examples, all frames
    # ── Training data utilization ─────────────────────────────────────────────
    "data_driven",  # Row 1: medoid examples + severity prior
    "two_stage",  # Row 2: action first, severity second
    "rag_icl",  # Row 3: MViT FAISS dynamic retrieval
    # ── Frame selection ────────────────────────────────────────────────────────
    "cos_two_stage",  # Row 4: Chain-of-Shot + two-stage
    # ── Severity-focused (new) ─────────────────────────────────────────────────
    "full_sev_two_stage",  # Row 5: CoS action + ALL frames for severity
    "per_action_prior",  # Row 6: static_few_shot + per-action severity priors
    "targeted_retrieval",  # Row 7: targeted medoid (predicted action + confusable)
    "ordinal_severity",  # Row 8: reframe severity as ordinal comparison task
    "cos_full_sev",
    "flow_hard_neg",
    "cos_disambig",
]
