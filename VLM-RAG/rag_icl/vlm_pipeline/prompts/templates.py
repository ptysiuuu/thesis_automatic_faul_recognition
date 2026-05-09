"""
prompts/templates.py
====================
All prompt template strings. Separated from builders so templates
can be read and compared easily without logic mixed in.
"""

from ..utils.constants import ACTION_CLASSES, SEVERITY_CLASSES

ACTION_LIST_STR = "\n".join(f"  - {a}" for a in ACTION_CLASSES)
SEVERITY_LIST_STR = "\n".join(f"  - {s}" for s in SEVERITY_CLASSES)

SYSTEM_PROMPT = (
    "You are an expert football referee assistant. "
    "Analyze video frames from multiple camera angles and classify football "
    "foul incidents according to FIFA Laws of the Game. "
    "Always respond with ONLY a JSON object — no other text."
)

# ── Row 0: Static few-shot ─────────────────────────────────────────────────────
STATIC_EXAMPLES = """\
EXAMPLE 1 — Tackling / Red card:
Incident: Player lunges from behind, foot raised, full contact with opponent's leg.
Decision: {"action": "Tackling", "severity": "Red card"}
Reason: Tackle from behind with excessive force — serious foul play.

EXAMPLE 2 — Elbowing / Red card:
Incident: Player extends elbow into opponent's face, ball not nearby.
Decision: {"action": "Elbowing", "severity": "Red card"}
Reason: Violent conduct regardless of ball proximity.

EXAMPLE 3 — Dive / No offence:
Incident: Player falls dramatically after minimal contact.
Decision: {"action": "Dive", "severity": "No offence"}
Reason: Simulation — yellow card for unsporting behaviour.

EXAMPLE 4 — Holding / Yellow card:
Incident: Player grabs opponent's shirt during a counterattack.
Decision: {"action": "Holding", "severity": "Yellow card"}
Reason: Reckless holding — disregards opponent's progress."""

STATIC_FEW_SHOT_TMPL = """\
You are analyzing a potential football foul from {n_views} camera angles.
View 0 is the live broadcast camera. Views 1+ are replay cameras.

{law12_context}

Here are examples of correctly classified incidents:
{examples}

Now classify the incident shown in the video frames.

ACTION TYPE (choose exactly one):
{action_list}

SEVERITY (choose exactly one):
{severity_list}

Respond with ONLY this JSON:
{{"action": "<action type>", "severity": "<severity>", "reasoning": "<one sentence>"}}"""

# ── Row 1: Data-driven ────────────────────────────────────────────────────────
DATA_DRIVEN_TMPL = """\
You are analyzing a potential football foul from {n_views} camera angles.
View 0 is the live broadcast camera. Views 1+ are replay cameras.

{law12_context}

The following are REAL examples from officially judged incidents:

{mined_examples}

SEVERITY CALIBRATION: In official match data the severity distribution is:
{prior_str}
Do not default to Yellow card — Red card and No offence are equally valid.

Now classify the NEW incident shown in the video frames above.

ACTION TYPE (choose exactly one):
{action_list}

SEVERITY (choose exactly one):
{severity_list}

Respond with ONLY this JSON:
{{"action": "<action type>", "severity": "<severity>", "reasoning": "<one sentence>"}}"""

# ── Row 2: Two-stage — stage 1 (action only) ──────────────────────────────────
TWO_STAGE_ACTION_TMPL = """\
You are analyzing a potential football foul from {n_views} camera angles.
View 0 is the live broadcast camera. Views 1+ are replay cameras.

{law12_context}

Reference examples:
{mined_examples}

Your task NOW: identify only the ACTION TYPE of the incident.

ACTION TYPE (choose exactly one):
{action_list}

Respond with ONLY this JSON:
{{"action": "<action type>", "reasoning": "<one sentence about the body movement>"}}"""

# ── Row 2: Two-stage — stage 2 (severity, given action) ───────────────────────
TWO_STAGE_SEVERITY_TMPL = """\
You are analyzing a potential football foul from {n_views} camera angles.
View 0 is the live broadcast camera. Views 1+ are replay cameras.

The action type has already been identified as: {predicted_action}

{law12_context}

Examples of "{predicted_action}" incidents with official decisions:
{severity_examples}

SEVERITY CALIBRATION: {prior_str}

Given this is a "{predicted_action}", assess the severity:

SEVERITY (choose exactly one):
{severity_list}

Rules:
- EXCESSIVE FORCE / endangering safety → RED CARD
- RECKLESS (disregard for opponent)    → YELLOW CARD
- CARELESS (lack of attention)         → No card but foul
- NO CONTACT / SIMULATION              → No offence

Respond with ONLY this JSON:
{{"severity": "<severity>", "reasoning": "<one sentence citing force level>"}}"""

# ── Row 3: RAG-ICL ────────────────────────────────────────────────────────────
RAGICL_TMPL = """\
You are analyzing a potential football foul from {n_views} camera angles.
View 0 is the live broadcast camera. Views 1+ are replay cameras.

{law12_context}

VISUALLY SIMILAR fouls retrieved by motion similarity:
{dynamic_examples}

Now classify the NEW incident shown in the video frames above.

ACTION TYPE (choose exactly one):
{action_list}

SEVERITY (choose exactly one):
{severity_list}

Respond with ONLY this JSON:
{{"action": "<action type>", "severity": "<severity>", "reasoning": "<one sentence>"}}"""

# ── Row 4: CoS — stage 0 (frame selection) ────────────────────────────────────
COS_FRAME_SELECT_TMPL = """\
You are analyzing a potential football foul incident.
You are shown {frames_per_view} frames from each of {n_views} camera angles.
Frames are numbered 0 (earliest) to {max_frame_idx} (latest).

Your task: for each camera view, identify the SINGLE most informative frame —
the frame that most clearly shows the moment of physical contact, attempted contact,
or peak action (foot position at impact, elbow contact, body fall).

Do NOT classify the foul yet — only select frame indices.

Camera views present:
{view_list}

Respond with ONLY this JSON (one key per view label, value is frame index 0-{max_frame_idx}):
{{{frame_json_template}}}"""

# ── Row 4: CoS — stage 1 (action from key frames) ────────────────────────────
COS_ACTION_TMPL = """\
You are analyzing a potential football foul from {n_views} camera angles.
View 0 is the live broadcast camera. Views 1+ are replay cameras.

IMPORTANT: The frames shown are the KEY FRAMES most informative for this incident:
{selected_frame_info}
Focus on these frames — they show the critical contact moment.

{law12_context}

Reference examples from training data:
{mined_examples}

Based on the key frames, identify the ACTION TYPE.

ACTION TYPE (choose exactly one):
{action_list}

Respond with ONLY this JSON:
{{"action": "<action type>", "reasoning": "<one sentence about body mechanics in key frame>"}}"""

# ── Row 4: CoS — stage 2 (severity from key frames) ──────────────────────────
COS_SEVERITY_TMPL = """\
You are analyzing a potential football foul from {n_views} camera angles.
View 0 is the live broadcast camera. Views 1+ are replay cameras.

IMPORTANT: The frames shown are the KEY FRAMES most informative for this incident:
{selected_frame_info}

The action type has been identified as: {predicted_action}

{law12_context}

Examples of "{predicted_action}" incidents:
{severity_examples}

SEVERITY CALIBRATION: {prior_str}

Assess the SEVERITY based on force level visible in the key frames:

SEVERITY (choose exactly one):
{severity_list}

Rules:
- EXCESSIVE FORCE / endangering safety → RED CARD
- RECKLESS (disregard for opponent)    → YELLOW CARD
- CARELESS (lack of attention)         → No card but foul
- NO CONTACT / SIMULATION              → No offence

Respond with ONLY this JSON:
{{"severity": "<severity>", "reasoning": "<one sentence citing force level in key frame>"}}"""

# ── NEW Row 5: Full-frame severity (cos_two_stage action + ALL frames severity)
FULL_FRAME_SEVERITY_TMPL = """\
You are assessing the SEVERITY of a football foul from {n_views} camera angles.
View 0 is the live broadcast camera. Views 1+ are replay cameras.

ALL FRAMES from each view are shown below (approach, contact, aftermath).
The action type has already been identified as: {predicted_action}

{law12_context}

The severity depends on the FORCE and DANGER visible across the full sequence:
- Watch the approach speed and body posture BEFORE contact
- Watch the contact moment and degree of force
- Watch the aftermath — did the opponent fall? Were they injured?

Examples of "{predicted_action}" incidents with official severity decisions:
{severity_examples}

SEVERITY CALIBRATION (data-driven): {prior_str}

SEVERITY (choose exactly one):
{severity_list}

- EXCESSIVE FORCE or endangering opponent → RED CARD
- RECKLESS (clear disregard for opponent) → YELLOW CARD
- CARELESS (lack of attention, ball-aimed) → No card
- No meaningful contact or simulation     → No offence

Respond with ONLY this JSON:
{{"severity": "<severity>", "reasoning": "<one sentence citing approach speed or force level>"}}"""

# ── NEW Row 6: Per-action severity prior ─────────────────────────────────────
PER_ACTION_PRIOR_TMPL = """\
You are analyzing a potential football foul from {n_views} camera angles.
View 0 is the live broadcast camera. Views 1+ are replay cameras.

{law12_context}

Here are examples of correctly classified incidents:
{examples}

Now classify the incident shown in the video frames.

ACTION TYPE (choose exactly one):
{action_list}

SEVERITY (choose exactly one):
{severity_list}

IMPORTANT SEVERITY CALIBRATION:
The distribution of severities varies significantly by action type.
After you identify the action, use this data to calibrate your severity judgment:
{per_action_prior_str}

Respond with ONLY this JSON:
{{"action": "<action type>", "severity": "<severity>", "reasoning": "<one sentence>"}}"""

# ── NEW Row 7: Targeted retrieval ────────────────────────────────────────────
TARGETED_RETRIEVAL_TMPL = """\
You are analyzing a potential football foul from {n_views} camera angles.
View 0 is the live broadcast camera. Views 1+ are replay cameras.

{law12_context}

The following are REAL examples most relevant to this specific incident.
They show similar challenges and the boundary between severity levels:

{targeted_examples}

Now classify the NEW incident. Pay special attention to what distinguishes
{predicted_action_hint} from the boundary cases shown above.

ACTION TYPE (choose exactly one):
{action_list}

SEVERITY (choose exactly one):
{severity_list}

Respond with ONLY this JSON:
{{"action": "<action type>", "severity": "<severity>", "reasoning": "<one sentence>"}}"""

# ── NEW Row 8: Ordinal severity ──────────────────────────────────────────────
ORDINAL_SEVERITY_TMPL = """\
You are assessing the SEVERITY of a "{predicted_action}" from {n_views} camera angles.
View 0 is the live broadcast camera. Views 1+ are replay cameras.

{law12_context}

The action type is confirmed: {predicted_action}

Think of severity as an ordinal scale from 0 to 3:
  0 = No offence     (no foul, or simulation)
  1 = No card        (foul, but careless — ball-oriented, limited force)
  2 = Yellow card    (reckless — clear disregard for opponent's safety)
  3 = Red card       (excessive force — endangers opponent, could cause injury)

Key question: WHERE on this scale does this {predicted_action} fall?

Step 1: Was there contact? (No → 0)
Step 2: Was it aimed at the ball? (Yes + limited force → 1)
Step 3: Was there clear recklessness? (Yes → 2)
Step 4: Was force excessive / opponent endangered? (Yes → 3)

Examples of "{predicted_action}" at each severity level:
{severity_examples}

Data: for {predicted_action}, the typical distribution is: {prior_str}

Respond with ONLY this JSON:
{{"severity_level": <0-3>, "severity": "<severity name>", "step_reached": <1-4>, "reasoning": "<one sentence>"}}"""

# ── NEW Row 9: Best combined system ──────────────────────────────────────────
COS_FULL_SEV_TMPL = FULL_FRAME_SEVERITY_TMPL  # reuses full-frame severity for stage 2


# ── NEW Row 10: Disambiguation prompt ────────────────────────────────────────
# Addresses the Elbowing collapse: Standing tackling / Holding / Pushing all
# predicted as Elbowing due to visual similarity in upper body contact frames.
ACTION_DISAMBIGUATION = """\
CRITICAL DISTINCTIONS:
- ELBOWING: arm/elbow aggressively strikes opponent's HEAD or FACE.
- HOLDING: player actively GRABS opponent's SHIRT, ARM or BODY to pull them back.
- PUSHING: player uses HANDS to shove opponent sideways or forward.
- STANDING TACKLING: player extends FOOT to play the BALL while remaining on their feet.
- TACKLING: player SLIDES on the ground to win the BALL.

Final Check: Look closely at the player's hands and feet. If they are grabbing fabric, it is Holding. If they are playing the ball with their foot, it is a Tackle."""
