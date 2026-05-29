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

# ── Row 2b: Description-first — stage 1 (physical description only) ──────────
DESCRIPTION_FIRST_DESCRIPTION_TMPL = """\
You are analyzing a football incident from {n_views} camera angles.
View 0 is the live broadcast camera. Views 1+ are replay cameras.

Describe only the visible physical interaction in factual terms.
Focus on:
- how fast the challenge is moving
- how high the foot, knee, or arm is raised
- whether the challenging player seems in control or out of control
- what happens to the opponent: fall, stumble, or no visible effect
- whether the movement appears aimed at the ball

Write a short factual description. Do not judge the incident.
"""

# ── Row 2b: Description-first — stage 2 (severity from description) ──────────
DESCRIPTION_FIRST_SEVERITY_TMPL = """\
You are classifying a football incident from a written description instead of video frames.

The action class is: {action_class}

{law12_context}

Description:
{description}

Use IFAB Law 12 criteria explicitly:
- careless -> no card
- reckless -> yellow card
- excessive force -> red card
- no meaningful contact or simulation -> no offence

Decide the severity from the description only.

SEVERITY (choose exactly one):
{severity_list}

Respond with ONLY this JSON:
{{"severity": "<severity>", "reasoning": "<one sentence>"}}"""

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

Before giving your final answer, reason step by step:
STEP 1 - BODY PART: What body part does the challenging player use? (foot, leg, elbow, arm, hand, shoulder, body)
STEP 2 - TARGET: What part of the opponent is affected? (legs, body, arm, head/face)
STEP 3 - BALL: Is the challenging player moving toward the ball or toward the opponent?
STEP 4 - MOTION: Describe the movement — slide, swing, push, grab, jump, raise leg, fall?
STEP 5 - ACTION: Based on steps 1-4, which action type fits best?

Respond with ONLY this JSON — the reasoning field must summarize your step-by-step thinking:
{{"action": "<action type>", "reasoning": "<summary of steps 1-4 leading to this conclusion>"}}"""

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


# ── NEW: Contrastive severity anchoring prompt ─────────────────────────────────
CONTRASTIVE_SEVERITY_TMPL = """\
You are assessing the relative FORCE of a football incident from {n_views} camera angles.
View 0 is the live broadcast camera. Views 1+ are replay cameras.

{law12_context}

You are shown THREE visual items:
- Reference example 1 (Example A): an incident labeled as "{anchor_a_sev}" for the action type "{action}".
- Reference example 2 (Example B): an incident labeled as "{anchor_b_sev}" for the action type "{action}".
- The TEST CLIP: the new incident to classify (the video frames shown).

Question: The FORCE level in the TEST CLIP is CLOSER to Example 1 or Example 2?

Respond with ONLY this JSON: {{"choice": 1}} or {{"choice": 2}}
"""


# ── NEW: Physics-grounded flow prompt (used when flow-based stats are available)
PHYSICS_FLOW_TMPL = """\
You are given a short video clip and a quantitative estimate of opponent displacement
measured in pixels per frame (max_displacement={max_disp:.2f}).

Use IFAB Law 12 alongside this measurement to decide severity for a "{action}" incident.

Decide severity (one of: {severity_list}) and explain in one sentence referencing the displacement.

Respond with ONLY this JSON: {{"severity": "<severity>", "reasoning": "<one sentence>"}}
"""

# ── Multi-Turn VAR Strategy ───────────────────────────────────────────────────
# Implements the 3-turn structured strategy for multi-view foul classification.
# Each turn has a distinct epistemic purpose: perception → classification → severity.

MULTI_TURN_VAR_SYSTEM_CONTEXT = """\
=== VIDEO ASSISTANT REFEREE ASSESSMENT CONTEXT ===

ROLE: You are a Video Assistant Referee (VAR). You have access to a multi-camera \
replay system and are examining a single incident from several synchronized \
perspectives. You must form an independent assessment of both the foul type and \
its severity.

EIGHT FOUL CATEGORIES — with distinguishing criteria:

Tackling            Player's body goes to ground or lunges HORIZONTALLY toward the ball.
                    KEY: horizontal/sliding momentum. Foot/leg sweeps along the ground.

Standing tackling   Player challenges for the ball while remaining UPRIGHT on both feet.
                    KEY: vertical posture maintained. Foot extends to ball, no slide.

High leg            Foot raised ABOVE OPPONENT'S WAIST HEIGHT during the challenge.
                    KEY: extreme leg elevation. Not a slide — the raised foot is the danger.

Holding             SUSTAINED GRIP of opponent's shirt, arm, or body (not momentary touch).
                    KEY: fingers grabbing fabric or a limb. Contact persists over time.

Pushing             OPEN PALM, forearm, or torso thrust that shoves the opponent.
                    KEY: force directed sideways or forward; not a shoulder-to-shoulder charge.

Elbowing            Elbow is the PRIMARY CONTACT POINT, deliberately raised to strike head/face.
                    KEY: elbow elevated as a weapon. Incidental arm contact during a Challenge
                    does NOT qualify.

Challenge           SHOULDER-TO-SHOULDER body contest using the upper arm in legal position.
                    KEY: legal shoulder charge competing for the ball. Body-to-body, not elbow.

Dive                Fouled player EXAGGERATES or FABRICATES contact. Fall is disproportionate.
                    KEY: the FALLING player is the accused, not the defender. Contact is minimal
                    or non-existent relative to the fall.

ANTI-COLLAPSE WARNINGS:
• Standing tackling + Tackling together cover ~45 % of incidents — do NOT default to
  these without confirming the body posture. If the challenging player's body goes to
  ground → Tackling. If they stay upright → Standing tackling.
• When a player falls but contact appears minimal or non-existent → consider Dive first.
  The fall being disproportionate to the contact force is the Dive diagnostic.

FOUR SEVERITY LEVELS — IFAB Law 12 language:

0 = No Offence   Action does not violate the Laws of the Game. Contact was legal or
                 the player played the ball cleanly. No free kick.

1 = No Card      Player shows lack of attention or consideration when making the
(Careless)       challenge. A free kick is awarded but no disciplinary sanction.

2 = Yellow Card  Player acts with disregard for the danger to, or consequences for,
(Reckless)       the opponent. Must be cautioned.

3 = Red Card     Player uses excessive force or endangers the safety of an opponent.
(Violent)        Must be sent off.

KEY SEVERITY DETERMINANTS:
• Did the player attempt to play the ball? → MITIGATING (pushes toward No Card boundary)
• Was the opponent's safety endangered?   → AGGRAVATING (pushes toward Red Card)
• Attempting to play the ball does NOT reduce a Red Card when force is excessive.
"""

# ── Turn 1: spatial and temporal context establishment ────────────────────────
MULTI_TURN_TURN1_TMPL = """\
{system_context}

=== TURN 1 OF 3 — PHYSICAL DESCRIPTION (no classification yet) ===

You are shown {n_frames} frames in the exact order listed below.
Each frame label states: [View | Frame index | Temporal zone].
The frame marked "← CONTACT FRAME" is the primary evidence frame.
Ignore any generic "Live camera" label the system may add — use the labels below:

{frame_labels}

{law12_context}

YOUR TASK — describe ONLY the visible physical interaction at the CONTACT FRAME.
Do NOT classify the foul type. Do NOT suggest a card. Just describe what you see.

Answer these five questions in order:

1. INITIATOR   Which player initiates the challenge? (e.g. left attacker, right defender)
2. BODY PART   What body part does the challenging player use for contact?
               (foot / knee / thigh / elbow / forearm / hand / shoulder / chest / torso)
3. CONTACT     Where on the opponent's body does the contact land?
               (feet / lower leg / thigh / hip / ribs/torso / arm / shoulder / head/face)
4. BALL        Is the ball within playing distance at the moment of contact? (yes / no / unclear)
5. POSTURE     Describe the challenging player's posture:
               (horizontal slide / upright standing / jumping / from behind / from the side)

If the contact is obscured in the CONTACT FRAME state which view and time window
would better reveal it (e.g. "Replay 1 at immediate-aftermath frame would show the fall").

Write a factual description — no verdict about fouls, cards, or action types.
"""

# ── Turn 2: forced-coverage action classification ─────────────────────────────
MULTI_TURN_TURN2_TMPL = """\
=== TURN 2 OF 3 — FOUL TYPE CLASSIFICATION (FORCED COVERAGE) ===

In Turn 1 you provided this physical description:
--- TURN 1 DESCRIPTION ---
{turn1_description}
--- END ---

You are now shown {n_frames} frames tightly clustered around the contact moment
from the closest replay camera. Frames appear in order:

{frame_labels}

FORCED-COVERAGE CHECKLIST — complete ALL eight entries before deciding.
For each class respond Yes / No / Uncertain with a one-phrase justification.
Evaluate them in the order shown (groups visually similar classes together):

  Tackling:           [Yes/No/Uncertain] — [player's body goes to ground horizontally?]
  Standing tackling:  [Yes/No/Uncertain] — [foot extends to ball while player stays upright?]
  High leg:           [Yes/No/Uncertain] — [foot raised above opponent's waist?]
  Challenge:          [Yes/No/Uncertain] — [shoulder-to-shoulder body contest for ball?]
  Pushing:            [Yes/No/Uncertain] — [open palm/forearm/torso shove sideways/forward?]
  Holding:            [Yes/No/Uncertain] — [sustained grip of shirt/arm/body?]
  Elbowing:           [Yes/No/Uncertain] — [elbow primary contact point at head/face?]
  Dive:               [Yes/No/Uncertain] — [fall clearly disproportionate to contact force?]

ANTI-MAJORITY-CLASS CHECK: Standing tackling and Tackling are NOT the automatic default.
If classifying as either, verify: (a) player's momentum was toward the ball, and
(b) the challenge was a deliberate attempt to win the ball — not a mistimed swing.

DIVE CHECK: When a player falls but contact appears minimal or non-existent, consider Dive.
Dive means the falling player exaggerated or fabricated the contact. The fall is
disproportionate to the actual force — look at the aftermath frames for confirmation.

Respond with ONLY this JSON (complete the checklist field before writing "action"):
{{
  "checklist": {{
    "Tackling":          "<Yes/No/Uncertain> — <one phrase>",
    "Standing tackling": "<Yes/No/Uncertain> — <one phrase>",
    "High leg":          "<Yes/No/Uncertain> — <one phrase>",
    "Challenge":         "<Yes/No/Uncertain> — <one phrase>",
    "Pushing":           "<Yes/No/Uncertain> — <one phrase>",
    "Holding":           "<Yes/No/Uncertain> — <one phrase>",
    "Elbowing":          "<Yes/No/Uncertain> — <one phrase>",
    "Dive":              "<Yes/No/Uncertain> — <one phrase>"
  }},
  "action": "<exactly one of: Tackling, Standing tackling, High leg, Holding, Pushing, Elbowing, Challenge, Dive>",
  "reasoning": "<one sentence citing the key visual evidence from the Turn 1 description and these frames>"
}}"""

# ── Turn 3: ordinal severity cascade ─────────────────────────────────────────
MULTI_TURN_TURN3_TMPL = """\
=== TURN 3 OF 3 — SEVERITY ASSESSMENT (ORDINAL CASCADE) ===

Turn 2 action classification: {action_type}

Physical description from Turn 1:
--- TURN 1 DESCRIPTION ---
{turn1_description}
--- END ---

You are now shown {n_frames} frames emphasising the AFTERMATH of the contact.
Use them to observe fall trajectory, fall distance, and recovery:

{frame_labels}

{law12_context}

Work through the four-step IFAB severity cascade for "{action_type}":

STEP 1 — OFFENCE THRESHOLD
Was there a violation of the Laws of the Game?
Consider: did the player play the ball cleanly, or was there illegal contact?
→ If NO offence: answer "No Offence" and proceed to the final JSON.

STEP 2 — FORCE QUANTIFICATION (only if Step 1 = offence)
Characterise the force level:
  (a) approach speed and momentum of the challenging player (use approach frames)
  (b) whether contact was incidental or the primary intent of the challenge
  (c) vulnerability of the body part contacted (head/face = higher danger)
  (d) what happened to the opponent in the aftermath frames above
→ Rate: minimal / moderate / excessive

STEP 3 — SEVERITY MAPPING (IFAB Law 12)
  Minimal force                                → No Card   (careless, no sanction)
  Moderate force + disregard for opponent      → Yellow Card (reckless, caution)
  Excessive force or endangering safety        → Red Card  (violent, send-off)

STEP 4 — BALL-PLAY CONSIDERATION
Did the player attempt to play the ball?
Attempting to play the ball MITIGATES toward No Card / Yellow boundary,
but does NOT reduce a Red Card where force is already excessive.

{prior_context}

Respond with ONLY this JSON:
{{
  "step1_offence":    "<yes — foul committed / no — clean play>",
  "step2_force":      "<minimal / moderate / excessive / N/A>",
  "step3_ifab":       "<No offence / No card / Yellow card / Red card>",
  "step4_ball_play":  "<yes / no / unclear>",
  "severity":         "<No offence / No card / Yellow card / Red card>",
  "severity_level":   <0, 1, 2, or 3>,
  "reasoning":        "<one sentence citing approach speed and/or aftermath evidence>"
}}"""
