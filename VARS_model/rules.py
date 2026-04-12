import torch

# config/classes.py
ACTION_HIGH_LEG = 2
ACTION_HOLDING  = 3
ACTION_PUSHING  = 4
ACTION_ELBOWING = 5
ACTION_DIVE     = 7

def rule_loss_with_stats(logits_severity, logits_action, weight=0.05):
    prob_sev = torch.softmax(logits_severity, dim=1)
    prob_act = torch.softmax(logits_action, dim=1)

    total_loss = torch.tensor(0.0, device=logits_severity.device)
    stats = {}

    with torch.no_grad():
        # Progi do uznania, że model "wybrał" daną klasę
        pred_sev = torch.argmax(prob_sev, dim=1)
        pred_act = torch.argmax(prob_act, dim=1)

    # R1: Dive (7) vs Offence (>0)
    p_dive = prob_act[:, ACTION_DIVE]
    p_not_no_offence = 1.0 - prob_sev[:, 0]
    total_loss += (p_dive * p_not_no_offence).mean()
    stats['R1_Dive_Violation'] = ((pred_act == ACTION_DIVE) & (pred_sev > 0)).sum().item()

    # R2 & R3: Violent (2, 5) vs Low Severity (0, 1)
    p_violent = prob_act[:, ACTION_HIGH_LEG] + prob_act[:, ACTION_ELBOWING]
    p_low_sev = prob_sev[:, 0] + prob_sev[:, 1]
    total_loss += (p_violent * p_low_sev).mean()
    stats['R2_3_Violent_LowSev'] = (((pred_act == ACTION_HIGH_LEG) | (pred_act == ACTION_ELBOWING)) & (pred_sev <= 1)).sum().item()

    # R5: Red Card (3) vs Mild Action (3, 4)
    p_red_card = prob_sev[:, 3]
    p_mild_act = prob_act[:, ACTION_HOLDING] + prob_act[:, ACTION_PUSHING]
    total_loss += (p_red_card * p_mild_act).mean()
    stats['R5_RedCard_MildAction'] = ((pred_sev == 3) & ((pred_act == ACTION_HOLDING) | (pred_act == ACTION_PUSHING))).sum().item()

    return weight * total_loss, stats