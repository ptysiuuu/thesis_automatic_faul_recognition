import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score

class VARMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.all_preds_act = []
        self.all_targets_act = []
        self.all_preds_sev = []
        self.all_targets_sev = []

    def update(self, out_sev, out_act, tar_sev, tar_act):
        self.all_preds_sev.extend(torch.argmax(out_sev, dim=1).cpu().numpy())
        self.all_targets_sev.extend(torch.argmax(tar_sev, dim=1).cpu().numpy())
        self.all_preds_act.extend(torch.argmax(out_act, dim=1).cpu().numpy())
        self.all_targets_act.extend(torch.argmax(tar_act, dim=1).cpu().numpy())

    def compute(self):
        bal_acc_act = balanced_accuracy_score(self.all_targets_act, self.all_preds_act)
        bal_acc_sev = balanced_accuracy_score(self.all_targets_sev, self.all_preds_sev)
        lb_value = (bal_acc_act + bal_acc_sev) / 2

        return {
            "Balanced Accuracy (Action)": bal_acc_act,
            "Balanced Accuracy (Severity)": bal_acc_sev,
            "Leaderboard Value": lb_value
        }