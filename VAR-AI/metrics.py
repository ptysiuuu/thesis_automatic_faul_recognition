from config.classes import INVERSE_EVENT_DICTIONARY

class VARMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.actions = {}

    def update(self, out_sev, out_act, action_ids):
        import torch
        preds_sev = torch.argmax(out_sev.detach().cpu(), dim=1)
        preds_act = torch.argmax(out_act.detach().cpu(), dim=1)

        for i in range(len(action_ids)):
            sev = preds_sev[i].item()
            values = {
                "Action class": INVERSE_EVENT_DICTIONARY["action_class"][preds_act[i].item()],
                "Offence": "No offence" if sev == 0 else "Offence",
                "Severity": ["", "1.0", "3.0", "5.0"][sev]
            }
            self.actions[action_ids[i]] = values

    def save(self, set_name, model_name="VAR-AI"):
        import os, json
        os.makedirs(model_name, exist_ok=True)
        data = {"Set": set_name, "Actions": self.actions}
        path = os.path.join(model_name, f"predictions_{set_name}.json")
        with open(path, "w") as f:
            json.dump(data, f)
        return path