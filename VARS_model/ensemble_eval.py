import os
import json
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import MultiViewDataset
from model import MVNetwork
from torchvision.models.video import MViT_V2_S_Weights
from SoccerNet.Evaluation.MV_FoulRecognition import evaluate
from config.classes import INVERSE_EVENT_DICTIONARY
import torch.nn as nn

# ── Ścieżki do wag ──────────────────────────────────────────
PATH_BASELINE = "models/VARS/5/mvit_v2_s/0.0001/_B2_F16_S_G0.1_Step3/baseline_model.tar"
PATH_TRANSFORMER = "models/VARS_step8/5/mvit_v2_s/0.0001/_B4_F16_S_G0.5_Step20/best_model.tar"
PATH_DATA = "/net/tscratch/people/plgaszos/SoccerNet_Data"
# ────────────────────────────────────────────────────────────

transforms_model = MViT_V2_S_Weights.KINETICS400_V1.transforms()

dataset_Test = MultiViewDataset(
    path=PATH_DATA, start=63, end=87, fps=17,
    split='Test', num_views=5,
    transform_model=transforms_model
)
test_loader = DataLoader(dataset_Test, batch_size=1, shuffle=False, num_workers=4)

# Ładuj oba modele
model_max = MVNetwork(net_name='mvit_v2_s', agr_type='attention').cuda()

# --- REKONSTRUKCJA STAREJ GŁOWY Z TWOJEGO ORYGINALNEGO KODU ---
feat_dim = 400 # Wymiar ze starych wag

model_max.mvnetwork.fc_offence = nn.Sequential(
    nn.LayerNorm(feat_dim),          # warstwa 0
    nn.Linear(feat_dim, feat_dim),   # warstwa 1
    nn.Linear(feat_dim, 4)           # warstwa 2
).cuda()

model_max.mvnetwork.fc_action = nn.Sequential(
    nn.LayerNorm(feat_dim),          # warstwa 0
    nn.Linear(feat_dim, feat_dim),   # warstwa 1
    nn.Linear(feat_dim, 8)           # warstwa 2
).cuda()
# -------------------------------------------------------------

# Teraz stare wagi z .tar załadują się idealnie:
model_max.load_state_dict(torch.load(PATH_BASELINE)['state_dict'])
model_max.eval()

model_tr = MVNetwork(net_name='mvit_v2_s', agr_type='transformer').cuda()
model_tr.load_state_dict(torch.load(PATH_TRANSFORMER)['state_dict'])
model_tr.eval()

# Ewaluacja ensemble
data = {"Set": "test"}
actions = {}

with torch.no_grad():
    for _, _, mvclips, action in test_loader:
        mvclips = mvclips.cuda().float()

        # Logity z obu modeli
        sev1, act1, _ = model_max(mvclips)
        sev2, act2, _ = model_tr(mvclips)

        # Uśrednienie logitów (można też softmax przed uśrednieniem)
        sev_avg = (sev1 + sev2) / 2
        act_avg = (act1 + act2) / 2

        preds_sev = torch.argmax(sev_avg.cpu(), dim=1)
        preds_act = torch.argmax(act_avg.cpu(), dim=1)

        for i in range(len(action)):
            sev = preds_sev[i].item()
            values = {
                "Action class": INVERSE_EVENT_DICTIONARY["action_class"][preds_act[i].item()],
                "Offence": "No offence" if sev == 0 else "Offence",
                "Severity": ["", "1.0", "3.0", "5.0"][sev]
            }
            actions[action[i]] = values

data["Actions"] = actions
with open("predictions_ensemble.json", "w") as f:
    json.dump(data, f)

results = evaluate(
    os.path.join(PATH_DATA, "Test", "annotations.json"),
    "predictions_ensemble.json"
)
print("ENSEMBLE TEST")
print(results)