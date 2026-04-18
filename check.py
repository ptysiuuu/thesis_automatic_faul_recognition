import json
with open("../SoccerNet/SN-MVFouls-2025/train_720p/annotations.json") as f:
    d = json.load(f)
print(list(d['Actions'][list(d['Actions'].keys())[0]].keys()))