import argparse
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer


def build_prompts():
    return {
        0: [
            "A clean challenge, ball-first contact, fair play, no foul.",
            "Legal tackle with no illegal contact, referee allows play to continue.",
            "Natural movement, shoulder-to-shoulder, no offence committed.",
            "Player wins the ball cleanly, no careless or reckless action.",
        ],
        1: [
            "Careless trip or push, minor contact, foul but no card.",
            "Late challenge with slight contact, careless but not reckless.",
            "Minor infringement, not dangerous, free kick only.",
            "Small holding or obstruction, no caution required.",
        ],
        2: [
            "Reckless tackle, excessive force, caution for a yellow card.",
            "Studs showing, reckless challenge, endangering an opponent.",
            "Late challenge with force, reckless but not serious foul play.",
            "Reckless challenge, opponent endangered, yellow card offence.",
        ],
        3: [
            "Serious foul play with excessive force, red card.",
            "Violent conduct, striking or elbowing, send-off.",
            "Studs up, two-footed lunge, endangering safety, red card.",
            "Brutal challenge, endangering safety, referee shows red card.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract CLIP severity class embeddings"
    )
    parser.add_argument(
        "--output",
        default="clip_embedings",
        type=str,
        help="Output path for the embeddings tensor",
    )
    parser.add_argument(
        "--model_name",
        default="openai/clip-vit-base-patch32",
        type=str,
        help="HuggingFace CLIP model id",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        type=str,
    )
    args = parser.parse_args()

    prompts = build_prompts()
    model = CLIPModel.from_pretrained(args.model_name)
    tokenizer = CLIPTokenizer.from_pretrained(args.model_name)
    model.to(args.device)
    model.eval()

    class_embeddings = []

    with torch.no_grad():
        for class_id in range(4):
            text = prompts[class_id]
            inputs = tokenizer(text, padding=True, return_tensors="pt")
            inputs = {k: v.to(args.device) for k, v in inputs.items()}
            text_features = model.get_text_features(**inputs)
            text_features = F.normalize(text_features, dim=-1)
            centroid = text_features.mean(dim=0)
            centroid = F.normalize(centroid, dim=-1)
            class_embeddings.append(centroid)

    embeddings = torch.stack(class_embeddings, dim=0)
    similarity = embeddings @ embeddings.T

    print("Cosine similarity matrix:")
    for i in range(4):
        row = " ".join([f"{similarity[i, j].item():.3f}" for j in range(4)])
        print(row)

    payload = {
        "embeddings": embeddings.cpu(),
        "prompts": prompts,
        "model_name": args.model_name,
    }
    torch.save(payload, args.output)
    print(f"Saved embeddings to {args.output}")


if __name__ == "__main__":
    main()
