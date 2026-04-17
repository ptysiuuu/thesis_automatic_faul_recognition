import torch
import torch.nn as nn
import os
from torchvision import transforms
from dataset import MultiViewDataset
from transformers import VideoMAEImageProcessor
from torch.utils.data import DataLoader

from model import MVNetworkV2
from trainer import VAR_Trainer


def get_videomae_transforms():
    """Pobiera dedykowaną normalizację dla VideoMAE V2."""
    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(
            mean=processor.image_mean,
            std=processor.image_std
        )
    ])

def main():
    config = {
        'rule_weight': 0.05,
        'lr': 1e-4,
        'weight_decay': 0.001,
        'epochs': 50
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Inicjalizacja MVNetworkV2 (VideoMAE + CrossAttention)...")
    model = MVNetworkV2().to(device)

    # Przekazujemy do optymalizatora tylko trenowalne parametry (agregator + głowy)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    transform_aug = transforms.Compose([
        transforms.RandomAffine(degrees=(0, 0), translate=(0.1, 0.1), scale=(0.9, 1)),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.5, saturation=0.5, contrast=0.5),
        transforms.RandomHorizontalFlip(),
    ])

    transform_model = get_videomae_transforms()

    config.update({
        'dataset_path': '/net/tscratch/people/plgaszos/SoccerNet_Data',
        'model_name': 'VAR-AI-VideoMAE',
        'start_frame': 63,
		'end_frame': 87,
		'fps': 17,
        'num_views': 5,
        'num_workers': 16,
        'batch_size': 4
    })

    print("Ładowanie datasetów...")

    dataset_train = MultiViewDataset(
        path=config['dataset_path'],
        start=config['start_frame'],
        end=config['end_frame'],
        fps=config['fps'],
        split='Train',
        num_views=config['num_views'],
        transform=transform_aug,
        transform_model=transform_model
    )
    weights_sev, weights_act = dataset_train.getWeights()

    criterions = [
        nn.CrossEntropyLoss(weight=weights_sev.to(device), label_smoothing=0.1),
        nn.CrossEntropyLoss(weight=weights_act.to(device), label_smoothing=0.1)
    ]
    os.makedirs(config['model_name'], exist_ok=True)
    trainer = VAR_Trainer(model, optimizer, scheduler, criterions, device, config)

    dataset_valid = MultiViewDataset(
        path=config['dataset_path'],
        start=config['start_frame'],
        end=config['end_frame'],
        fps=config['fps'],
        split='Valid',
        num_views=config['num_views'],
        transform=None,
        transform_model=transform_model
    )

    dataset_test = MultiViewDataset(
        path=config['dataset_path'],
        start=config['start_frame'],
        end=config['end_frame'],
        fps=config['fps'],
        split='Test',
        num_views=config['num_views'],
        transform=None,
        transform_model=transform_model
    )

    train_loader = DataLoader(
        dataset_train,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        dataset_valid,
        batch_size=1,
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    test_loader = DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    trainer.fit(train_loader, val_loader, test_loader)

    print(f"Liczba trenowalnych parametrów: {sum(p.numel() for p in trainable_params) / 1e6:.2f} M")
    print("VAR-AI gotowy do pracy z VideoMAE V2.")

if __name__ == "__main__":
    main()