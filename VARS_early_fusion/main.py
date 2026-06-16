import os
import logging
import time
import warnings
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from SoccerNet.Evaluation.MV_FoulRecognition import evaluate

from dataset import MultiViewDataset
from model import MVNetwork, EarlyFusionNetwork
from train import trainer, evaluation, EMA, UncertaintyWeighting
from config.classes import EVENT_DICTIONARY, INVERSE_EVENT_DICTIONARY
from torchvision.models.video import (
    R3D_18_Weights,
    MC3_18_Weights,
    R2Plus1D_18_Weights,
    S3D_Weights,
    MViT_V2_S_Weights,
    MViT_V1_B_Weights,
)
from model import HF_VIDEOMAE_REGISTRY

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

_TORCHVISION_MODELS = {
    "r3d_18",
    "mc3_18",
    "r2plus1d_18",
    "s3d",
    "mvit_v2_s",
    "mvit_v1_b",
    "tadaformer_b16",
    "tadaformer_l14",
    "aim_vitb16",
    "vjepa21_vitb",
    "intern_video2_dist_b",
    "vjepa2_vitl_256",
}
HIERA_MODELS = {
    "hiera_base_16x224",
    "hiera_base_plus_16x224",
    "hiera_large_16x224",
    "hiera_huge_16x224",
}
_ALL_MODELS = _TORCHVISION_MODELS | set(HF_VIDEOMAE_REGISTRY.keys()) | HIERA_MODELS


# ---------------------------------------------------------------------------
# VideoMAE normalisation transform
# ---------------------------------------------------------------------------
class VideoMAETransform:
    def __init__(self, size: int = 224):
        self.resize = transforms.Resize(size, antialias=True)
        self.crop = transforms.CenterCrop(size)
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, C, H, W]  values in [0, 1]
        x = self.resize(x)
        x = self.crop(x)
        x = self.normalize(x)
        return x


class TAdaFormerTransform:
    """CLIP normalisation used by TAdaFormer pretraining."""

    def __init__(self, size: int = 224):
        self.resize = transforms.Resize(size, antialias=True)
        self.crop = transforms.CenterCrop(size)
        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, C, H, W] in [0, 1]
        x = self.resize(x)
        x = self.crop(x)
        x = self.normalize(x)
        return x


class HieraTransform:
    """Kinetics-400 normalization used by Hiera video inference."""

    def __init__(self, size: int = 224):
        self.resize = transforms.Resize(size, antialias=True)
        self.crop = transforms.CenterCrop(size)
        self.normalize = transforms.Normalize(
            mean=[0.45, 0.45, 0.45],
            std=[0.225, 0.225, 0.255],
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, C, H, W] in [0, 1]
        x = self.resize(x)
        x = self.crop(x)
        x = self.normalize(x)
        return x


class VJEPA21Transform:
    def __init__(self, size: int = 384):
        self.resize = transforms.Resize(size, antialias=True)
        self.crop = transforms.CenterCrop(size)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __call__(self, x):
        x = self.resize(x)
        x = self.crop(x)
        x = self.normalize(x)
        return x


class InternVideo2Transform:
    """ImageNet normalisation for InternVideo2 dist-B/14."""

    def __init__(self, size: int = 224):
        self.resize = transforms.Resize(size, antialias=True)
        self.crop = transforms.CenterCrop(size)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalize(self.crop(self.resize(x)))


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def checkArguments(args):
    if not (1 <= args.num_views <= 5):
        raise ValueError("--num_views must be between 1 and 5")
    if args.data_aug not in ("Yes", "No"):
        raise ValueError("--data_aug must be 'Yes' or 'No'")
    if args.aug_preset not in ("none", "default", "strong"):
        raise ValueError("--aug_preset must be 'none', 'default', or 'strong'")
    if args.weighted_loss not in ("Yes", "No"):
        raise ValueError("--weighted_loss must be 'Yes' or 'No'")
    if not args.fusion_mode:
        if args.pooling_type not in (
            "max",
            "attention",
            "transformer",
            "crossattn",
            "gat",
            "bidir_crossattn",
            "dynagat",
            "cva",
            "mamba",
            "hybrid",
        ):
            raise ValueError(
                "--pooling_type must be one of: max, attention, transformer, crossattn, gat, bidir_crossattn, dynagat, cva, mamba"
            )
    if args.pre_model not in _ALL_MODELS:
        raise ValueError(f"--pre_model must be one of: {sorted(_ALL_MODELS)}")
    if args.graph_topology not in ("structured", "fully_connected", "replay_only"):
        raise ValueError(
            "--graph_topology must be one of: structured, fully_connected, replay_only"
        )
    if not (0 <= args.start_frame <= 124):
        raise ValueError("--start_frame must be 0-124")
    if not (1 <= args.end_frame <= 125):
        raise ValueError("--end_frame must be 1-125")
    if args.end_frame - args.start_frame < 2:
        raise ValueError("end_frame - start_frame must be >= 2")
    if not (1 <= args.fps <= 25):
        raise ValueError("--fps must be 1-25")
    if args.clip_weight < 0:
        raise ValueError("--clip_weight must be >= 0")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    if args.hf_cache_dir:
        os.environ["HF_HOME"] = args.hf_cache_dir
        os.environ["TRANSFORMERS_CACHE"] = args.hf_cache_dir

    LR = args.LR
    gamma = args.gamma
    step_size = args.step_size
    start_frame = args.start_frame
    end_frame = args.end_frame
    weight_decay = args.weight_decay
    model_name = args.model_name
    pre_model = args.pre_model
    num_views = args.num_views
    fps = args.fps
    batch_size = args.batch_size
    data_aug = args.data_aug
    path = args.data_dir if args.data_dir else args.path
    output_dir = args.output_dir
    pooling_type = args.pooling_type
    weighted_loss = args.weighted_loss
    max_num_worker = args.max_num_worker
    max_epochs = args.max_epochs
    only_evaluation = args.only_evaluation
    path_to_model_weights = args.path_to_model_weights
    aux_weight = args.aux_weight
    ema_decay = args.ema_decay
    use_tta = args.use_tta
    balanced_sampler = args.balanced_sampler
    fusion_mode = args.fusion_mode
    graph_topology = args.graph_topology
    cascade_severity = args.cascade_severity
    uncertainty_weighting = args.uncertainty_weighting
    freeze_epoch = args.freeze_epoch
    use_text_bridge = args.use_text_bridge
    train_all_but_test = args.train_all_but_test
    clip_weight = args.clip_weight
    clip_embeddings_path = args.clip_embeddings_path
    use_mffm = args.use_mffm
    use_decoder = args.use_decoder
    descriptions_path = args.descriptions_path
    use_jepa = args.use_jepa
    jepa_lambda_max = args.jepa_lambda_max
    jepa_warmup_epochs = args.jepa_warmup_epochs

    number_of_frames = int(
        (end_frame - start_frame) / (((end_frame - start_frame) / 25) * fps)
    )

    # Use "early_fusion" as the backbone identifier in the model path when fusing.
    backbone_id = "early_fusion" if fusion_mode else pre_model

    # --- logging ---
    os.makedirs(
        os.path.join(
            output_dir,
            model_name,
            str(num_views),
            backbone_id,
            str(LR),
            f"_B{batch_size}_F{number_of_frames}_G{gamma}_Step{step_size}",
        ),
        exist_ok=True,
    )
    best_model_path = os.path.join(
        output_dir,
        model_name,
        str(num_views),
        backbone_id,
        str(LR),
        f"_B{batch_size}_F{number_of_frames}_G{gamma}_Step{step_size}",
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(best_model_path, "logging.log")),
            logging.StreamHandler(),
        ],
    )

    # --- augmentation ---
    aug_preset = args.aug_preset
    if aug_preset == "default":
        transformAug = transforms.Compose(
            [
                transforms.RandomAffine(
                    degrees=(0, 0), translate=(0.1, 0.1), scale=(0.9, 1)
                ),
                transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
                transforms.RandomRotation(degrees=5),
                transforms.ColorJitter(brightness=0.5, saturation=0.5, contrast=0.5),
                transforms.RandomHorizontalFlip(),
            ]
        )
    elif aug_preset == "strong":
        transformAug = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
                transforms.RandomAffine(
                    degrees=(0, 0), translate=(0.2, 0.2), scale=(0.8, 1.2)
                ),
                transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.6, saturation=0.6, contrast=0.6),
                transforms.RandomHorizontalFlip(),
                transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            ]
        )
    else:  # aug_preset == "none"
        transformAug = None

    # --- backbone-specific transforms ---
    _videomae_transform = VideoMAETransform(size=224)
    _transform_map = {
        "r3d_18": R3D_18_Weights.KINETICS400_V1.transforms(),
        "mc3_18": MC3_18_Weights.KINETICS400_V1.transforms(),
        "r2plus1d_18": R2Plus1D_18_Weights.KINETICS400_V1.transforms(),
        "s3d": S3D_Weights.KINETICS400_V1.transforms(),
        "mvit_v2_s": MViT_V2_S_Weights.KINETICS400_V1.transforms(),
        "mvit_v1_b": MViT_V1_B_Weights.KINETICS400_V1.transforms(),
        **{k: _videomae_transform for k in HF_VIDEOMAE_REGISTRY},
        "tadaformer_b16": TAdaFormerTransform(size=224),
        "tadaformer_l14": TAdaFormerTransform(size=224),
        "aim_vitb16": TAdaFormerTransform(size=224),
        **{k: HieraTransform(size=224) for k in HIERA_MODELS},
        "vjepa21_vitb": VJEPA21Transform(size=384),
        "intern_video2_dist_b": InternVideo2Transform(size=224),
        "vjepa2_vitl": VJEPA21Transform(size=384),
    }
    # Early fusion always uses MViT-v2-S weights transforms (backbone is hardcoded).
    transforms_model = (
        MViT_V2_S_Weights.KINETICS400_V1.transforms()
        if fusion_mode
        else _transform_map.get(
            pre_model, R2Plus1D_18_Weights.KINETICS400_V1.transforms()
        )
    )

    # --- datasets & loaders ---
    dataset_kwargs = dict(
        path=path,
        start=start_frame,
        end=end_frame,
        fps=fps,
        transform_model=transforms_model,
        fusion_mode=fusion_mode,
        backbone_type=pre_model,
        descriptions_path=descriptions_path,
        train_sample_views=args.train_sample_views,
    )

    if only_evaluation == 0:
        dataset_Test = MultiViewDataset(**dataset_kwargs, split="Test", num_views=5)
        test_loader = torch.utils.data.DataLoader(
            dataset_Test,
            batch_size=1,
            shuffle=False,
            num_workers=max_num_worker,
            pin_memory=True,
        )

    elif only_evaluation == 1:
        dataset_Chall = MultiViewDataset(**dataset_kwargs, split="Chall", num_views=5)
        chall_loader = torch.utils.data.DataLoader(
            dataset_Chall,
            batch_size=1,
            shuffle=False,
            num_workers=max_num_worker,
            pin_memory=True,
        )

    elif only_evaluation == 2:
        dataset_Test = MultiViewDataset(**dataset_kwargs, split="Test", num_views=5)
        dataset_Chall = MultiViewDataset(**dataset_kwargs, split="Chall", num_views=5)
        test_loader = torch.utils.data.DataLoader(
            dataset_Test,
            batch_size=1,
            shuffle=False,
            num_workers=max_num_worker,
            pin_memory=True,
        )
        chall_loader = torch.utils.data.DataLoader(
            dataset_Chall,
            batch_size=1,
            shuffle=False,
            num_workers=max_num_worker,
            pin_memory=True,
        )

    else:  # training mode (only_evaluation == 3)
        dataset_Train = MultiViewDataset(
            **dataset_kwargs, split="Train", num_views=num_views, transform=transformAug
        )
        dataset_Valid = MultiViewDataset(
            **dataset_kwargs, split="Valid", num_views=num_views
        )
        dataset_Test = MultiViewDataset(
            **dataset_kwargs, split="Test", num_views=num_views
        )
        if train_all_but_test:
            # Combine Train+Valid into one training set, use Test as val proxy
            from torch.utils.data import ConcatDataset, WeightedRandomSampler

            dataset_Valid_aug = MultiViewDataset(
                **dataset_kwargs,
                split="Valid",
                num_views=num_views,
                transform=transformAug,
            )
            combined = ConcatDataset([dataset_Train, dataset_Valid_aug])

            if balanced_sampler == "Yes":
                # Build combined sample weights from severity labels.
                train_weights = dataset_Train.get_severity_classes()
                valid_weights = dataset_Valid.get_severity_classes()
                all_classes = torch.tensor(train_weights + valid_weights)
                counts = torch.bincount(all_classes)
                class_weights = 1.0 / counts.float().clamp(min=1)
                sample_weights = class_weights[all_classes]
                sampler = WeightedRandomSampler(
                    sample_weights,
                    len(sample_weights),
                    replacement=True,
                )
                train_loader = torch.utils.data.DataLoader(
                    combined,
                    batch_size=batch_size,
                    sampler=sampler,
                    num_workers=max_num_worker,
                    pin_memory=True,
                )
            else:
                train_loader = torch.utils.data.DataLoader(
                    combined,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=max_num_worker,
                    pin_memory=True,
                )

            # Use Test as validation proxy for early stopping
            val_loader = torch.utils.data.DataLoader(
                dataset_Test,
                batch_size=1,
                shuffle=False,
                num_workers=max_num_worker,
                pin_memory=True,
            )
            test_loader = val_loader  # same loader, evaluated as "test"
            logging.info(
                "train_all_but_test: training on Train+Valid, evaluating on Test"
            )

        else:
            if balanced_sampler == "Yes":
                sampler = dataset_Train.get_balanced_sampler()
                train_loader = torch.utils.data.DataLoader(
                    dataset_Train,
                    batch_size=batch_size,
                    sampler=sampler,
                    num_workers=max_num_worker,
                    pin_memory=True,
                )
            else:
                train_loader = torch.utils.data.DataLoader(
                    dataset_Train,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=max_num_worker,
                    pin_memory=True,
                )

            val_loader = torch.utils.data.DataLoader(
                dataset_Valid,
                batch_size=1,
                shuffle=False,
                num_workers=max_num_worker,
                pin_memory=True,
            )
            test_loader = torch.utils.data.DataLoader(
                dataset_Test,
                batch_size=1,
                shuffle=False,
                num_workers=max_num_worker,
                pin_memory=True,
            )

    # --- model ---
    if fusion_mode:
        if use_mffm or use_decoder:
            logging.warning(
                "--use_mffm/--use_decoder are ignored in --fusion_mode"
            )
        model = EarlyFusionNetwork(
            num_views=num_views,
            T_per_view=number_of_frames,
            cascade_severity=cascade_severity,
        ).cuda()
        backbone_prefix = "backbone._base."
        logging.info(
            f"Early fusion mode: EarlyFusionNetwork (num_views={num_views}, "
            f"T={number_of_frames}, cascade_severity={cascade_severity})"
        )
    else:
        model = MVNetwork(
            net_name=pre_model,
            agr_type=pooling_type,
            graph_topology=graph_topology,
            cascade_severity=cascade_severity,
            use_text_bridge=use_text_bridge,
            clip_embeddings_path=clip_embeddings_path,
            use_mffm=use_mffm,
            use_decoder=use_decoder,
            use_jepa=use_jepa,
            backbone_ckpt_dir=args.backbone_ckpt_dir,
            backbone_num_frames=args.backbone_num_frames,
        ).cuda()
        backbone_prefix = "aggregation_model.model."
        logging.info(
            f"Multi-view mode: MVNetwork (backbone={pre_model}, agr={pooling_type}, "
            f"topology={graph_topology if pooling_type == 'gat' else 'n/a'}, "
            f"cascade_severity={cascade_severity}, text_bridge={use_text_bridge}, "
            f"mffm={use_mffm}, decoder={use_decoder})"
        )

        if use_decoder and pooling_type != "transformer":
            logging.warning(
                "--use_decoder expects pooling_type='transformer' for token access"
            )
        if use_jepa and pooling_type != "transformer":
            logging.warning(
                "--use_jepa requires pooling_type='transformer'; disabling JEPA."
            )

    if use_mffm and not descriptions_path:
        logging.warning("--use_mffm enabled without --descriptions_path")

    if path_to_model_weights != "":
        load = torch.load(path_to_model_weights)
        missing, unexpected = model.load_state_dict(load["state_dict"], strict=False)
        if missing:
            logging.info(f"New parameters (random init): {missing}")
        if unexpected:
            logging.info(f"Dropped parameters (not in model): {unexpected}")
        logging.info(f"Loaded weights from {path_to_model_weights}")

    # --- evaluation-only paths ---
    if only_evaluation in (0, 1, 2):
        ema = EMA(model, decay=ema_decay)
        if path_to_model_weights != "":
            load = torch.load(path_to_model_weights)
            if "ema" in load:
                ema.load_state_dict(load["ema"])

        if only_evaluation == 0:
            pred_file = evaluation(
                test_loader, model, ema=ema, set_name="test", use_tta=use_tta
            )
            print(
                "TEST:",
                evaluate(os.path.join(path, "Test", "annotations.json"), pred_file),
            )

        elif only_evaluation == 1:
            pred_file = evaluation(
                chall_loader, model, ema=ema, set_name="chall", use_tta=use_tta
            )
            print(
                "CHALL:",
                evaluate(os.path.join(path, "Chall", "annotations.json"), pred_file),
            )

        else:
            pred_file = evaluation(
                test_loader, model, ema=ema, set_name="test", use_tta=use_tta
            )
            print(
                "TEST:",
                evaluate(os.path.join(path, "Test", "annotations.json"), pred_file),
            )
            pred_file = evaluation(
                chall_loader, model, ema=ema, set_name="chall", use_tta=use_tta
            )
            print(
                "CHALL:",
                evaluate(os.path.join(path, "Chall", "annotations.json"), pred_file),
            )
        return 0

    # --- training setup ---
    # Freeze backbone params initially; keep VJEPA adapter trainable.
    is_vjepa = (not fusion_mode) and pre_model == "vjepa21_vitb"
    is_aim = (not fusion_mode) and pre_model == "aim_vitb16"
    vjepa_backbone = None
    if is_vjepa:
        for module in model.modules():
            if hasattr(module, "get_layerwise_lr_groups"):
                vjepa_backbone = module
                break

    if not is_aim:
        for name, param in model.named_parameters():
            if backbone_prefix in name and "jepa_adapter" not in name:
                param.requires_grad = False

    adapter_params = [
        p
        for n, p in model.named_parameters()
        if "jepa_adapter" in n and p.requires_grad
    ]
    head_params = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and "jepa_adapter" not in n
    ]
    backbone_params = [
        p
        for n, p in model.named_parameters()
        if backbone_prefix in n and "jepa_adapter" not in n
    ]
    adapter_lr = 1e-3 if adapter_params else None

    # Uncertainty weighter — 2 learnable scalars (log-variances) trained at head LR.
    uncertainty_weighter = None
    if uncertainty_weighting:
        uncertainty_weighter = UncertaintyWeighting(num_tasks=2).cuda()
        head_params = head_params + list(uncertainty_weighter.parameters())
        logging.info("Uncertainty weighting enabled (Kendall & Gal, 2018)")

    epoch_start = 0
    ema = EMA(model, decay=ema_decay)

    if args.continue_training and path_to_model_weights != "":
        load = torch.load(path_to_model_weights)
        missing, unexpected = model.load_state_dict(load["state_dict"], strict=False)
        if missing:
            logging.info(f"New parameters (random init): {missing}")
        if unexpected:
            logging.info(f"Dropped parameters (not in model): {unexpected}")
        epoch_start = load["epoch"]

        if epoch_start > freeze_epoch:
            for p in backbone_params:
                p.requires_grad = True
            optimizer_groups = [{"params": head_params, "lr": LR}]
            if adapter_params:
                optimizer_groups.append({"params": adapter_params, "lr": adapter_lr})
            if is_vjepa and vjepa_backbone is not None:
                optimizer_groups.extend(
                    vjepa_backbone.get_layerwise_lr_groups(base_lr=1e-5, decay=0.65)
                )
            else:
                optimizer_groups.append({"params": backbone_params, "lr": 1e-5})
            optimizer = torch.optim.AdamW(
                optimizer_groups,
                betas=(0.9, 0.999),
                eps=1e-7,
                weight_decay=weight_decay,
            )
            logging.info(
                f"Resuming from epoch {epoch_start} — backbone already unfrozen"
            )
        else:
            optimizer_groups = [{"params": head_params, "lr": LR}]
            if adapter_params:
                optimizer_groups.append({"params": adapter_params, "lr": adapter_lr})
            optimizer = torch.optim.AdamW(
                optimizer_groups,
                betas=(0.9, 0.999),
                eps=1e-7,
                weight_decay=weight_decay,
            )

        ema = EMA(model, decay=ema_decay)
        if "ema" in load:
            try:
                ema.load_state_dict(load["ema"])
            except Exception as e:
                logging.warning(
                    f"Could not restore EMA state ({e}). Reinitialising EMA from current weights."
                )

        if uncertainty_weighter is not None and "uw" in load and load["uw"] is not None:
            uncertainty_weighter.load_state_dict(load["uw"])

        try:
            optimizer.load_state_dict(load["optimizer"])
            for group in optimizer.param_groups:
                group["weight_decay"] = weight_decay
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_epochs, eta_min=1e-6
            )
            scheduler.load_state_dict(load["scheduler"])
            logging.info("Optimizer and scheduler state restored from checkpoint.")
        except (ValueError, KeyError) as e:
            logging.warning(
                f"Could not restore optimizer/scheduler state ({e}). "
                f"Starting optimizer fresh — fast-forwarding scheduler to epoch {epoch_start}."
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_epochs, eta_min=1e-6
            )
            for _ in range(epoch_start):
                scheduler.step()

    else:
        optimizer_groups = [{"params": head_params, "lr": LR}]
        if adapter_params:
            optimizer_groups.append({"params": adapter_params, "lr": adapter_lr})
        optimizer = torch.optim.AdamW(
            optimizer_groups,
            betas=(0.9, 0.999),
            eps=1e-7,
            weight_decay=weight_decay,
            amsgrad=False,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=1e-6
        )

    logging.info(
        f"Optimizer param groups: {[len(g['params']) for g in optimizer.param_groups]}"
    )

    # --- loss functions ---
    def _build_ordinal_pos_weight(sev_distribution):
        sev_distribution = sev_distribution.float().flatten()
        if sev_distribution.sum() <= 0:
            return None
        sev_distribution = sev_distribution / sev_distribution.sum()
        num_classes = sev_distribution.numel()
        weights = []
        for k in range(num_classes - 1):
            pos = sev_distribution[k + 1 :].sum()
            neg = sev_distribution[: k + 1].sum()
            if pos <= 0:
                weights.append((neg / pos).clamp(max=20.0))
            else:
                weights.append(neg / pos)
        return torch.stack(weights)

    combined_sev_dist = None
    combined_act_dist = None
    if train_all_but_test:
        train_sev_dist, train_act_dist = dataset_Train.getDistribution()
        valid_sev_dist, valid_act_dist = dataset_Valid.getDistribution()
        n_train = len(dataset_Train)
        n_valid = len(dataset_Valid)
        total = n_train + n_valid
        if total > 0:
            combined_sev_dist = (
                train_sev_dist * n_train + valid_sev_dist * n_valid
            ) / total
            combined_act_dist = (
                train_act_dist * n_train + valid_act_dist * n_valid
            ) / total

    if weighted_loss == "Yes":
        if combined_act_dist is not None:
            action_weights = 1.0 / torch.sqrt(combined_act_dist)
            action_weights = action_weights / action_weights.mean()
            weights_source = action_weights
        else:
            weights_source = dataset_Train.getWeights()[1]
        criterion_action = nn.CrossEntropyLoss(
            weight=weights_source.cuda(),
            label_smoothing=0.1,
        )
    else:
        criterion_action = nn.CrossEntropyLoss(label_smoothing=0.1)

    severity_dist = (
        combined_sev_dist
        if combined_sev_dist is not None
        else dataset_Train.getDistribution()[0]
    )
    ordinal_pos_weight = (
        None if args.no_pos_weight else _build_ordinal_pos_weight(severity_dist)
    )
    if ordinal_pos_weight is not None:
        ordinal_pos_weight = ordinal_pos_weight.cuda()

    criterion = {
        "action": criterion_action,
        "bce": nn.BCEWithLogitsLoss(),
        "ordinal_pos_weight": ordinal_pos_weight,
    }

    trainer(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        ema=ema,
        best_model_path=best_model_path,
        epoch_start=epoch_start,
        model_name=model_name,
        path_dataset=path,
        max_epochs=max_epochs,
        patience=args.patience,
        aux_weight=aux_weight,
        use_tta=use_tta,
        accum_steps=args.accum_steps,
        backbone_prefix=backbone_prefix,
        freeze_epoch=freeze_epoch,
        uncertainty_weighter=uncertainty_weighter,
        val_ann_path=(
            os.path.join(path, "Test", "annotations.json")
            if train_all_but_test
            else os.path.join(path, "Valid", "annotations.json")
        ),
        train_all_but_test=train_all_but_test,
        clip_weight=clip_weight,
        clip_embeddings_path=clip_embeddings_path,
        jepa_lambda_max=jepa_lambda_max,
        jepa_warmup_epochs=jepa_warmup_epochs,
        keep_top_k=args.keep_top_k,
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser(
        description="VARS — early fusion + multi-view modes",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--path", required=True, type=str)
    parser.add_argument("--start_frame", default=0, type=int)
    parser.add_argument("--end_frame", default=125, type=int)
    parser.add_argument("--fps", default=25, type=int)
    parser.add_argument("--num_views", default=5, type=int)
    parser.add_argument(
        "--train_sample_views",
        default=2,
        type=int,
        help="Number of views randomly sampled per training sample (view dropout regularisation)",
    )
    parser.add_argument("--data_aug", default="Yes", type=str)
    parser.add_argument(
        "--aug_preset",
        default="default",
        type=str,
        help="Augmentation preset: 'none' (no augmentation) | 'default' (light-medium augmentation) | 'strong' (strong augmentation with GaussianBlur)",
    )

    # Model
    parser.add_argument(
        "--fusion_mode",
        action="store_true",
        default=False,
        help="Early fusion: interleave all views along time, single backbone pass",
    )
    parser.add_argument(
        "--pre_model",
        default="mvit_v2_s",
        type=str,
        help="Backbone for multi-view mode (ignored when --fusion_mode)",
    )
    parser.add_argument(
        "--pooling_type",
        default="transformer",
        type=str,
        help="max | attention | transformer | crossattn | gat | bidir_crossattn | dynagat | cva | mamba (ignored when --fusion_mode)",
    )
    parser.add_argument(
        "--use_text_bridge",
        action="store_true",
        default=False,
        help="Enable per-sample CLIP text conditioning (multi-view transformer only)",
    )
    parser.add_argument(
        "--use_mffm",
        action="store_true",
        default=False,
        help="Enable MFFM fusion of visual features with text embeddings",
    )
    parser.add_argument(
        "--use_decoder",
        action="store_true",
        default=False,
        help="Enable task-specific query decoder between tokens and heads",
    )
    parser.add_argument(
        "--use_jepa",
        action="store_true",
        default=False,
        help="Enable JEPA self-supervised auxiliary objectives (requires --pooling_type transformer)",
    )
    parser.add_argument(
        "--jepa_lambda_max",
        default=0.1,
        type=float,
        help="Maximum JEPA loss weight, ramped up over --jepa_warmup_epochs",
    )
    parser.add_argument(
        "--jepa_warmup_epochs",
        default=2,
        type=int,
        help="Epochs to linearly ramp JEPA lambda from 0 to jepa_lambda_max",
    )
    parser.add_argument(
        "--descriptions_path",
        default=None,
        type=str,
        help="Path to precomputed text embeddings HDF5",
    )
    parser.add_argument(
        "--graph_topology",
        default="structured",
        type=str,
        help="structured | fully_connected | replay_only (only used when --pooling_type gat)",
    )

    # Training
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--LR", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-3, type=float)
    parser.add_argument("--max_epochs", default=40, type=int)
    parser.add_argument("--patience", default=8, type=int)
    parser.add_argument("--step_size", default=3, type=int)
    parser.add_argument("--gamma", default=0.1, type=float)
    parser.add_argument("--weighted_loss", default="Yes", type=str)
    parser.add_argument(
        "--balanced_sampler",
        default="Yes",
        type=str,
        help="Use class-balanced sampler for severity (Yes/No)",
    )

    # Advanced model options
    parser.add_argument(
        "--cascade_severity",
        action="store_true",
        default=False,
        help="Concat action logits with aggregator output before severity head",
    )
    parser.add_argument(
        "--uncertainty_weighting",
        action="store_true",
        default=False,
        help="Kendall & Gal uncertainty weighting for severity + action losses",
    )
    parser.add_argument(
        "--freeze_epoch",
        default=8,
        type=int,
        help="Epoch at which backbone is unfrozen for fine-tuning",
    )

    # New levers
    parser.add_argument(
        "--aux_weight",
        default=0.2,
        type=float,
        help="Weight for auxiliary BCE losses (contact, bodypart, etc.)",
    )
    parser.add_argument(
        "--clip_weight",
        default=0.0,
        type=float,
        help="Weight for CLIP severity alignment loss (0 disables)",
    )
    parser.add_argument(
        "--clip_embeddings_path",
        default="",
        type=str,
        help="Path to precomputed CLIP severity embeddings (.pt)",
    )
    parser.add_argument(
        "--ema_decay",
        default=0.999,
        type=float,
        help="EMA decay factor for shadow weights",
    )
    parser.add_argument(
        "--use_tta",
        action="store_true",
        default=True,
        help="Enable Test-Time Augmentation (horizontal flip)",
    )

    # Infra
    parser.add_argument("--model_name", default="VARS_early_fusion", type=str)
    parser.add_argument("--GPU", default=-1, type=int)
    parser.add_argument("--max_num_worker", default=4, type=int)
    parser.add_argument(
        "--only_evaluation",
        default=3,
        type=int,
        help="3=train | 0=test | 1=chall | 2=test+chall",
    )
    parser.add_argument("--path_to_model_weights", default="", type=str)
    parser.add_argument("--continue_training", action="store_true")
    parser.add_argument(
        "--accum_steps",
        default=2,
        type=int,
        help="Gradient accumulation steps (effective batch = batch_size * accum_steps)",
    )
    parser.add_argument(
        "--train_all_but_test",
        action="store_true",
        default=False,
        help="Train on Train+Valid combined, evaluate on Test only",
    )
    parser.add_argument(
        "--no_pos_weight",
        action="store_true",
        default=False,
        help="Disable ordinal pos_weight on severity loss",
    )

    # Path configuration
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="Dataset root directory (overrides --path)",
    )
    parser.add_argument(
        "--output_dir",
        default="models",
        type=str,
        help="Root directory for checkpoint output (default: models)",
    )
    parser.add_argument(
        "--backbone_ckpt_dir",
        default="",
        type=str,
        help="Directory containing pretrained backbone .pth files",
    )
    parser.add_argument(
        "--backbone_num_frames",
        default=None,
        type=int,
        help=(
            "Override the temporal frame count inside the backbone "
            "(e.g. 32 for TAdaFormer-L/14 32-frame). "
            "Temporal positional embeddings are interpolated automatically "
            "when the value differs from the checkpoint default (16). "
            "Use --fps to control how many frames the dataset delivers."
        ),
    )
    parser.add_argument(
        "--hf_cache_dir",
        default="",
        type=str,
        help="Override HF_HOME and TRANSFORMERS_CACHE to this directory",
    )
    parser.add_argument(
        "--hdf5_dir",
        default=None,
        type=str,
        help="Directory for HDF5 embedding files; defaults to --data_dir if not set",
    )
    parser.add_argument(
        "--keep_top_k",
        default=3,
        type=int,
        help="Keep this many most-recent per-epoch checkpoints (best_model is never pruned)",
    )

    args = parser.parse_args()
    checkArguments(args)

    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    start = time.time()
    logging.info("Starting VARS (early fusion + multi-view)")
    main(args)
    logging.info(f"Total time: {time.time() - start:.1f}s")
