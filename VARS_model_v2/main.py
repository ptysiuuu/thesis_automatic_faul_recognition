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
from model import MVNetwork
from train import trainer, evaluation, EMA
from config.classes import EVENT_DICTIONARY, INVERSE_EVENT_DICTIONARY
from torchvision.models.video import (
    R3D_18_Weights, MC3_18_Weights,
    R2Plus1D_18_Weights, S3D_Weights,
    MViT_V2_S_Weights, MViT_V1_B_Weights,
)

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

def checkArguments(args):
    if not (1 <= args.num_views <= 5):
        raise ValueError("--num_views must be between 1 and 5")
    if args.data_aug not in ('Yes', 'No'):
        raise ValueError("--data_aug must be 'Yes' or 'No'")
    if args.pooling_type not in ('max', 'attention', 'transformer', 'crossattn'):
        raise ValueError("--pooling_type must be one of: max, attention, transformer, crossattn")
    if args.weighted_loss not in ('Yes', 'No'):
        raise ValueError("--weighted_loss must be 'Yes' or 'No'")
    if not (0 <= args.start_frame <= 124):
        raise ValueError("--start_frame must be 0-124")
    if not (1 <= args.end_frame <= 125):
        raise ValueError("--end_frame must be 1-125")
    if args.end_frame - args.start_frame < 2:
        raise ValueError("end_frame - start_frame must be >= 2")
    if not (1 <= args.fps <= 25):
        raise ValueError("--fps must be 1-25")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    LR           = args.LR
    gamma        = args.gamma
    step_size    = args.step_size
    start_frame  = args.start_frame
    end_frame    = args.end_frame
    weight_decay = args.weight_decay
    model_name   = args.model_name
    pre_model    = args.pre_model
    num_views    = args.num_views
    fps          = args.fps
    batch_size   = args.batch_size
    data_aug     = args.data_aug
    path         = args.path
    pooling_type = args.pooling_type
    weighted_loss = args.weighted_loss
    max_num_worker = args.max_num_worker
    max_epochs   = args.max_epochs
    only_evaluation = args.only_evaluation
    path_to_model_weights = args.path_to_model_weights
    aux_weight   = args.aux_weight
    ema_decay    = args.ema_decay
    use_tta      = args.use_tta
    balanced_sampler = args.balanced_sampler

    number_of_frames = int(
        (end_frame - start_frame) /
        (((end_frame - start_frame) / 25) * fps)
    )

    # --- logging ---
    os.makedirs(
        os.path.join("models", model_name, str(num_views), pre_model,
                     str(LR), f"_B{batch_size}_F{number_of_frames}_G{gamma}_Step{step_size}"),
        exist_ok=True
    )
    best_model_path = os.path.join(
        "models", model_name, str(num_views), pre_model,
        str(LR), f"_B{batch_size}_F{number_of_frames}_G{gamma}_Step{step_size}"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(best_model_path, "logging.log")),
            logging.StreamHandler(),
        ]
    )

    # --- augmentation ---
    if data_aug == 'Yes':
        transformAug = transforms.Compose([
            transforms.RandomAffine(degrees=(0, 0), translate=(0.1, 0.1), scale=(0.9, 1)),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.5, saturation=0.5, contrast=0.5),
            transforms.RandomHorizontalFlip(),
        ])
    else:
        transformAug = None

    # --- backbone-specific transforms ---
    _transform_map = {
        "r3d_18":     R3D_18_Weights.KINETICS400_V1.transforms(),
        "mc3_18":     MC3_18_Weights.KINETICS400_V1.transforms(),
        "r2plus1d_18": R2Plus1D_18_Weights.KINETICS400_V1.transforms(),
        "s3d":        S3D_Weights.KINETICS400_V1.transforms(),
        "mvit_v2_s":  MViT_V2_S_Weights.KINETICS400_V1.transforms(),
        "mvit_v1_b":  MViT_V1_B_Weights.KINETICS400_V1.transforms(),
    }
    transforms_model = _transform_map.get(pre_model, R2Plus1D_18_Weights.KINETICS400_V1.transforms())

    # --- datasets & loaders ---
    if only_evaluation == 0:
        dataset_Test = MultiViewDataset(path=path, start=start_frame, end=end_frame,
                                        fps=fps, split='Test', num_views=5,
                                        transform_model=transforms_model)
        test_loader = torch.utils.data.DataLoader(
            dataset_Test, batch_size=1, shuffle=False,
            num_workers=max_num_worker, pin_memory=True)

    elif only_evaluation == 1:
        dataset_Chall = MultiViewDataset(path=path, start=start_frame, end=end_frame,
                                         fps=fps, split='Chall', num_views=5,
                                         transform_model=transforms_model)
        chall_loader = torch.utils.data.DataLoader(
            dataset_Chall, batch_size=1, shuffle=False,
            num_workers=max_num_worker, pin_memory=True)

    elif only_evaluation == 2:
        dataset_Test  = MultiViewDataset(path=path, start=start_frame, end=end_frame,
                                          fps=fps, split='Test', num_views=5,
                                          transform_model=transforms_model)
        dataset_Chall = MultiViewDataset(path=path, start=start_frame, end=end_frame,
                                          fps=fps, split='Chall', num_views=5,
                                          transform_model=transforms_model)
        test_loader  = torch.utils.data.DataLoader(
            dataset_Test,  batch_size=1, shuffle=False,
            num_workers=max_num_worker, pin_memory=True)
        chall_loader = torch.utils.data.DataLoader(
            dataset_Chall, batch_size=1, shuffle=False,
            num_workers=max_num_worker, pin_memory=True)

    else:  # training mode (only_evaluation == 3)
        dataset_Train = MultiViewDataset(
            path=path, start=start_frame, end=end_frame, fps=fps,
            split='Train', num_views=num_views,
            transform=transformAug, transform_model=transforms_model)
        dataset_Valid = MultiViewDataset(
            path=path, start=start_frame, end=end_frame, fps=fps,
            split='Valid', num_views=num_views,
            transform_model=transforms_model)
        dataset_Test  = MultiViewDataset(
            path=path, start=start_frame, end=end_frame, fps=fps,
            split='Test', num_views=num_views,
            transform_model=transforms_model)

        if balanced_sampler == 'Yes':
            sampler = dataset_Train.get_balanced_sampler()
            train_loader = torch.utils.data.DataLoader(
                dataset_Train, batch_size=batch_size, sampler=sampler,
                num_workers=max_num_worker, pin_memory=True)
        else:
            train_loader = torch.utils.data.DataLoader(
                dataset_Train, batch_size=batch_size, shuffle=True,
                num_workers=max_num_worker, pin_memory=True)

        val_loader  = torch.utils.data.DataLoader(
            dataset_Valid, batch_size=1, shuffle=False,
            num_workers=max_num_worker, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(
            dataset_Test, batch_size=1, shuffle=False,
            num_workers=max_num_worker, pin_memory=True)

    # --- model ---
    model = MVNetwork(net_name=pre_model, agr_type=pooling_type).cuda()

    if path_to_model_weights != "":
        load = torch.load(path_to_model_weights)
        model.load_state_dict(load['state_dict'])
        logging.info(f"Loaded weights from {path_to_model_weights}")

    # --- evaluation-only paths ---
    if only_evaluation in (0, 1, 2):
        ema = EMA(model, decay=ema_decay)
        if path_to_model_weights != "":
            load = torch.load(path_to_model_weights)
            if 'ema' in load:
                ema.load_state_dict(load['ema'])

        if only_evaluation == 0:
            pred_file = evaluation(test_loader, model, ema=ema,
                                   set_name="test", use_tta=use_tta)
            print("TEST:", evaluate(os.path.join(path, "Test", "annotations.json"), pred_file))

        elif only_evaluation == 1:
            pred_file = evaluation(chall_loader, model, ema=ema,
                                   set_name="chall", use_tta=use_tta)
            print("CHALL:", evaluate(os.path.join(path, "Chall", "annotations.json"), pred_file))

        else:
            pred_file = evaluation(test_loader, model, ema=ema,
                                   set_name="test", use_tta=use_tta)
            print("TEST:", evaluate(os.path.join(path, "Test", "annotations.json"), pred_file))
            pred_file = evaluation(chall_loader, model, ema=ema,
                                   set_name="chall", use_tta=use_tta)
            print("CHALL:", evaluate(os.path.join(path, "Chall", "annotations.json"), pred_file))
        return 0

    # --- training setup ---
    # Freeze backbone initially; only aggregation + classification heads train
    for name, param in model.named_parameters():
        if "aggregation_model.model." not in name and "fc_" not in name and "inter" not in name:
            param.requires_grad = False
    logging.info("Backbone frozen for first 5 epochs")

    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{'params': head_params, 'lr': LR}],
        betas=(0.9, 0.999), eps=1e-7,
        weight_decay=weight_decay, amsgrad=False,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=1e-6
    )

    epoch_start = 0
    ema = EMA(model, decay=ema_decay)

    if args.continue_training and path_to_model_weights != "":
        load = torch.load(path_to_model_weights)
        model.load_state_dict(load['state_dict'])
        optimizer.load_state_dict(load['optimizer'])
        scheduler.load_state_dict(load['scheduler'])
        epoch_start = load['epoch']
        if 'ema' in load:
            ema.load_state_dict(load['ema'])
        logging.info(f"Resuming training from epoch {epoch_start}")

    # --- loss functions ---
    if weighted_loss == 'Yes':
        criterion_action = nn.CrossEntropyLoss(
            weight=dataset_Train.getWeights()[1].cuda(),
            label_smoothing=0.1,
        )
    else:
        criterion_action = nn.CrossEntropyLoss(label_smoothing=0.1)

    criterion = {
        'action': criterion_action,
        'bce':    nn.BCEWithLogitsLoss(),
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
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = ArgumentParser(description='VARS v2 — multi-task + ordinal',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    # Data
    parser.add_argument('--path',            required=True,  type=str)
    parser.add_argument('--start_frame',     default=0,      type=int)
    parser.add_argument('--end_frame',       default=125,    type=int)
    parser.add_argument('--fps',             default=25,     type=int)
    parser.add_argument('--num_views',       default=5,      type=int)
    parser.add_argument('--data_aug',        default='Yes',  type=str)

    # Model
    parser.add_argument('--pre_model',       default='mvit_v2_s', type=str,
                        help='r3d_18 | mc3_18 | r2plus1d_18 | s3d | mvit_v2_s | mvit_v1_b')
    parser.add_argument('--pooling_type',    default='transformer', type=str,
                        help='max | attention | transformer | crossattn')

    # Training
    parser.add_argument('--batch_size',      default=4,      type=int)
    parser.add_argument('--LR',              default=1e-4,   type=float)
    parser.add_argument('--weight_decay',    default=1e-3,   type=float)
    parser.add_argument('--max_epochs',      default=40,     type=int)
    parser.add_argument('--patience',        default=8,      type=int)
    parser.add_argument('--step_size',       default=3,      type=int)
    parser.add_argument('--gamma',           default=0.1,    type=float)
    parser.add_argument('--weighted_loss',   default='Yes',  type=str)
    parser.add_argument('--balanced_sampler',default='Yes',  type=str,
                        help='Use class-balanced sampler for severity (Yes/No)')

    # New levers
    parser.add_argument('--aux_weight',      default=0.2,    type=float,
                        help='Weight for auxiliary BCE losses (contact, bodypart, etc.)')
    parser.add_argument('--ema_decay',       default=0.999,  type=float,
                        help='EMA decay factor for shadow weights')
    parser.add_argument('--use_tta',         action='store_true', default=True,
                        help='Enable Test-Time Augmentation (horizontal flip)')

    # Infra
    parser.add_argument('--model_name',      default='VARS_v2', type=str)
    parser.add_argument('--GPU',             default=-1,     type=int)
    parser.add_argument('--max_num_worker',  default=4,      type=int)
    parser.add_argument('--only_evaluation', default=3,      type=int,
                        help='3=train | 0=test | 1=chall | 2=test+chall')
    parser.add_argument('--path_to_model_weights', default='', type=str)
    parser.add_argument('--continue_training', action='store_true')

    args = parser.parse_args()
    checkArguments(args)

    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    start = time.time()
    logging.info('Starting VARS v2')
    main(args)
    logging.info(f'Total time: {time.time() - start:.1f}s')
