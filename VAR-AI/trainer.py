import os
import json
import torch
import logging
from tqdm import tqdm
from SoccerNet.Evaluation.MV_FoulRecognition import evaluate
from rules import rule_loss_with_stats
from metrics import VARMetrics


class VAR_Trainer:
    def __init__(self, model, optimizer, scheduler, criterions, device, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.crit_sev, self.crit_act = criterions
        self.device = device
        self.config = config
        self.dataset_path = config['dataset_path']
        self.model_name = config.get('model_name', 'VAR-AI')

    def fit(self, train_loader, val_loader, test_loader):
        best_lb = 0.0
        epochs_no_improve = 0
        patience = self.config.get('patience', 8)

        for epoch in range(1, self.config['epochs'] + 1):
            print(f"\n{'='*20} EPOKA {epoch}/{self.config['epochs']} {'='*20}")

            rule_stats = self.train_epoch(train_loader, epoch)
            print(f"Naruszenia reguł: R1={rule_stats['R1']}, R2/3={rule_stats['R2_3']}, R5={rule_stats['R5']}")

            # Walidacja
            val_pred_file = self.evaluate_split(val_loader, "valid", epoch)
            val_results = evaluate(
                os.path.join(self.dataset_path, "Valid", "annotations.json"),
                val_pred_file
            )
            val_lb = val_results['leaderboard_value']
            print(f"VALIDATION: {val_results}")

            if val_lb > best_lb:
                prev_best = best_lb
                best_lb = val_lb
                epochs_no_improve = 0
                torch.save(self.model.state_dict(),
                           os.path.join(self.model_name, "best_model.pth"))
                print(f"--> Nowy najlepszy model! ({prev_best:.4f} → {best_lb:.4f})")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping w epoce {epoch}. Najlepszy LB: {best_lb:.4f}")
                    break

            # Test
            test_pred_file = self.evaluate_split(test_loader, "test", epoch)
            test_results = evaluate(
                os.path.join(self.dataset_path, "Test", "annotations.json"),
                test_pred_file
            )
            print(f"TEST: {test_results}")

            self.scheduler.step()

    def train_epoch(self, loader, epoch):
        self.model.train()
        rule_stats = {'R1': 0, 'R2_3': 0, 'R5': 0}
        pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")

        for tar_sev, tar_act, clips, _ in pbar:
            tar_sev = tar_sev.to(self.device)
            tar_act = tar_act.to(self.device)
            clips   = clips.float().to(self.device)

            out_sev, out_act, _ = self.model(clips)

            if out_sev.dim() == 1:
                out_sev = out_sev.unsqueeze(0)
            if out_act.dim() == 1:
                out_act = out_act.unsqueeze(0)

            loss = self.crit_sev(out_sev, tar_sev) + self.crit_act(out_act, tar_act)

            r_loss, batch_stats = rule_loss_with_stats(
                out_sev, out_act, weight=self.config['rule_weight']
            )
            rule_stats['R1']   += batch_stats.get('R1_Dive_Violation', 0)
            rule_stats['R2_3'] += batch_stats.get('R2_3_Violent_LowSev', 0)
            rule_stats['R5']   += batch_stats.get('R5_RedCard_MildAction', 0)

            if self.config['rule_weight'] > 0:
                loss = loss + r_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return rule_stats

    @torch.no_grad()
    def evaluate_split(self, loader, set_name, epoch):
        self.model.eval()
        metrics = VARMetrics()

        for tar_sev, tar_act, clips, action_ids in loader:
            clips = clips.float().to(self.device)
            out_sev, out_act, _ = self.model(clips)

            if out_sev.dim() == 1:
                out_sev = out_sev.unsqueeze(0)
            if out_act.dim() == 1:
                out_act = out_act.unsqueeze(0)

            metrics.update(out_sev, out_act, action_ids)

        return metrics.save(set_name, self.model_name)