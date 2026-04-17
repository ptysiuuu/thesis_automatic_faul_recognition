import os
import json
import torch
import logging
from tqdm import tqdm
from rules import rule_loss_with_stats # Skopiuj z VARS_model
from config.classes import INVERSE_EVENT_DICTIONARY


class VAR_Trainer:
    def __init__(self, model, optimizer, scheduler, criterions, device, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.crit_sev, self.crit_act = criterions
        self.device = device
        self.config = config

    def fit(self, train_loader, val_loader, test_loader, metrics_obj):
        best_lb = 0.0
        epochs_no_improve = 0
        patience = self.config.get('patience', 8)

        for epoch in range(1, self.config['epochs'] + 1):
            print(f"\n{'='*20} EPOKA {epoch}/{self.config['epochs']} {'='*20}")

            rule_stats = self.train_epoch(train_loader, epoch)
            print(f"Trening - Naruszenia reguł: R1: {rule_stats['R1']}, R2/3: {rule_stats['R2_3']}, R5: {rule_stats['R5']}")

            val_results = self.validate(val_loader, metrics_obj)
            val_lb = val_results['Leaderboard Value']
            print(f"Walidacja - Leaderboard Value: {val_lb:.4f} | Acc Action: {val_results['Balanced Accuracy (Action)']:.4f} | Acc Sev: {val_results['Balanced Accuracy (Severity)']:.4f}")

            if val_lb > best_lb:
                best_lb = val_lb
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), "best_var_model.pth")
                print(f"--> Zapisano nowy najlepszy model! (Wzrost z {best_lb:.4f} do {val_lb:.4f})")
            else:
                epochs_no_improve += 1
                print(f"Brak poprawy od {epochs_no_improve} epok (cierpliwość: {patience}).")

                if epochs_no_improve >= patience:
                    print(f"\n[!] Wczesne zatrzymanie (Early Stopping) aktywowane w epoce {epoch}.")
                    print(f"Najlepszy wynik Leaderboard: {best_lb:.4f}")
                    break

            test_results = self.validate(test_loader, metrics_obj)
            print(f"Test - Leaderboard Value: {test_results['Leaderboard Value']:.4f}")

            self.scheduler.step()

    def train_epoch(self, loader, epoch):
        self.model.train()
        rule_stats = {'R1': 0, 'R2_3': 0, 'R5': 0}
        pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")

        for tar_sev, tar_act, clips, _ in pbar:
            tar_sev, tar_act, clips = tar_sev.to(self.device), tar_act.to(self.device), clips.float().to(self.device)

            out_sev, out_act, _ = self.model(clips)
            loss = self.crit_sev(out_sev, tar_sev) + self.crit_act(out_act, tar_act)

            r_loss, batch_stats = rule_loss_with_stats(out_sev, out_act, weight=self.config['rule_weight'])
            for k in rule_stats: rule_stats[k] += batch_stats.get(k, 0)

            if self.config['rule_weight'] > 0:
                loss += r_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return rule_stats

    @torch.no_grad()
    def validate(self, loader, metrics_obj):
        self.model.eval()
        metrics_obj.reset()
        for tar_sev, tar_act, clips, _ in loader:
            tar_sev, tar_act, clips = tar_sev.to(self.device), tar_act.to(self.device), clips.float().to(self.device)
            out_sev, out_act, _ = self.model(clips)
            metrics_obj.update(out_sev, out_act, tar_sev, tar_act)
        return metrics_obj.compute()