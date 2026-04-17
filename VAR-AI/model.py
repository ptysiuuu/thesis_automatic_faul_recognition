import torch
import torch.nn as nn
from transformers import VideoMAEModel
import torch.nn.functional as F


class CrossAttentionAggregator(nn.Module):
    def __init__(self, feat_dim=768, num_heads=4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.norm = nn.LayerNorm(feat_dim)
        # View embeddings — adaptacja do różnych kątów kamer
        self.view_embeds = nn.Parameter(torch.zeros(1, 5, feat_dim))
        nn.init.trunc_normal_(self.view_embeds, std=0.02)

    def forward(self, features, view_mask):
        # features: [B, V, feat_dim]
        B, V, D = features.shape

        # Dodanie view embeddings
        features = features + self.view_embeds[:, :V, :]

        main_cam = features[:, 0:1, :]   # [B, 1, D] — Query (Główna kamera)
        replays  = features[:, 1:,  :]   # [B, V-1, D] — Key/Value (Powtórki)

        replay_mask = view_mask[:, 1:]   # Maska brakujących powtórek [B, V-1]

        # Sprawdzenie, czy wszystkie powtórki dla danego batcha są puste
        all_missing = replay_mask.all(dim=1)  # [B]

        attn_out, _ = self.cross_attn(
            query=main_cam,
            key=replays,
            value=replays,
            key_padding_mask=replay_mask
        )

        output = self.norm(main_cam + attn_out).squeeze(1)  # [B, D]

        # Fallback: Dla sampli bez żadnych powtórek — użyj samej main cam
        main_only = self.norm(main_cam.squeeze(1))
        output = torch.where(
            all_missing.unsqueeze(1).expand_as(output),
            main_only,
            output
        )

        return output

class MVNetworkV2(nn.Module):
    def __init__(self, feat_dim=768):
        super().__init__()
        self.feat_dim = feat_dim

        # Backbone — całkowicie zamrożony
        self.backbone = VideoMAEModel.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics"
        )
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Agregacja cross-attention
        self.aggregator = CrossAttentionAggregator(feat_dim=feat_dim, num_heads=4)

        # Głowy klasyfikacyjne z dropoutem zapobiegającym overfittingowi
        self.fc_offence = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(p=0.3),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(feat_dim // 2, 4)
        )

        self.fc_action = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(p=0.3),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(feat_dim // 2, 8)
        )

    def extract_features(self, video):
        B, T, C, H, W = video.shape

        if T != 16:
            video_for_interp = video.permute(0, 2, 1, 3, 4)

            video_interp = torch.nn.functional.interpolate(
                video_for_interp, size=(16, H, W), mode='trilinear', align_corners=False
            )

            pixel_values = video_interp.permute(0, 2, 1, 3, 4)
        else:
            pixel_values = video

        with torch.no_grad():
            outputs = self.backbone(pixel_values=pixel_values)

        features = outputs.last_hidden_state.mean(dim=1)
        return features

    def forward(self, mvimages):
        # mvimages: [B, V, C, T, H, W]
        B, V, C, T, H, W = mvimages.shape

        all_features = []
        for v in range(V):
            feat = self.extract_features(mvimages[:, v])  # [B, feat_dim]
            all_features.append(feat)

        features = torch.stack(all_features, dim=1)  # [B, V, feat_dim]

        # Obliczanie maski: widok jest pusty, jeśli suma absolutna to 0
        view_mask = (mvimages.abs().sum(dim=(2, 3, 4, 5)) == 0)  # [B, V]

        # Cross-attention agregacja
        aggregated = self.aggregator(features, view_mask)  # [B, feat_dim]

        # Klasyfikacja
        pred_offence = self.fc_offence(aggregated)
        pred_action  = self.fc_action(aggregated)

        return pred_offence, pred_action, None