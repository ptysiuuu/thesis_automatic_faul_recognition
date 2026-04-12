from utils import batch_tensor, unbatch_tensor
import torch
from torch import nn


class TransformerAggregate(nn.Module):
    def __init__(self, model, feat_dim, num_layers=1, num_heads=4, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net
        self.feat_dim = feat_dim

        # 1. Token [CLS] - zbiera informacje ze wszystkich widoków
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feat_dim))

        # 2. View Embeddings - model uczy się charakterystyki każdego ujęcia (max 5 widoków)
        self.view_embeds = nn.Parameter(torch.zeros(1, 5, feat_dim))

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=feat_dim, nhead=num_heads, batch_first=True, dropout=0.1, dim_feedforward=feat_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.view_embeds, std=0.02)

    def forward(self, mvimages):
        B, V, C, D, H, W = mvimages.shape
        # Ekstrakcja cech z każdego widoku przez backbone (np. MViT)
        aux = self.lifting_net(unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True))

        # Dodanie embeddingów widoku (View Embeddings)
        # Przycinamy embeddingi do aktualnej liczby widoków V w batchu
        aux = aux + self.view_embeds[:, :V, :]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, aux), dim=1) # [B, V+1, feat_dim]

        # Tworzenie maski dla brakujących widoków (True oznacza "ignoruj")
        # Wykrywamy wyzerowane tensory z dataset.py
        view_mask = (mvimages.abs().sum(dim=(2, 3, 4, 5)) == 0)

        # Token [CLS] na pozycji 0 nigdy nie jest maskowany (False)
        cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=mvimages.device)
        padding_mask = torch.cat((cls_mask, view_mask), dim=1) # [B, V+1]

        # Przetworzenie przez Transformer z użyciem maski
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # Zwracamy tylko stan tokena [CLS] (indeks 0) jako reprezentację całej akcji
        output = x[:, 0]

        return output, None # None zamiast wag uwagi dla kompatybilności z resztą kodu


class CrossAttentionAggregate(nn.Module):
    def __init__(self, model, feat_dim, num_heads=4, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net

        # Cross-attention: Q=main cam, K/V=replays
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, mvimages):
        B, V, C, D, H, W = mvimages.shape

        # Ekstrakcja cech przez backbone
        aux = self.lifting_net(
            unbatch_tensor(
                self.model(batch_tensor(mvimages, dim=1, squeeze=True)),
                B, dim=1, unsqueeze=True
            )
        )  # [B, V, feat_dim]

        # Rozdziel main camera i replaye
        main_cam = aux[:, 0:1, :]      # [B, 1, feat_dim] — clip_0 jako Query
        replays  = aux[:, 1:,  :]      # [B, V-1, feat_dim] — clip_1+ jako K/V

        # Maska brakujących replayów (wyzerowane tensory)
        replay_mask = (mvimages[:, 1:, :].abs().sum(dim=(2, 3, 4, 5)) == 0)  # [B, V-1]

        # Jeśli wszystkie replaye brakują — fallback do main cam
        if replay_mask.all():
            return self.norm(main_cam.squeeze(1)), None

        # Cross-attention: main cam patrzy na replaye
        attn_out, attn_weights = self.cross_attn(
            query=main_cam,           # [B, 1, feat_dim]
            key=replays,              # [B, V-1, feat_dim]
            value=replays,
            key_padding_mask=replay_mask  # ignoruj brakujące replaye
        )  # attn_out: [B, 1, feat_dim]

        # Residual connection: main cam + wyciągnięte detale z replayów
        output = self.norm(main_cam + attn_out)  # [B, 1, feat_dim]

        return output.squeeze(1), attn_weights  # [B, feat_dim]


class ViewMaxAggregate(nn.Module):
    def __init__(self, model, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net
    def forward(self, mvimages):
        B, V, C, D, H, W = mvimages.shape
        aux = self.lifting_net(unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True))
        pooled_view = torch.max(aux, dim=1)[0]
        return pooled_view.squeeze(), aux

class MVAggregate(nn.Module):
    def __init__(self, model, agr_type="max", feat_dim=400, lifting_net=nn.Sequential()):
        super().__init__()
        self.agr_type = agr_type

        self.fc_offence = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(p=0.3),
            nn.Linear(feat_dim, feat_dim),
            nn.Dropout(p=0.3),
            nn.Linear(feat_dim, 4)
        )

        self.fc_action = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(p=0.3),
            nn.Linear(feat_dim, feat_dim),
            nn.Dropout(p=0.3),
            nn.Linear(feat_dim, 8)
        )

        self.inter = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, feat_dim),
        )

        if self.agr_type == "max":
            self.aggregation_model = ViewMaxAggregate(model=model, lifting_net=lifting_net)
        elif self.agr_type == "transformer":
            self.aggregation_model = TransformerAggregate(model=model, feat_dim=feat_dim, lifting_net=lifting_net)
        elif self.agr_type == "crossattn":
            self.aggregation_model = CrossAttentionAggregate(
                model=model,
                feat_dim=feat_dim,
                num_heads=4,
                lifting_net=lifting_net
            )
        else:
            from mvaggregate import WeightedAggregate
            self.aggregation_model = WeightedAggregate(model=model, feat_dim=feat_dim, lifting_net=lifting_net)

    def forward(self, mvimages):
        pooled_view, attention = self.aggregation_model(mvimages)
        inter = self.inter(pooled_view)
        pred_action = self.fc_action(inter)
        pred_offence_severity = self.fc_offence(inter)
        return pred_offence_severity, pred_action, attention
