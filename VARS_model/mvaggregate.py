from utils import batch_tensor, unbatch_tensor
import torch
from torch import nn

class TransformerAggregate(nn.Module):
    def __init__(self, model, feat_dim, num_layers=2, num_heads=8, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net
        self.feat_dim = feat_dim

        # 1. Token [CLS] - zbiera informacje ze wszystkich widoków
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feat_dim))

        # 2. View Embeddings - model uczy się charakterystyki każdego ujęcia (max 5 widoków)
        self.view_embeds = nn.Parameter(torch.zeros(1, 5, feat_dim))

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=feat_dim, nhead=num_heads, batch_first=True)
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

        self.inter = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, feat_dim),
        )

        self.fc_offence = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, 4)
        )

        self.fc_action = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, 8)
        )

        if self.agr_type == "max":
            self.aggregation_model = ViewMaxAggregate(model=model, lifting_net=lifting_net)
        elif self.agr_type == "transformer":
            self.aggregation_model = TransformerAggregate(model=model, feat_dim=feat_dim, lifting_net=lifting_net)
        else:
            # Domyślnie używamy Twojego poprzedniego WeightedAggregate jeśli chcesz
            from mvaggregate import WeightedAggregate
            self.aggregation_model = WeightedAggregate(model=model, feat_dim=feat_dim, lifting_net=lifting_net)

    def forward(self, mvimages):
        pooled_view, attention = self.aggregation_model(mvimages)
        inter = self.inter(pooled_view)
        pred_action = self.fc_action(inter)
        pred_offence_severity = self.fc_offence(inter)
        return pred_offence_severity, pred_action, attention
