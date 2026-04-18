import torch
from mvaggregate import MVAggregate
from torchvision.models.video import (
    r3d_18, R3D_18_Weights,
    mc3_18, MC3_18_Weights,
    r2plus1d_18, R2Plus1D_18_Weights,
    s3d, S3D_Weights,
    mvit_v2_s, MViT_V2_S_Weights,
    mvit_v1_b, MViT_V1_B_Weights,
)


class MVNetwork(torch.nn.Module):
    """
    Backbone + multi-view aggregation wrapper.

    Forward output:
        (ordinal_severity_logits [B,3],
         action_logits [B,8],
         contact_logit [B],
         bodypart_logit [B],
         try_to_play_logit [B],
         handball_logit [B],
         attention)
    """

    def __init__(self, net_name='mvit_v2_s', agr_type='transformer',
                 lifting_net=torch.nn.Sequential()):
        super().__init__()
        self.net_name = net_name
        self.agr_type = agr_type
        self.feat_dim = 512  # default; overridden below for some models

        if net_name == "r3d_18":
            network = r3d_18(weights=R3D_18_Weights.DEFAULT)
        elif net_name == "s3d":
            network = s3d(weights=S3D_Weights.DEFAULT)
            self.feat_dim = 400
        elif net_name == "mc3_18":
            network = mc3_18(weights=MC3_18_Weights.DEFAULT)
        elif net_name == "r2plus1d_18":
            network = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
        elif net_name == "mvit_v2_s":
            network = mvit_v2_s(weights=MViT_V2_S_Weights.DEFAULT)
            self.feat_dim = 400
        elif net_name == "mvit_v1_b":
            network = mvit_v1_b(weights=MViT_V1_B_Weights.DEFAULT)
            self.feat_dim = 768
        else:
            print(f"Warning: unknown model '{net_name}', falling back to r2plus1d_18")
            network = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)

        # Remove the classification FC so the backbone becomes a feature extractor
        network.fc = torch.nn.Sequential()

        self.mvnetwork = MVAggregate(
            model=network,
            agr_type=self.agr_type,
            feat_dim=self.feat_dim,
            lifting_net=lifting_net,
        )

    def forward(self, mvimages):
        return self.mvnetwork(mvimages)
