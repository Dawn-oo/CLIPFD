from models.orign_CLIP_model import clip
import torch.nn as nn
import torch

# 主要作用为提取局部patch向量和局部的网格对应以及全局表征
class FeatureExtractor(nn.Module):
    def __init__(self, name="ViT-L/14", freeze=True, device="cpu"):
        super().__init__()

        self.backbone, self.preprocess = clip.load(name, device=device)
        self.backbone_name = name
        self.global_dim = 768
        self.local_dim = 1024

        if freeze:
            self.freeze()

    def extract_global_feat(self, x):
        return self.backbone.encode_image(x)

    def freeze(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def extract_feat_and_tokens(self, x):
        return self.backbone.encode_image_with_tokens(x)

    # 全局分支的前向传播
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, but got {type(x)}")

        if x.dim() != 4:
            raise ValueError(f"x must have shape [B, C, H, W], but got {tuple(x.shape)}")

        global_feat, cls_token, patch_tokens ,(gh, gw)= self.extract_feat_and_tokens(x)

        return {
            "global_feat": global_feat,  # [B, 768]
            "cls_token": cls_token,  # [B, 1024]
            "patch_tokens": patch_tokens, # [B, N, 1024] N为 patch token 的数量
            "grid_size":(gh, gw)
        }



