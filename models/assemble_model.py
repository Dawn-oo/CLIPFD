
# # 该项目文件完成对主体、分支、头部架构的组合，向外提供统一的完整的分类模型接口

import torch
import torch.nn as nn

# 按你本地实际路径修改
from .orign_CLIP_model.feature_extract import FeatureExtractor
from .fusion.fusion import FeatureFusion

from .branches.local_branch import LocalPatchBranch
from .heads.distinct_head import ClassifierHead


class CLIPFDModel(nn.Module):
    """
    CLIPFD 总模型组装模块

    输入:
        x: [B, 3, H, W]

    输出:
        {
            "global_feat":   [B, 768],
            "patch_tokens":  [B, N, 1024],
            "global_logits": [B, num_classes],
            "local_feat":    [B, 768],
            "local_heatmap": [B, 1, H, W],
            "fused_feat":    [B, 768],
            "logits":        [B, num_classes],
            "cls_token":     [B, 1024],        # 如果特征提取器返回
            "grid_size":     (gh, gw),         # 如果特征提取器返回
        }
    """

    def __init__(
        self,
        backbone_name: str = "ViT-L/14",
        freeze_backbone: bool = True,
        device: str = "cpu",
        num_classes: int = 1,
        input_size=(224, 224),
        grid_size=None,
        local_hidden_dim: int = 256,
        local_out_dim: int = 768,
        local_num_blocks: int = 2,
        proj_dropout: float = 0.1,
        block_dropout: float = 0.0,
        gn_groups: int = 8,
        eps: float = 1e-6,
        fusion_dropout: float = 0.1,
    ):
        super().__init__()

        # 1. CLIP 特征提取模块
        self.feature_extractor = FeatureExtractor(
            name=backbone_name,
            freeze=freeze_backbone,
            device=device,
        )

        # 方便外部直接访问 CLIP preprocess
        self.preprocess = getattr(self.feature_extractor, "preprocess", None)

        # 2. 全局辅助分类头
        self.global_head = ClassifierHead(
            in_dim=self.feature_extractor.global_dim,   # 通常 768
            num_classes=num_classes,
        )

        # 3. 局部分支
        self.local_branch = LocalPatchBranch(
            in_dim=self.feature_extractor.local_dim,    # 通常 1024
            hidden_dim=local_hidden_dim,
            out_dim=local_out_dim,                      # 你当前设计是 768
            num_blocks=local_num_blocks,
            grid_size=grid_size,
            input_size=input_size,
            proj_dropout=proj_dropout,
            block_dropout=block_dropout,
            gn_groups=gn_groups,
            eps=eps,
        )

        # 4. 融合模块
        self.fusion_module = FeatureFusion(
            feat_dim=local_out_dim,                     # 当前固定成 768
            dropout=fusion_dropout,
        )

        # 5. 最终分类头
        self.final_head = ClassifierHead(
            in_dim=local_out_dim,
            num_classes=num_classes,
        )

    def freeze_backbone(self):
        if hasattr(self.feature_extractor, "freeze"):
            self.feature_extractor.freeze()

    def unfreeze_backbone(self):
        if hasattr(self.feature_extractor, "unfreeze"):
            self.feature_extractor.unfreeze()

    def forward(self, x: torch.Tensor) -> dict:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, but got {type(x)}")

        if x.dim() != 4:
            raise ValueError(f"x must have shape [B, C, H, W], but got {tuple(x.shape)}")

        # 1. 提取 CLIP 全局特征和 patch token
        feat_out = self.feature_extractor(x)

        if not isinstance(feat_out, dict):
            raise TypeError(f"feature_extractor must return dict, but got {type(feat_out)}")

        if "global_feat" not in feat_out:
            raise KeyError("feature_extractor output must contain 'global_feat'")
        if "patch_tokens" not in feat_out:
            raise KeyError("feature_extractor output must contain 'patch_tokens'")

        global_feat = feat_out["global_feat"]        # [B, 768]
        patch_tokens = feat_out["patch_tokens"]      # [B, N, 1024]

        # 兼容你之前的两种 key 写法
        cls_token = feat_out.get("cls_token", None)
        grid_info = feat_out.get("grid_size", feat_out.get("(gh, gw)", None))

        # 2. 全局辅助头
        global_out = self.global_head(global_feat)
        if not isinstance(global_out, dict) or "logits" not in global_out:
            raise KeyError("global_head output must be dict and contain 'logits'")
        global_logits = global_out["logits"]

        # 3. 局部分支
        local_out = self.local_branch(patch_tokens)
        if not isinstance(local_out, dict):
            raise TypeError(f"local_branch must return dict, but got {type(local_out)}")
        if "local_feat" not in local_out:
            raise KeyError("local_branch output must contain 'local_feat'")
        if "local_heatmap" not in local_out:
            raise KeyError("local_branch output must contain 'local_heatmap'")

        local_feat = local_out["local_feat"]              # [B, 768]
        local_heatmap = local_out["local_heatmap"]        # [B, 1, H, W]

        # 4. 融合
        fusion_out = self.fusion_module(global_feat, local_feat)
        if not isinstance(fusion_out, dict) or "fused_feat" not in fusion_out:
            raise KeyError("fusion_module output must be dict and contain 'fused_feat'")
        fused_feat = fusion_out["fused_feat"]             # [B, 768]

        # 5. 最终分类
        final_out = self.final_head(fused_feat)
        if not isinstance(final_out, dict) or "logits" not in final_out:
            raise KeyError("final_head output must be dict and contain 'logits'")
        logits = final_out["logits"]

        outputs = {
            "global_feat": global_feat,
            "patch_tokens": patch_tokens,
            "global_logits": global_logits,
            "local_feat": local_feat,
            "local_heatmap": local_heatmap,
            "fused_feat": fused_feat,
            "logits": logits,
        }

        if cls_token is not None:
            outputs["cls_token"] = cls_token

        if grid_info is not None:
            outputs["grid_size"] = grid_info

        return outputs
