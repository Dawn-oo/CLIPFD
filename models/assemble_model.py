# 该项目文件完成对主体、分支、头部架构的组合，向外提供统一的完整的分类模型接口
import torch
import torch.nn as nn

from .orign_CLIP_model.feature_extract import FeatureExtractor
from .branches.local_branch import LocalPatchBranch
from .heads.distinct_head import ClassifierHead
from .fusion.fusion import FeatureFusion


class CLIPFDModel(nn.Module):
    """
    纯模型组装模块：
    只负责把特征提取、局部分支、融合、分类头串起来
    不负责损失函数、优化器、训练流程
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
        use_global_aux_head: bool = True,
    ):
        super().__init__()

        # 1. CLIP 特征提取
        self.feature_extractor = FeatureExtractor(
            name=backbone_name,
            freeze=freeze_backbone,
            device=device,
        )
        self.preprocess = getattr(self.feature_extractor, "preprocess", None)

        # 2. 全局辅助头（可选）
        self.use_global_aux_head = use_global_aux_head
        if self.use_global_aux_head:
            self.global_head = ClassifierHead(
                in_dim=self.feature_extractor.global_dim,   # 768
                num_classes=num_classes,
            )
        else:
            self.global_head = None

        # 3. 局部分支
        self.local_branch = LocalPatchBranch(
            in_dim=self.feature_extractor.local_dim,        # 1024
            hidden_dim=local_hidden_dim,
            out_dim=local_out_dim,                          # 768
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
            feat_dim=local_out_dim,                         # 768
            dropout=fusion_dropout,
        )

        # 5. 最终分类头
        self.final_head = ClassifierHead(
            in_dim=local_out_dim,
            num_classes=num_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
        return_features: bool = False,
        return_heatmap: bool = False,
    ) -> dict:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, but got {type(x)}")
        if x.dim() != 4:
            raise ValueError(f"x must have shape [B, C, H, W], but got {tuple(x.shape)}")

        # 1. 提特征
        feat_out = self.feature_extractor(x)
        global_feat = feat_out["global_feat"]          # [B, 768]
        patch_tokens = feat_out["patch_tokens"]        # [B, N, 1024]

        # 2. 局部分支
        local_out = self.local_branch(patch_tokens)
        local_feat = local_out["local_feat"]           # [B, 768]
        local_heatmap = local_out["local_heatmap"]     # [B, 1, H, W]

        # 3. 融合
        fused_feat = self.fusion_module(global_feat, local_feat)["fused_feat"]

        # 4. 最终分类
        logits = self.final_head(fused_feat)["logits"]

        outputs = {
            "logits": logits
        }

        # 训练时可选返回辅助头
        if return_aux and self.use_global_aux_head:
            outputs["global_logits"] = self.global_head(global_feat)["logits"]
        #
        # # 可视化时可选返回热图
        # if return_heatmap:
        #     outputs["local_heatmap"] = local_heatmap

        # 调试/分析时可选返回中间特征
        if return_features:
            outputs["global_feat"] = global_feat
            outputs["local_feat"] = local_feat
            outputs["fused_feat"] = fused_feat

        return outputs
