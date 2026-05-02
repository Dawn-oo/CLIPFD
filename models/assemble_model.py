import torch
import torch.nn as nn

from .orign_CLIP_model.feature_extract import FeatureExtractor
from .branches.local_branch import LocalPatchBranch
from .heads.distinct_head import ClassifierHead
from .fusion.fusion import FeatureFusion
from models.branches.global_branch import GlobalAdapter


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
        final_num_classes: int = 3,     # 最终融合头：三分类
        aux_num_classes: int = 1,       # 全局辅助头：二分类单 logit
        grid_size=None,
        local_hidden_dim: int = 256,
        local_out_dim: int = 768,
        local_num_blocks: int = 2,
        proj_dropout: float = 0.1,
        block_dropout: float = 0.0,
        gn_groups: int = 8,
        fusion_dropout: float = 0.1,
        use_global_aux_head: bool = True,
        fusion_mode: str = "full",
        use_global_adapter: bool = True,
        global_adapter_dropout: float = 0.1,
    ):
        """
        :param backbone_name: 使用的主干模型名称
        :param freeze_backbone: 是否冻结主干参数，默认冻结
        :param device: 训练时使用的设备gpu或cpu
        :param final_num_classes: 最终融合头的类别
        :param aux_num_classes: 全局分支的判别
        :param grid_size:局部分支中patch的网格大小
        :param local_hidden_dim: 局部分支中隐藏层的维度
        :param local_out_dim: 局部分支最终输出特征的维度
        :param local_num_blocks: 局部分支残差块堆叠层数
        :param proj_dropout: 局部分支中投影层的Dropout概率
        :param block_dropout: 局部分支中每个残差块内部的Dropout概率
        :param gn_groups: 组归一化的分组数量
        :param fusion_dropout: 融合模块的dropout概率
        :param use_global_aux_head: 是否启用全局辅助头
        """
        super().__init__()

        self.fusion_mode = fusion_mode
        valid_fusion_modes = {"full", "global_only", "local_only"}
        if self.fusion_mode not in valid_fusion_modes:
            raise ValueError(
                f"fusion_mode must be one of {valid_fusion_modes}, but got {self.fusion_mode}"
            )

        # 1. CLIP 特征提取
        self.feature_extractor = FeatureExtractor(
            name=backbone_name,
            freeze=freeze_backbone,
            device=device,
        )
        self.preprocess = getattr(self.feature_extractor, "preprocess", None)

        # 补充 全局特征适配层
        self.use_global_adapter = use_global_adapter
        if self.use_global_adapter:
            self.global_adapter = GlobalAdapter(
                feat_dim=self.feature_extractor.global_dim,  # 768
                dropout=global_adapter_dropout,
            )
        else:
            self.global_adapter = nn.Identity()

        # 2. 全局辅助头
        self.use_global_aux_head = use_global_aux_head
        if self.use_global_aux_head:
            self.global_head = ClassifierHead(
                in_dim=self.feature_extractor.global_dim,   # 768
                num_classes=aux_num_classes,                # 1
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
            proj_dropout=proj_dropout,
            block_dropout=block_dropout,
            gn_groups=gn_groups,
        )

        # 4. 融合模块
        self.fusion_module = FeatureFusion(
            feat_dim=local_out_dim,                         # 768
            dropout=fusion_dropout,
        )

        # 5. 最终分类头
        self.final_head = ClassifierHead(
            in_dim=local_out_dim,
            num_classes=final_num_classes,                 # 3
        )

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
        return_features: bool = False,
    ) -> dict:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, but got {type(x)}")
        if x.dim() != 4:
            raise ValueError(f"x must have shape [B, C, H, W], but got {tuple(x.shape)}")

        # 1. 提特征
        feat_out = self.feature_extractor(x)
        global_feat_raw = feat_out["global_feat"].float()  # [B, 768]
        global_feat = self.global_adapter(global_feat_raw)  # [B, 768]
        patch_tokens = feat_out["patch_tokens"].float()  # [B, N, 1024]

        # 2. 局部分支
        local_out = self.local_branch(patch_tokens)
        local_feat = local_out["local_feat"]  # [B, 768]

        # 3. 根据消融模式选择三分类输入特征
        if self.fusion_mode == "full":
            final_feat = self.fusion_module(global_feat, local_feat)["fused_feat"]
        elif self.fusion_mode == "global_only":
            final_feat = global_feat
        elif self.fusion_mode == "local_only":
            final_feat = local_feat
        else:
            raise ValueError(f"Unsupported fusion_mode: {self.fusion_mode}")

        # 4. 最终分类
        logits = self.final_head(final_feat)["logits"]

        outputs = {
            "logits": logits
        }

        if return_aux and self.use_global_aux_head:
            outputs["global_logits"] = self.global_head(global_feat)["logits"]  # [B, 1]

        if return_features:
            outputs["global_feat_raw"] = global_feat_raw
            outputs["global_feat"] = global_feat
            outputs["local_feat"] = local_feat
            outputs["final_feat"] = final_feat
            if self.fusion_mode == "full":
                outputs["fused_feat"] = final_feat

        return outputs
