import math
import torch
import torch.nn as nn
import torch.nn.functional as F



# 将二维卷积、组归一化和 GELU 激活函数按顺序封装成一个模块
class ConvGNAct(nn.Sequential):
    """
    Conv -> GroupNorm -> GELU
    """
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, groups=1, gn_groups=8, act=True):
        """
        :param in_ch:输入特征图的通道数
        :param out_ch:输出特征的通道数
        :param k:卷积核大小，默认3*3
        :param s:卷积步长大小
        :param p:填充的大小
        :param groups:卷积的分组数
        :param gn_groups:组归一化期望的分组数量
        :param act:是否添加GELU激活函数
        """
        gn_groups = self._safe_groups(out_ch, gn_groups)
        layers = [
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=k,
                stride=s,
                padding=p,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(gn_groups, out_ch),
        ]
        if act:
            layers.append(nn.GELU())
        super().__init__(*layers)

    # 确定组归一化的分组，如果无法被整除，则进行自动调整
    @staticmethod
    def _safe_groups(channels, target_groups):
        target_groups = max(1, min(target_groups, channels))
        for g in range(target_groups, 0, -1):
            if channels % g == 0:
                return g
        return 1


class ResidualLocalBlock(nn.Module):
    """
    轻量局部关系建模：
    深度卷积：对输入特征图的每个通道独立进行空间特征提取（不跨通道混合）；
    逐点卷积：对深度卷积的输出进行跨通道的信息融合（类似全连接层）
    """
    def __init__(self, channels, gn_groups=8, dropout=0.0):
        super().__init__()
        self.dw = ConvGNAct(
            in_ch=channels,
            out_ch=channels,
            k=3,
            s=1,
            p=1,
            groups=channels,
            gn_groups=gn_groups,
            act=True,
        )
        self.pw = ConvGNAct(
            in_ch=channels,
            out_ch=channels,
            k=1,
            s=1,
            p=0,
            groups=1,
            gn_groups=gn_groups,
            act=True,
        )
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return x + self.drop(self.pw(self.dw(x)))


# 把局部patch特征向量转换成二维局部特征图，做局部关系建模，再压缩成一个局部表征向量，供后续与全局特征融合使用。
class LocalPatchBranch(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim=256,
        out_dim=768,
        num_blocks=2,
        grid_size=None,
        # input_size=(224, 224),
        proj_dropout=0.1,
        block_dropout=0.0,
        gn_groups=8,
        # eps=1e-6,
    ):
        """
        :param in_dim:输入patch token特征向量的维度，默认为1024维；
        :param hidden_dim:局部分支内部处理时使用的隐藏通道数；
        :param out_dim:局部分支最后输出向量的维度；
        :param num_blocks:局部分支在 patch 网格上做多少层局部卷积推理
        :param grid_size:patch网格的高宽
        :param proj_dropout:投影层和最终输出投影层使用的dropout比例；
        :param block_dropout:局部残差块内部的dropout强度；
        :param gn_groups:分组数目标值
        """
        super().__init__()

        if in_dim <= 0 or hidden_dim <= 0 or out_dim <= 0:
            raise ValueError("in_dim, hidden_dim and out_dim must be positive")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        # if len(input_size) != 2 or input_size[0] <= 0 or input_size[1] <= 0:
        #     raise ValueError(f"invalid input_size: {input_size}")

        if grid_size is not None:
            if len(grid_size) != 2 or grid_size[0] <= 0 or grid_size[1] <= 0:
                raise ValueError(f"invalid grid_size: {grid_size}")

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        # self.input_size = input_size
        # self.eps = eps

        gn_proj = ConvGNAct._safe_groups(hidden_dim, gn_groups)

        # 先把原始patch token特征，投影到一个更适合做局部检测的特征空间
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(gn_proj, hidden_dim),
            nn.GELU(),
            nn.Dropout2d(proj_dropout) if proj_dropout > 0 else nn.Identity(),
        )

        # 局部关系建模
        self.local_blocks = nn.Sequential(
            *[
                ResidualLocalBlock(
                    channels=hidden_dim,
                    gn_groups=gn_groups,
                    dropout=block_dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        # 生成 网格热图
        # self.mask_head = nn.Conv2d(hidden_dim, 1, kernel_size=1, stride=1, padding=0)

        # 局部向量投影到 768 维
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
            nn.GELU(),
            nn.Dropout(proj_dropout) if proj_dropout > 0 else nn.Identity(),
        )

    # 根据token数量N推断patch网格大小 (H, W)
    def _infer_grid_size(self, n):
        if self.grid_size is not None:
            h, w = self.grid_size
            if h * w != n:
                raise ValueError(f"grid_size={self.grid_size} does not match N={n}")
            return h, w

        side = int(math.sqrt(n))
        if side * side != n:
            raise ValueError(f"Cannot infer square grid from N={n}, please set grid_size")
        return side, side

    # 把输入的局部向量还原成二维特征图
    def _tokens_to_map(self, patch_tokens):
        if patch_tokens.dim() != 3:
            raise ValueError(f"patch_tokens must be [B, N, C], got {tuple(patch_tokens.shape)}")

        b, n, c = patch_tokens.shape
        if c != self.in_dim:
            raise ValueError(f"Expected token dim {self.in_dim}, got {c}")

        h, w = self._infer_grid_size(n)
        feat_map = patch_tokens.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return feat_map

    # 利用热图权重，对局部特征图做加权平均池化，得到一个局部向量
    # def _weighted_pool(self, feat_map, mask_logits):
        """
        feat_map: [B, C, H, W]
        mask_logits: [B, 1, H, W]
        """
        mask_prob = torch.sigmoid(mask_logits)
        weighted_feat = feat_map * mask_prob

        numerator = weighted_feat.sum(dim=(2, 3))                # [B, C]
        denominator = mask_prob.sum(dim=(2, 3)).clamp_min(self.eps)
        local_feat = numerator / denominator

        return local_feat, mask_prob

    # 网格热图上采样到原图
    # def _upsample_heatmap(self, heatmap):
        """
        heatmap: [B, 1, h, w]
        return:  [B, 1, H_in, W_in]
        """
        return F.interpolate(
            heatmap,
            size=self.input_size,
            mode="bilinear",
            align_corners=False,
        )

    # 前向完整过程
    def forward(self, patch_tokens):
        # [B, N, C] -> [B, C, H, W]
        # feat_map = self._tokens_to_map(patch_tokens)
        #
        # # 局部建模
        # x = self.proj(feat_map)
        # x = self.local_blocks(x)
        #
        # # # patch 网格热图（内部使用）
        # mask_logits = self.mask_head(x)
        #
        # 局部向量
        # local_feat_raw, mask_prob = self._weighted_pool(x, mask_logits)
        # local_feat = self.feature_proj(local_feat_raw)           # [B, 768]

        # 映射回输入图大小后的热图
        # local_heatmap = self._upsample_heatmap(mask_prob)
        # return {
        #     "local_feat": local_feat,
        #     "local_heatmap": local_heatmap,
        # }

        feat_map = self._tokens_to_map(patch_tokens)

        x = self.proj(feat_map)
        x = self.local_blocks(x)

        # 直接平均池化，不再使用热图
        local_feat_raw = x.mean(dim=(2, 3))  # [B, hidden_dim]
        local_feat = self.feature_proj(local_feat_raw)

        return {
            "local_feat": local_feat,
        }



