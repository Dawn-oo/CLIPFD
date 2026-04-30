import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data_deal import build_test_loader
from models.assemble_model import CLIPFDModel
from options.train_options import TrainOptions


CKPT_PATH = r"E:\Project\CLIPFD\checkpoints\clipfd_exp\best.pth"
SAVE_DIR = r"outputs/局部位置贡献"
MAX_SAVE = 20

# 用验证集做可视化
USE_VAL_SET = True


MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]


class GradCAMHook:
    def __init__(self, target_layer):
        self.activations = None
        self.gradients = None

        self.fwd_hook = target_layer.register_forward_hook(self._forward_hook)
        self.bwd_hook = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def close(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()


def denormalize(img_tensor, mean, std):
    """
    img_tensor: [3, H, W]
    """
    mean = torch.tensor(mean, device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor(std, device=img_tensor.device).view(3, 1, 1)
    return img_tensor * std + mean


def cam_to_color(cam_2d: np.ndarray) -> np.ndarray:
    """
    [H, W], 0~1 -> [H, W, 3], uint8
    简单暖色伪彩色，不依赖 opencv
    """
    x = cam_2d.astype(np.float32)
    r = np.clip(1.8 * x, 0, 1)
    g = np.clip(1.8 * x - 0.6, 0, 1)
    b = np.clip(1.2 * x - 1.0, 0, 1)
    color = np.stack([r, g, b], axis=-1)
    return (color * 255).astype(np.uint8)


def overlay_cam_on_image(img_tensor, cam_tensor, mean, std, alpha=0.45):
    """
    img_tensor: [3, H, W]
    cam_tensor: [1, H, W]
    return: PIL.Image
    """
    img = denormalize(img_tensor, mean, std).detach().cpu()
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)

    cam = cam_tensor.squeeze(0).detach().cpu().numpy()
    heatmap = cam_to_color(cam)

    overlay = ((1 - alpha) * img + alpha * heatmap).clip(0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def build_model_and_loader():
    opt = TrainOptions().parse()

    device = "cuda" if torch.cuda.is_available() and len(getattr(opt, "gpu_ids", [])) > 0 else "cpu"

    model = CLIPFDModel(
        backbone_name=opt.backbone_name,
        freeze_backbone=getattr(opt, "freeze_backbone", True),
        device=device,
        final_num_classes=getattr(opt, "final_num_classes", 3),
        aux_num_classes=getattr(opt, "aux_num_classes", 1),
        local_hidden_dim=getattr(opt, "local_hidden_dim", 256),
        local_out_dim=getattr(opt, "local_out_dim", 768),
        local_num_blocks=getattr(opt, "local_num_blocks", 2),
        proj_dropout=getattr(opt, "proj_dropout", 0.1),
        block_dropout=getattr(opt, "block_dropout", 0.0),
        gn_groups=getattr(opt, "gn_groups", 8),
        fusion_dropout=getattr(opt, "fusion_dropout", 0.1),
        use_global_aux_head=True,
    ).to(device)

    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    common_kwargs = dict(
        batch_size=getattr(opt, "batch_size", 16),
        image_size=getattr(opt, "image_size", 224),
        load_size=getattr(opt, "load_size", 256),
        num_workers=getattr(opt, "num_workers", 4),
        pin_memory=getattr(opt, "pin_memory", False),
        persistent_workers=getattr(opt, "persistent_workers", False),
        no_crop=getattr(opt, "no_crop", False),
    )

    if USE_VAL_SET:
        _, loader = build_test_loader(
            image_root=opt.val_image_root,
            label_json_path=opt.val_label_json,
            **common_kwargs,
        )
    else:
        _, loader = build_test_loader(
            image_root=opt.train_image_root,
            label_json_path=opt.train_label_json,
            **common_kwargs,
        )

    return model, loader, device


def main():
    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    model, loader, device = build_model_and_loader()

    # 关键：挂在局部分支最后一个空间层
    target_layer = model.local_branch.local_blocks[-1]
    cam_hook = GradCAMHook(target_layer)

    saved = 0

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device)

        # 这里不能用 no_grad，因为要反传梯度
        outputs = model(images, return_aux=True, return_features=False)
        logits = outputs["logits"]   # [B, 3]

        pred_class = logits.argmax(dim=1)  # 用模型预测类别做 CAM

        score = logits[torch.arange(logits.size(0), device=device), pred_class].sum()

        model.zero_grad(set_to_none=True)
        score.backward()

        activations = cam_hook.activations     # [B, C, h, w]
        gradients = cam_hook.gradients         # [B, C, h, w]

        weights = gradients.mean(dim=(2, 3), keepdim=True)              # [B, C, 1, 1]
        cam = (weights * activations).sum(dim=1, keepdim=True)          # [B, 1, h, w]
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # 每张图单独归一化到 [0, 1]
        B = cam.size(0)
        cam_flat = cam.view(B, -1)
        cam_min = cam_flat.min(dim=1)[0].view(B, 1, 1, 1)
        cam_max = cam_flat.max(dim=1)[0].view(B, 1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        for i in range(B):
            if saved >= MAX_SAVE:
                cam_hook.close()
                print(f"Saved {saved} CAM images to: {save_dir}")
                return

            overlay_img = overlay_cam_on_image(
                img_tensor=images[i],
                cam_tensor=cam[i],
                mean=MEAN,
                std=STD,
                alpha=0.45,
            )

            out_path = save_dir / f"cam_{saved:03d}_pred{int(pred_class[i].item())}.png"
            overlay_img.save(out_path)
            saved += 1

    cam_hook.close()
    print(f"Saved {saved} CAM images to: {save_dir}")


if __name__ == "__main__":
    main()