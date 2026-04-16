import sys
from pathlib import Path
from utils.log import Tee

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_FILE = PROJECT_ROOT / "log.txt"

sys.stdout = Tee(LOG_FILE, sys.stdout)
sys.stderr = Tee(LOG_FILE, sys.stderr)

from pathlib import Path

import torch

from data_deal import build_train_loader
from models.assemble_model import CLIPFDModel
from trainer.trainer import Trainer


def main():
    # =========================
    # 1. 路径与基础配置
    # =========================
    project_root = Path(__file__).resolve().parent

    train_image_root = project_root / "datasets" / "train_images"
    train_label_json = project_root / "datasets" / "train_labels.json"

    # 你的本地 CLIP 权重
    backbone_ckpt = project_root / "models" / "parameters" / "ViT-L-14.pt"

    if not train_image_root.exists():
        raise FileNotFoundError(f"train_image_root not found: {train_image_root}")
    if not train_label_json.exists():
        raise FileNotFoundError(f"train_label_json not found: {train_label_json}")
    if not backbone_ckpt.exists():
        raise FileNotFoundError(f"backbone_ckpt not found: {backbone_ckpt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("Debug model start")
    print(f"device           : {device}")
    print(f"train_image_root : {train_image_root}")
    print(f"train_label_json : {train_label_json}")
    print(f"backbone_ckpt    : {backbone_ckpt}")
    print("=" * 70)

    # =========================
    # 2. dataloader
    # =========================
    dataset, loader = build_train_loader(
        image_root=str(train_image_root),
        label_json_path=str(train_label_json),
        batch_size=2,
        image_size=224,
        load_size=256,
        num_workers=0,              # 调试阶段先用 0
        pin_memory=False,
        persistent_workers=False,
        no_crop=False,
        no_flip=False,
        blur_prob=0.0,              # 调试时先关掉增强，减少干扰
        jpg_prob=0.0,
    )

    print(f"dataset size : {len(dataset)}")
    print(f"loader steps : {len(loader)}")
    print("=" * 70)

    batch = next(iter(loader))

    print("Batch keys:", batch.keys())
    print("image shape      :", batch["image"].shape)
    print("image dtype      :", batch["image"].dtype)
    print("binary_label     :", batch["binary_label"])
    print("multi_label      :", batch["multi_label"])
    print("sample_id        :", batch["sample_id"])
    print("image_path       :", batch["image_path"])
    print("=" * 70)

    assert batch["image"].dim() == 4, "image batch should be [B, C, H, W]"
    assert batch["image"].shape[1] == 3, "image channel should be 3"
    assert batch["image"].shape[2] == 224 and batch["image"].shape[3] == 224, "image size should be 224x224"

    # =========================
    # 3. model
    # =========================
    model = CLIPFDModel(
        backbone_name=str(backbone_ckpt),   # 直接使用本地权重路径
        freeze_backbone=True,
        device=device,
        final_num_classes=3,
        aux_num_classes=1,
        local_hidden_dim=256,
        local_out_dim=768,
        local_num_blocks=2,
        proj_dropout=0.1,
        block_dropout=0.0,
        gn_groups=8,
        fusion_dropout=0.1,
        use_global_aux_head=True,
    )

    # =========================
    # 4. trainer
    # =========================
    trainer = Trainer(
        model=model,
        device=device,
        lr=1e-4,
        weight_decay=1e-4,
        optimizer_type="adamw",
        aux_loss_weight=0.3,
        label_smoothing=0.0,
        use_amp=False,             # 调试阶段先关 AMP，排错更直接
        grad_clip_norm=1.0,
        save_dir=str(project_root / "checkpoints" / "debug_model"),
    )

    # 手动搬到 device
    batch_device = trainer._move_batch_to_device(batch)

    # =========================
    # 5. forward
    # =========================
    outputs = trainer.model(
        batch_device["image"],
        return_aux=True,
        return_features=True,
    )

    print("Model outputs keys:", outputs.keys())

    if "logits" in outputs:
        print("logits shape       :", outputs["logits"].shape)
    if "global_logits" in outputs:
        print("global_logits shape:", outputs["global_logits"].shape)
    if "global_feat" in outputs:
        print("global_feat shape  :", outputs["global_feat"].shape)
    if "local_feat" in outputs:
        print("local_feat shape   :", outputs["local_feat"].shape)
    if "fused_feat" in outputs:
        print("fused_feat shape   :", outputs["fused_feat"].shape)

    print("=" * 70)

    # 形状检查
    assert outputs["logits"].shape[0] == batch_device["image"].shape[0]
    assert outputs["logits"].shape[1] == 3, "final head should output 3-class logits"
    assert outputs["global_logits"].shape[0] == batch_device["image"].shape[0]
    assert outputs["global_logits"].shape[1] == 1, "aux head should output 1 logit"

    # =========================
    # 6. loss
    # =========================
    loss_dict = trainer.compute_losses(outputs, batch_device)

    print("Loss dict:")
    for k, v in loss_dict.items():
        if torch.is_tensor(v):
            print(f"{k}: {v.item():.6f}")
        else:
            print(f"{k}: {v}")

    print("=" * 70)

    assert "loss" in loss_dict
    assert "loss_tri" in loss_dict
    assert torch.is_tensor(loss_dict["loss"])
    assert not torch.isnan(loss_dict["loss"]).any(), "loss is NaN"

    # =========================
    # 7. backward + step
    # =========================
    trainer.optimizer.zero_grad(set_to_none=True)
    loss_dict["loss"].backward()

    # 统计一下有梯度的参数数量
    grad_param_count = 0
    for name, p in trainer.model.named_parameters():
        if p.requires_grad and p.grad is not None:
            grad_param_count += 1

    print(f"Parameters with grad: {grad_param_count}")

    assert grad_param_count > 0, "No gradients found after backward()"

    trainer.optimizer.step()

    print("=" * 70)
    print("Backward + optimizer.step() passed.")
    print("debug_model.py smoke test passed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
