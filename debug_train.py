from pathlib import Path

import torch

from data_deal import build_train_loader, build_test_loader
from models.assemble_model import CLIPFDModel
from trainer.trainer import Trainer


def main():
    # =========================
    # 1. 基础路径与配置
    # =========================
    project_root = Path(__file__).resolve().parent

    train_image_root = project_root / "datasets" / "train_images"
    train_label_json = project_root / "datasets" / "train_labels.json"

    val_image_root = project_root / "datasets" / "val_images"
    val_label_json = project_root / "datasets" / "val_labels.json"

    backbone_ckpt = project_root / "models" / "parameters" / "ViT-L-14.pt"

    if not train_image_root.exists():
        raise FileNotFoundError(f"train_image_root not found: {train_image_root}")
    if not train_label_json.exists():
        raise FileNotFoundError(f"train_label_json not found: {train_label_json}")
    if not val_image_root.exists():
        raise FileNotFoundError(f"val_image_root not found: {val_image_root}")
    if not val_label_json.exists():
        raise FileNotFoundError(f"val_label_json not found: {val_label_json}")
    if not backbone_ckpt.exists():
        raise FileNotFoundError(f"backbone_ckpt not found: {backbone_ckpt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("Debug trainer start")
    print(f"device           : {device}")
    print(f"train_image_root : {train_image_root}")
    print(f"train_label_json : {train_label_json}")
    print(f"val_image_root   : {val_image_root}")
    print(f"val_label_json   : {val_label_json}")
    print(f"backbone_ckpt    : {backbone_ckpt}")
    print("=" * 70)

    # =========================
    # 2. dataloader
    # =========================
    train_dataset, train_loader = build_train_loader(
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
        blur_prob=0.0,              # 调试先关增强
        jpg_prob=0.0,
    )

    val_dataset, val_loader = build_test_loader(
        image_root=str(val_image_root),
        label_json_path=str(val_label_json),
        batch_size=2,
        image_size=224,
        load_size=256,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        no_crop=False,
    )

    print(f"train dataset size : {len(train_dataset)}")
    print(f"train loader steps : {len(train_loader)}")
    print(f"val dataset size   : {len(val_dataset)}")
    print(f"val loader steps   : {len(val_loader)}")
    print("=" * 70)

    # =========================
    # 3. model
    # =========================
    model = CLIPFDModel(
        backbone_name=str(backbone_ckpt),
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
    ckpt_dir = project_root / "checkpoints" / "debug_trainer"

    trainer = Trainer(
        model=model,
        device=device,
        lr=1e-4,
        weight_decay=1e-4,
        optimizer_type="adamw",
        aux_loss_weight=0.3,
        label_smoothing=0.0,
        use_amp=False,              # 调试阶段先关 AMP
        grad_clip_norm=1.0,
        save_dir=str(ckpt_dir),
    )

    # =========================
    # 5. 训练 1 个 epoch
    # =========================
    print("Running trainer.train_one_epoch(...)")
    train_metrics = trainer.train_one_epoch(
        train_loader,
        epoch=0,
        log_interval=1,
    )

    print("=" * 70)
    print("train_metrics:")
    for k, v in train_metrics.items():
        print(f"{k}: {v}")
    print("=" * 70)

    assert "loss" in train_metrics
    assert "loss_tri" in train_metrics
    assert "tri_acc" in train_metrics
    assert "lr" in train_metrics

    # =========================
    # 6. 验证 1 次
    # =========================
    print("Running trainer.evaluate(...)")
    val_metrics = trainer.evaluate(
        val_loader,
        epoch=0,
    )

    print("=" * 70)
    print("val_metrics:")
    for k, v in val_metrics.items():
        print(f"{k}: {v}")
    print("=" * 70)

    assert "loss" in val_metrics
    assert "loss_tri" in val_metrics
    assert "tri_acc" in val_metrics

    # AUC 不一定在极小样本集里总能算出来，但如果能算出来就打印
    if "macro_auc" in val_metrics:
        print(f"macro_auc: {val_metrics['macro_auc']}")
    if "binary_auc" in val_metrics:
        print(f"binary_auc: {val_metrics['binary_auc']}")

    # =========================
    # 7. 保存 checkpoint
    # =========================
    ckpt_name = "debug_epoch0.pth"
    trainer.save_checkpoint(
        filename=ckpt_name,
        epoch=0,
        extra={
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        },
    )

    ckpt_path = ckpt_dir / ckpt_name
    assert ckpt_path.exists(), f"checkpoint not found: {ckpt_path}"

    print("=" * 70)
    print(f"Checkpoint saved successfully: {ckpt_path}")
    print("debug_trainer.py passed.")
    print("=" * 70)


if __name__ == "__main__":
    main()