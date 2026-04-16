from __future__ import annotations

from pathlib import Path

from data_deal import build_train_loader, build_test_loader
from models.assemble_model import CLIPFDModel
from options.train_options import TrainOptions
from trainer.trainer import Trainer


def get_device(opt) -> str:
    if len(opt.gpu_ids) > 0:
        return "cuda"
    return "cpu"


def build_model(opt, device: str) -> CLIPFDModel:
    model = CLIPFDModel(
        backbone_name=opt.backbone_name,
        freeze_backbone=opt.freeze_backbone,
        device=device,
        final_num_classes=opt.final_num_classes,
        aux_num_classes=opt.aux_num_classes,
        local_hidden_dim=opt.local_hidden_dim,
        local_out_dim=opt.local_out_dim,
        local_num_blocks=opt.local_num_blocks,
        proj_dropout=opt.proj_dropout,
        block_dropout=opt.block_dropout,
        gn_groups=opt.gn_groups,
        fusion_dropout=opt.fusion_dropout,
        use_global_aux_head=opt.use_global_aux_head,
    )
    return model


def build_dataloaders(opt):
    train_dataset, train_loader = build_train_loader(
        image_root=opt.train_image_root,
        label_json_path=opt.train_label_json,
        batch_size=opt.batch_size,
        image_size=opt.image_size,
        load_size=opt.load_size,
        num_workers=opt.num_workers,
        pin_memory=opt.pin_memory,
        persistent_workers=opt.persistent_workers,
        no_crop=opt.no_crop,
        no_flip=opt.no_flip,
        blur_prob=opt.blur_prob,
        blur_radius=opt.blur_radius,
        jpg_prob=opt.jpg_prob,
        jpg_quality=opt.jpg_quality,
    )

    val_dataset, val_loader = build_test_loader(
        image_root=opt.val_image_root,
        label_json_path=opt.val_label_json,
        batch_size=opt.batch_size,
        image_size=opt.image_size,
        load_size=opt.load_size,
        num_workers=opt.num_workers,
        pin_memory=opt.pin_memory,
        persistent_workers=opt.persistent_workers,
        no_crop=opt.no_crop,
    )

    return train_dataset, train_loader, val_dataset, val_loader


def main():
    opt = TrainOptions().parse()
    device = get_device(opt)

    save_dir = Path(opt.checkpoints_dir) / opt.name
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. dataloader
    train_dataset, train_loader, val_dataset, val_loader = build_dataloaders(opt)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size:   {len(val_dataset)}")
    print(f"Train loader steps per epoch: {len(train_loader)}")
    print(f"Val loader steps:            {len(val_loader)}")

    # 2. model
    model = build_model(opt, device=device)

    # 3. trainer
    trainer = Trainer(
        model=model,
        device=device,
        lr=opt.lr,
        weight_decay=opt.weight_decay,
        optimizer_type=opt.optimizer,
        aux_loss_weight=opt.aux_loss_weight,
        label_smoothing=opt.label_smoothing,
        use_amp=opt.use_amp,
        grad_clip_norm=opt.grad_clip_norm,
        save_dir=str(save_dir),
    )

    # ===== 可选：断点恢复 =====
    start_epoch = 0
    # 如果你后面加 resume 参数，可以在这里恢复
    # if opt.resume_path:
    #     start_epoch, _ = trainer.load_checkpoint(opt.resume_path, strict=True)
    #     start_epoch += 1

    # 4. train loop
    best_metric = float("-inf")
    best_metric_name = "macro_auc" if opt.final_num_classes > 2 else "tri_acc"

    print(f"Best model selection metric: {best_metric_name}")

    for epoch in range(start_epoch, opt.epochs):
        print(f"\n========== Epoch {epoch + 1}/{opt.epochs} ==========")

        train_metrics = trainer.train_one_epoch(
            train_loader,
            epoch=epoch,
            log_interval=opt.log_interval,
        )
        print(f"[Train] {train_metrics}")

        val_metrics = trainer.evaluate(
            val_loader,
            epoch=epoch,
        )
        print(f"[Val]   {val_metrics}")

        # 5. 保存最新/周期模型
        if (epoch + 1) % opt.save_epoch_freq == 0:
            trainer.save_checkpoint(
                filename=f"epoch_{epoch + 1}.pth",
                epoch=epoch,
                extra={
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                },
            )

        # 6. 保存最佳模型
        current_metric = val_metrics.get(best_metric_name, None)

        # 如果没有 macro_auc，就退回 tri_acc
        if current_metric is None:
            current_metric = val_metrics.get("tri_acc", None)

        if current_metric is not None and current_metric > best_metric:
            best_metric = current_metric
            trainer.save_checkpoint(
                filename="best.pth",
                epoch=epoch,
                extra={
                    "best_metric_name": best_metric_name,
                    "best_metric_value": best_metric,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                },
            )
            print(f"[Best] Updated best model: {best_metric_name}={best_metric:.6f}")

    print("\nTraining finished.")


if __name__ == "__main__":
    main()
