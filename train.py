from __future__ import annotations

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from data_deal import build_train_loader, build_test_loader
from models.assemble_model import CLIPFDModel
from options.train_options import TrainOptions
from trainer.trainer import Trainer
from utils.training_monitor import TrainingVisualizer,save_epoch_classification_artifacts


# 用于从配置对象中安全地提取参数值
def opt_get(opt, name, default):
    return getattr(opt, name, default)


def get_device(opt) -> str:
    gpu_ids = opt_get(opt, "gpu_ids", [])
    return "cuda" if len(gpu_ids) > 0 else "cpu"

# 构建训练集和测试集两个数据集的实例对象，构建训练集和测试集两个数据加载器的实例对象
def build_dataloaders(opt):
    common_kwargs = dict(
        batch_size=opt.batch_size,
        image_size=opt_get(opt, "image_size", 224),
        load_size=opt_get(opt, "load_size", 256),
        num_workers=opt_get(opt, "num_workers", 4),
        pin_memory=opt_get(opt, "pin_memory", False),
        persistent_workers=opt_get(opt, "persistent_workers", False),
        no_crop=opt_get(opt, "no_crop", False),
    )

    train_dataset, train_loader = build_train_loader(
        image_root=opt.train_image_root,
        label_json_path=opt.train_label_json,
        no_flip=opt_get(opt, "no_flip", False),
        blur_prob=opt_get(opt, "blur_prob", 0.0),
        blur_radius=opt_get(opt, "blur_radius", (0.1, 1.5)),
        jpg_prob=opt_get(opt, "jpg_prob", 0.0),
        jpg_quality=opt_get(opt, "jpg_quality", (65, 95)),
        **common_kwargs,
    )

    _, train_eval_loader = build_test_loader(image_root=opt.train_image_root,label_json_path=opt.train_label_json,**common_kwargs)

    val_dataset, val_loader = build_test_loader(image_root=opt.val_image_root,label_json_path=opt.val_label_json,**common_kwargs)

    return train_dataset, train_loader, train_eval_loader, val_dataset, val_loader


def build_model(opt, device: str) -> CLIPFDModel:
    model = CLIPFDModel(
        backbone_name=opt.backbone_name,
        freeze_backbone=opt_get(opt, "freeze_backbone", True),
        device=device,
        final_num_classes=opt_get(opt, "final_num_classes", 3),
        aux_num_classes=opt_get(opt, "aux_num_classes", 1),
        local_hidden_dim=opt_get(opt, "local_hidden_dim", 256),
        local_out_dim=opt_get(opt, "local_out_dim", 768),
        local_num_blocks=opt_get(opt, "local_num_blocks", 2),
        proj_dropout=opt_get(opt, "proj_dropout", 0.1),
        block_dropout=opt_get(opt, "block_dropout", 0.0),
        gn_groups=opt_get(opt, "gn_groups", 8),
        fusion_dropout=opt_get(opt, "fusion_dropout", 0.1),
        use_global_aux_head=opt_get(opt, "use_global_aux_head", True),
    )
    return model


def build_trainer(opt, model, save_dir: Path, device: str) -> Trainer:
    trainer = Trainer(
        model=model,
        device=device,
        lr=opt_get(opt, "lr", 1e-4),
        weight_decay=opt_get(opt, "weight_decay", 1e-4),
        optimizer_type=opt_get(opt, "optimizer", "adamw"),
        aux_loss_weight=opt_get(opt, "aux_loss_weight", 0.3),
        label_smoothing=opt_get(opt, "label_smoothing", 0.0),
        use_amp=opt_get(opt, "use_amp", False),
        grad_clip_norm=opt_get(opt, "grad_clip_norm", 1.0),
        save_dir=str(save_dir),
    )
    return trainer


def choose_best_metric(val_metrics: dict) -> str:
    # 平均AUC水平
    if "macro_auc" in val_metrics and val_metrics["macro_auc"] == val_metrics["macro_auc"]:
        return "macro_auc"
    if "tri_acc" in val_metrics:
        return "tri_acc"
    if "binary_auc" in val_metrics and val_metrics["binary_auc"] == val_metrics["binary_auc"]:
        return "binary_auc"
    return "loss"


def is_better(metric_name: str, current: float, best: float) -> bool:
    if metric_name == "loss":
        return current < best
    return current > best


def metric_init_value(metric_name: str) -> float:
    if metric_name == "loss":
        return float("inf")
    return float("-inf")


def print_metrics(prefix: str, metrics: dict):
    parts = [prefix]
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.6f}")
        else:
            parts.append(f"{k}={v}")
    print(" | ".join(parts))

# 将每个epoch计算得到的评估指标记录到tensorboard的日志中
def log_metrics(writer, split: str, metrics: dict, epoch: int):
    if writer is None:
        return
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            writer.add_scalar(f"{split}/{k}", v, epoch)

def make_ckpt_extra(train_loop_metrics, train_metrics, val_metrics):
    return {
        "train_loop_metrics": train_loop_metrics,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }

def main():
    print("=" * 80)
    print("开始加载训练阶段参数配置...")
    opt = TrainOptions().parse()
    device = get_device(opt)
    print("训练阶段参数配置加载完成")
    save_dir = Path(opt.checkpoints_dir) / opt.name
    save_dir.mkdir(parents=True, exist_ok=True)
    print("参数配置文件保存成功")

    print("=" * 80)
    print("Start training")
    print(f"device   : {device}")
    print(f"save_dir : {save_dir}")
    print("=" * 80)

    # 1. 数据集实例和数据加载实例
    print("=" * 80)
    print("加载数据导入模块")
    train_dataset, train_loader, train_eval_loader, val_dataset, val_loader = build_dataloaders(opt)

    print(f"Train dataset size : {len(train_dataset)}")
    print(f"Val dataset size   : {len(val_dataset)}")
    print(f"Train steps/epoch  : {len(train_loader)}")
    print(f"Val steps          : {len(val_loader)}")
    print("数据导入模块加载完成")
    print("=" * 80)

    # 2. 构建模型实例
    print("=" * 80)
    print("开始加载模型")
    model = build_model(opt, device)
    print("模型加载完成")
    print("=" * 80)

    # 3. 构建训练加载器实例
    print("=" * 80)
    print("开始加载训练器")
    trainer = build_trainer(opt, model, save_dir, device)
    print("训练器加载完成，开始训练...")
    print("=" * 80)


    # 4. 训练过程中实时监控
    train_writer = SummaryWriter(str(save_dir / "tensorboard" / "train")) if SummaryWriter else None
    val_writer = SummaryWriter(str(save_dir / "tensorboard" / "val")) if SummaryWriter else None

    # 5. 可视化实例
    visualizer = TrainingVisualizer(save_root=str(save_dir / "training_vis")) if TrainingVisualizer else None

    # 6. 在检查点恢复训练
    start_epoch = 0
    resume_path = opt_get(opt, "resume_path", None)
    if resume_path:
        print(f"Resume from: {resume_path}")
        start_epoch, _ = trainer.load_checkpoint(resume_path, strict=True)
        start_epoch += 1

    # 7. 训练实施
    epochs = opt_get(opt, "epochs", opt_get(opt, "niter", 20))
    save_epoch_freq = opt_get(opt, "save_epoch_freq", 1)

    best_metric_name = None
    best_metric_value = None

    tri_class_names = ["真实图", "AI生成", "AI修改"]
    for epoch in range(start_epoch, epochs):
        print(f"\n{'=' * 30} Epoch {epoch + 1}/{epochs} {'=' * 30}")

        # 这里的顺序一定要保持：
        # 1) 先真正训练
        # 2) 再用 eval 模式评估 train set
        # 3) 再评估 val set
        train_loop_metrics = trainer.train_one_epoch(
            train_loader,
            epoch=epoch,
            log_interval=opt_get(opt, "log_interval", 20),
        )
        print_metrics("[TrainLoop]", train_loop_metrics)
        log_metrics(train_writer, "train_loop", train_loop_metrics, epoch)

        train_metrics, train_details = trainer.evaluate(
            train_eval_loader,
            epoch=epoch,
            return_details=True,
        )
        print_metrics("[TrainEval]", train_metrics)
        log_metrics(train_writer, "train_eval", train_metrics, epoch)

        val_metrics, val_details = trainer.evaluate(
            val_loader,
            epoch=epoch,
            return_details=True,
        )
        print_metrics("[Val]", val_metrics)
        log_metrics(val_writer, "val", val_metrics, epoch)

        if save_epoch_classification_artifacts is not None:
            epoch_root = save_dir / "epoch_reports" / f"epoch_{epoch + 1:03d}"

            train_report_metrics = save_epoch_classification_artifacts(
                save_dir=str(epoch_root / "train"),
                tri_y_true=train_details["tri_y_true"],
                tri_y_prob=train_details["tri_y_prob"],
                tri_class_names=tri_class_names,
                bin_y_true=train_details["bin_y_true"],
                bin_y_prob=train_details["bin_y_prob"],
            )

            val_report_metrics = save_epoch_classification_artifacts(
                save_dir=str(epoch_root / "val"),
                tri_y_true=val_details["tri_y_true"],
                tri_y_prob=val_details["tri_y_prob"],
                tri_class_names=tri_class_names,
                bin_y_true=val_details["bin_y_true"],
                bin_y_prob=val_details["bin_y_prob"],
            )

            # 把三类准确率等数值补回metrics
            train_metrics.update({
                k: v for k, v in train_report_metrics.items()
                if isinstance(v, (int, float))
            })
            val_metrics.update({
                k: v for k, v in val_report_metrics.items()
                if isinstance(v, (int, float))
            })

        if visualizer is not None:
            vis_train_metrics = dict(train_metrics)
            vis_train_metrics["optim_loss"] = train_loop_metrics.get("loss")
            vis_train_metrics["lr"] = train_loop_metrics.get("lr")

            visualizer.update(
                epoch=epoch,
                train_metrics=vis_train_metrics,
                val_metrics=val_metrics,
            )

        # 周期保存
        if (epoch + 1) % save_epoch_freq == 0:
            trainer.save_checkpoint(
                filename=f"epoch_{epoch + 1}.pth",
                epoch=epoch,
                extra=make_ckpt_extra(train_loop_metrics, train_metrics, val_metrics)
            )

        # best模型保存
        if best_metric_name is None:
            best_metric_name = choose_best_metric(val_metrics)
            best_metric_value = metric_init_value(best_metric_name)
            print(f"Best metric = {best_metric_name}")

        current_metric_value = float(val_metrics[best_metric_name])

        if is_better(best_metric_name, current_metric_value, best_metric_value):
            best_metric_value = current_metric_value
            trainer.save_checkpoint(
                filename="best.pth",
                epoch=epoch,
                extra=make_ckpt_extra(train_loop_metrics, train_metrics, val_metrics)
            )
            print(f"[Best] Updated: {best_metric_name}={best_metric_value:.6f}")

    # 资源清理
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()
    if visualizer is not None:
        visualizer.finalize()

    print("\nTraining finished.")


if __name__ == "__main__":
    main()