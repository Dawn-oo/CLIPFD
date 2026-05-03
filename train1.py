from __future__ import annotations

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from data_deal import build_train_loader, build_test_loader
from models.assemble_model import CLIPFDModel
from options.train_options import TrainOptions
from trainer.trainer import Trainer
from utils.training_monitor import TrainingVisualizer
from utils.eval_report import EvaluationReporter
import csv


def opt_get(opt, name, default):
    return getattr(opt, name, default)


def get_device(opt) -> str:
    gpu_ids = opt_get(opt, "gpu_ids", [])
    return "cuda" if len(gpu_ids) > 0 else "cpu"


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

    _, train_eval_loader = build_test_loader(
        image_root=opt.train_image_root,
        label_json_path=opt.train_label_json,
        **common_kwargs,
    )

    val_dataset, val_loader = build_test_loader(
        image_root=opt.val_image_root,
        label_json_path=opt.val_label_json,
        **common_kwargs,
    )

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
        use_global_aux_head=opt_get(opt, "use_global_aux_head", False),
        use_global_adapter=opt_get(opt, "use_global_adapter", True),
        global_adapter_dropout=opt_get(opt, "global_adapter_dropout", 0.1),
        fusion_mode=opt_get(opt, "fusion_mode", "full"),
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
        aux_loss_weight_end=opt_get(opt, "aux_loss_weight_end", 0.05),
        aux_weight_schedule=opt_get(opt, "aux_weight_schedule", "cosine_decay"),
        total_epochs=opt_get(opt, "epochs", opt_get(opt, "niter", 20)),
        scheduler_type=opt_get(opt, "scheduler_type", "cosine"),
        min_lr=opt_get(opt, "min_lr", 1e-6),
        label_smoothing=opt_get(opt, "label_smoothing", 0.0),
        use_amp=opt_get(opt, "use_amp", False),
        grad_clip_norm=opt_get(opt, "grad_clip_norm", 1.0),
        save_dir=str(save_dir),
    )
    return trainer


def choose_best_metric(val_metrics: dict) -> str:
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


def save_prediction_csv(
    save_path: Path,
    details: dict,
    tri_class_names=None,
    binary_class_names=None,
):
    if tri_class_names is None:
        tri_class_names = ["真实图", "AI生成", "AI修改"]

    if binary_class_names is None:
        binary_class_names = ["真实图", "AI介入"]

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    sample_ids = details.get("sample_ids", [])
    image_paths = details.get("image_paths", [])

    tri_y_true = details.get("tri_y_true", None)
    tri_y_pred = details.get("tri_y_pred", None)
    tri_y_prob = details.get("tri_y_prob", None)

    bin_y_true = details.get("bin_y_true", None)
    bin_y_pred = details.get("bin_y_pred", None)
    bin_y_prob = details.get("bin_y_prob", None)

    if image_paths:
        n = len(image_paths)
    elif tri_y_true is not None:
        n = len(tri_y_true)
    else:
        n = 0

    fieldnames = [
        "sample_id",
        "image_name",
        "image_path",
        "binary_true_label",
        "binary_true_name",
        "binary_pred_label",
        "binary_pred_name",
        "binary_ai_prob",
        "tri_true_label",
        "tri_true_name",
        "tri_pred_label",
        "tri_pred_name",
        "tri_prob_real",
        "tri_prob_ai_generate",
        "tri_prob_ai_edit",
    ]

    with open(save_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(n):
            image_path = image_paths[i] if i < len(image_paths) else ""
            image_name = Path(image_path).name if image_path else ""

            if i < len(sample_ids):
                sample_id = sample_ids[i]
            elif image_path:
                sample_id = Path(image_path).stem
            else:
                sample_id = ""

            row = {
                "sample_id": sample_id,
                "image_name": image_name,
                "image_path": image_path,
                "binary_true_label": "",
                "binary_true_name": "",
                "binary_pred_label": "",
                "binary_pred_name": "",
                "binary_ai_prob": "",
                "tri_true_label": "",
                "tri_true_name": "",
                "tri_pred_label": "",
                "tri_pred_name": "",
                "tri_prob_real": "",
                "tri_prob_ai_generate": "",
                "tri_prob_ai_edit": "",
            }

            if bin_y_true is not None:
                true_bin = int(bin_y_true[i])
                row["binary_true_label"] = true_bin
                row["binary_true_name"] = binary_class_names[true_bin]

            if bin_y_pred is not None:
                pred_bin = int(bin_y_pred[i])
                row["binary_pred_label"] = pred_bin
                row["binary_pred_name"] = binary_class_names[pred_bin]

            if bin_y_prob is not None:
                row["binary_ai_prob"] = float(bin_y_prob[i])

            if tri_y_true is not None:
                true_tri = int(tri_y_true[i])
                row["tri_true_label"] = true_tri
                row["tri_true_name"] = tri_class_names[true_tri]

            if tri_y_pred is not None:
                pred_tri = int(tri_y_pred[i])
                row["tri_pred_label"] = pred_tri
                row["tri_pred_name"] = tri_class_names[pred_tri]

            if tri_y_prob is not None:
                row["tri_prob_real"] = float(tri_y_prob[i][0])
                row["tri_prob_ai_generate"] = float(tri_y_prob[i][1])
                row["tri_prob_ai_edit"] = float(tri_y_prob[i][2])

            writer.writerow(row)

    print(f"Prediction CSV saved to: {save_path}")


def main():
    print("=" * 80)
    print("开始加载消融实验训练参数配置...")
    opt = TrainOptions().parse()
    opt.fusion_mode = "global_only"
    opt.use_global_aux_head = False
    opt.aux_loss_weight = 0.0
    opt.use_global_adapter = True
    device = get_device(opt)

    print("训练参数配置加载完成")
    print(f"当前 fusion_mode = {opt.fusion_mode}")
    print(f"当前 use_global_aux_head = {opt.use_global_aux_head}")
    print(f"当前 use_global_adapter = {opt.use_global_adapter}")

    save_dir = Path(opt.checkpoints_dir) / opt.name
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("训练开始")
    print(f"device   : {device}")
    print(f"save_dir : {save_dir}")
    print("=" * 80)

    print("=" * 80)
    print("加载数据导入模块")
    train_dataset, train_loader, train_eval_loader, val_dataset, val_loader = build_dataloaders(opt)
    print(f"训练集大小 : {len(train_dataset)}")
    print(f"验证集大小 : {len(val_dataset)}")
    print(f"训练集加载批次 : {len(train_loader)}")
    print(f"验证集加载批次 : {len(val_loader)}")
    print("数据导入模块加载完成")
    print("=" * 80)

    print("=" * 80)
    print("开始加载模型")
    model = build_model(opt, device)
    print("模型加载完成")
    print("=" * 80)

    print("=" * 80)
    print("开始加载训练器")
    trainer = build_trainer(opt, model, save_dir, device)
    print("训练器加载完成，开始训练...")
    print("=" * 80)

    train_writer = SummaryWriter(str(save_dir / "tensorboard" / "train"))
    val_writer = SummaryWriter(str(save_dir / "tensorboard" / "val"))

    visualizer = TrainingVisualizer(save_root=str(save_dir / "training_vis")) if TrainingVisualizer else None
    reporter = EvaluationReporter(
        save_root=str(save_dir / "eval_reports"),
        tri_class_names=["真实图", "AI生成", "AI修改"],
        aux_binary_class_names=("真实图", "AI介入"),
    )

    start_epoch = 0
    resume_path = opt_get(opt, "resume_path", None)
    if resume_path:
        print(f"Resume from: {resume_path}")
        start_epoch, _ = trainer.load_checkpoint(resume_path, strict=True)
        start_epoch += 1

    epochs = opt_get(opt, "epochs", opt_get(opt, "niter", 20))
    save_epoch_freq = opt_get(opt, "save_epoch_freq", 1)

    best_metric_name = None
    best_metric_value = None

    for epoch in range(start_epoch, epochs):
        epoch_num = epoch + 1
        print(f"\n{'=' * 30} Epoch {epoch_num}/{epochs} {'=' * 30}")

        trainer.update_aux_loss_weight(epoch)
        print(f"[辅助二分类损失参数权重] 训练周期={epoch_num}, 二分类辅助损失权重 = {trainer.aux_loss_weight:.6f}")

        current_lr = trainer.update_learning_rate(epoch)
        print(f"[学习率参数] epoch={epoch_num}, lr={current_lr:.8f}")

        train_loop_metrics = trainer.train_one_epoch(
            train_loader,
            epoch=epoch,
            log_interval=opt_get(opt, "log_interval", 20),
        )
        print_metrics("[TrainLoop]", train_loop_metrics)
        log_metrics(train_writer, "train_loop", train_loop_metrics, epoch_num)

        print("训练集评估中...")
        train_metrics, train_details = trainer.evaluate(
            train_eval_loader,
            epoch=epoch,
            return_details=True,
        )
        print_metrics("[TrainEval]", train_metrics)
        log_metrics(train_writer, "train_eval", train_metrics, epoch_num)

        print("验证集评估中...")
        val_metrics, val_details = trainer.evaluate(
            val_loader,
            epoch=epoch,
            return_details=True,
        )
        print_metrics("[Val]", val_metrics)
        log_metrics(val_writer, "val", val_metrics, epoch_num)

        train_report_metrics = reporter.save_epoch_report(
            split="train",
            epoch=epoch_num,
            tri_y_true=train_details["tri_y_true"],
            tri_y_prob=train_details["tri_y_prob"],
            bin_y_true=train_details["bin_y_true"],
            bin_y_prob=train_details["bin_y_prob"],
        )

        val_report_metrics = reporter.save_epoch_report(
            split="val",
            epoch=epoch_num,
            tri_y_true=val_details["tri_y_true"],
            tri_y_prob=val_details["tri_y_prob"],
            bin_y_true=val_details["bin_y_true"],
            bin_y_prob=val_details["bin_y_prob"],
        )

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
                epoch=epoch_num,
                train_metrics=vis_train_metrics,
                val_metrics=val_metrics,
            )

        if epoch_num % save_epoch_freq == 0:
            trainer.save_checkpoint(
                filename=f"epoch_{epoch_num}.pth",
                epoch=epoch,
                extra=make_ckpt_extra(train_loop_metrics, train_metrics, val_metrics),
            )

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
                extra=make_ckpt_extra(train_loop_metrics, train_metrics, val_metrics),
            )

            reporter.save_best_report(
                split="val",
                epoch=epoch_num,
                best_metric_name=best_metric_name,
                best_metric_value=best_metric_value,
                tri_y_true=val_details["tri_y_true"],
                tri_y_prob=val_details["tri_y_prob"],
                bin_y_true=val_details["bin_y_true"],
                bin_y_prob=val_details["bin_y_prob"],
            )

            save_prediction_csv(
                save_path=save_dir / "prediction_csv" / "best_val_predictions.csv",
                details=val_details,
                tri_class_names=["真实图", "AI生成", "AI修改"],
                binary_class_names=["真实图", "AI介入"],
            )

            save_prediction_csv(
                save_path=save_dir / "prediction_csv" / "best_train_predictions.csv",
                details=train_details,
                tri_class_names=["真实图", "AI生成", "AI修改"],
                binary_class_names=["真实图", "AI介入"],
            )

            print(f"[Best] Updated: {best_metric_name}={best_metric_value:.6f}")

    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()
    if visualizer is not None:
        visualizer.finalize()
    reporter.finalize()

    print("\nTraining finished.")


if __name__ == "__main__":
    main()