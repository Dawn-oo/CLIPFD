from __future__ import annotations

from pathlib import Path
import csv

from utils.eval_report import EvaluationReporter
from data_deal import build_test_loader
from models.assemble_model import CLIPFDModel
from options.test_options import TestOptions
from trainer.trainer import Trainer


def opt_get(opt, name, default):
    return getattr(opt, name, default)


def get_device(opt) -> str:
    gpu_ids = opt_get(opt, "gpu_ids", [])
    return "cuda" if len(gpu_ids) > 0 else "cpu"


def build_test_dataloader(opt):
    common_kwargs = dict(
        batch_size=opt.batch_size,
        image_size=opt_get(opt, "image_size", 224),
        load_size=opt_get(opt, "load_size", 256),
        num_workers=opt_get(opt, "num_workers", 4),
        pin_memory=opt_get(opt, "pin_memory", False),
        persistent_workers=opt_get(opt, "persistent_workers", False),
        no_crop=opt_get(opt, "no_crop", False),
    )

    test_dataset, test_loader = build_test_loader(
        image_root=opt.test_image_root,
        label_json_path=opt.test_label_json,
        **common_kwargs,
    )
    return test_dataset, test_loader


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
        use_global_adapter=opt_get(opt, "use_global_adapter", True),
        global_adapter_dropout=opt_get(opt, "global_adapter_dropout", 0.1),
        fusion_mode=opt_get(opt, "fusion_mode", "full")
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
        total_epochs=opt_get(opt, "epochs", 12),
        scheduler_type=opt_get(opt, "scheduler_type", "cosine"),
        min_lr=opt_get(opt, "min_lr", 1e-6),
        label_smoothing=opt_get(opt, "label_smoothing", 0.0),
        use_amp=opt_get(opt, "use_amp", False),
        grad_clip_norm=opt_get(opt, "grad_clip_norm", 1.0),
        save_dir=str(save_dir),
    )
    return trainer


def print_metrics(prefix: str, metrics: dict):
    parts = [prefix]
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.6f}")
        else:
            parts.append(f"{k}={v}")
    print(" | ".join(parts))


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
    print("开始加载测试阶段参数配置...")
    opt = TestOptions().parse()
    opt.use_global_aux_head = True
    device = get_device(opt)
    print("测试阶段参数配置加载完成")

    project_root = Path(__file__).resolve().parent
    save_dir = project_root / "test_result"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("开始测试")
    print(f"device   : {device}")
    print(f"save_dir : {save_dir}")
    print(f"ckpt     : {opt.ckpt_path}")
    print("=" * 80)

    # 1. 数据
    print("加载测试数据...")
    test_dataset, test_loader = build_test_dataloader(opt)
    print(f"测试集大小 : {len(test_dataset)}")
    print(f"测试集批次 : {len(test_loader)}")
    print("测试数据加载完成")
    print("=" * 80)

    # 2. 模型
    print("加载模型...")
    model = build_model(opt, device)
    print("模型加载完成")
    print("=" * 80)

    # 3. Trainer
    trainer = build_trainer(opt, model, save_dir, device)
    reporter = EvaluationReporter(save_root=str(save_dir / "eval_reports"),tri_class_names=["真实图", "AI生成", "AI修改"],aux_binary_class_names=("真实图", "AI介入"))

    # 4. 加载权重
    ckpt_path = Path(opt.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    trainer.load_checkpoint(str(ckpt_path), strict=True)

    # 5. 测试评估
    test_metrics, test_details = trainer.evaluate(
        test_loader,
        epoch=0,
        return_details=True,
    )
    print_metrics("[Test]", test_metrics)

    # 6. 保存预测结果
    reporter.save_best_report(
        split="test",
        epoch=1,
        best_metric_name="macro_auc" if "macro_auc" in test_metrics else "loss",
        best_metric_value=float(test_metrics["macro_auc"]) if "macro_auc" in test_metrics else float(
            test_metrics["loss"]),
        tri_y_true=test_details["tri_y_true"],
        tri_y_prob=test_details["tri_y_prob"],
        bin_y_true=test_details["bin_y_true"],
        bin_y_prob=test_details["bin_y_prob"],
    )

    if opt.save_predictions:
        save_prediction_csv(
            save_path=save_dir / "prediction_csv" / opt.prediction_csv_name,
            details=test_details,
            tri_class_names=["真实图", "AI生成", "AI修改"],
            binary_class_names=["真实图", "AI介入"],
        )

    print("\nTest finished.")


if __name__ == "__main__":
    main()