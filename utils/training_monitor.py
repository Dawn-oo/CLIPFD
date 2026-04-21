from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _to_float(value):
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class TrainingVisualizer:
    """
    记录 epoch 级训练/验证指标，并在训练结束后统一输出：
    1) history.csv / history.json
    2) 标量曲线图（loss, acc, auc, lr...）
    3) summary.txt
    """
    save_root: str = "./training_vis"
    run_name: Optional[str] = None
    records: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.run_name is None:
            self.run_name = f"run_{timestamp}"
        self.save_dir = Path(self.save_root) / self.run_name
        _ensure_dir(self.save_dir)

    def update(
        self,
        epoch: int,
        train_metrics: Optional[Dict] = None,
        val_metrics: Optional[Dict] = None,
    ) -> None:
        row = {"epoch": int(epoch)}

        train_metrics = train_metrics or {}
        val_metrics = val_metrics or {}

        for k, v in train_metrics.items():
            fv = _to_float(v)
            if fv is not None:
                row[f"train_{k}"] = fv

        for k, v in val_metrics.items():
            fv = _to_float(v)
            if fv is not None:
                row[f"val_{k}"] = fv

        self.records.append(row)

    def _all_columns(self) -> List[str]:
        cols = {"epoch"}
        for row in self.records:
            cols.update(row.keys())
        cols.remove("epoch")
        return ["epoch"] + sorted(cols)

    def _base_metric_names(self) -> List[str]:
        names = set()
        for row in self.records:
            for k in row.keys():
                if k.startswith("train_"):
                    names.add(k[len("train_"):])
                elif k.startswith("val_"):
                    names.add(k[len("val_"):])
        return sorted(names)

    def save_history_csv(self) -> None:
        if not self.records:
            return
        save_path = self.save_dir / "history.csv"
        columns = self._all_columns()
        with open(save_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.records)

    def save_history_json(self) -> None:
        save_path = self.save_dir / "history.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)

    def _plot_scalar_curve(self, metric_name: str) -> None:
        if not self.records:
            return

        epochs = [row["epoch"] for row in self.records]
        train_key = f"train_{metric_name}"
        val_key = f"val_{metric_name}"

        train_vals = [row.get(train_key) for row in self.records]
        val_vals = [row.get(val_key) for row in self.records]

        has_train = any(v is not None for v in train_vals)
        has_val = any(v is not None for v in val_vals)
        if not has_train and not has_val:
            return

        plt.figure(figsize=(8, 5))

        if has_train:
            x = [e for e, v in zip(epochs, train_vals) if v is not None]
            y = [v for v in train_vals if v is not None]
            plt.plot(x, y, marker="o", label=f"train_{metric_name}")

        if has_val:
            x = [e for e, v in zip(epochs, val_vals) if v is not None]
            y = [v for v in val_vals if v is not None]
            plt.plot(x, y, marker="o", label=f"val_{metric_name}")

        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} vs Epoch")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{metric_name}.png", dpi=200, bbox_inches="tight")
        plt.close()

    def save_scalar_plots(self) -> None:
        for metric_name in self._base_metric_names():
            self._plot_scalar_curve(metric_name)

    def save_summary(self) -> None:
        if not self.records:
            return

        larger_better = {"tri_acc", "bin_acc", "macro_auc", "binary_auc", "auc", "f1", "precision", "recall", "ap"}
        smaller_better = {"loss", "loss_tri", "loss_bin"}

        lines = [f"Total epochs recorded: {len(self.records)}"]

        for metric_name in self._base_metric_names():
            val_key = f"val_{metric_name}"
            train_key = f"train_{metric_name}"

            val_series = [(row["epoch"], row[val_key]) for row in self.records if val_key in row]
            train_series = [(row["epoch"], row[train_key]) for row in self.records if train_key in row]

            if metric_name in smaller_better:
                series = val_series if val_series else train_series
                if series:
                    best_epoch, best_val = min(series, key=lambda x: x[1])
                    prefix = "val" if val_series else "train"
                    lines.append(f"Best {prefix}_{metric_name}: {best_val:.6f} @ epoch {best_epoch}")

            elif metric_name in larger_better:
                series = val_series if val_series else train_series
                if series:
                    best_epoch, best_val = max(series, key=lambda x: x[1])
                    prefix = "val" if val_series else "train"
                    lines.append(f"Best {prefix}_{metric_name}: {best_val:.6f} @ epoch {best_epoch}")

        with open(self.save_dir / "summary.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def finalize(self) -> None:
        self.save_history_csv()
        self.save_history_json()
        self.save_scalar_plots()
        self.save_summary()
        print(f"[TrainingVisualizer] saved to: {self.save_dir}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    save_path: str,
    normalize: bool = True,
    title: str = "Confusion Matrix",
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        cm = cm.astype(np.float64) / row_sum

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0 if cm.size > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_binary_roc_pr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    save_dir: str,
    prefix: str = "binary",
) -> Dict[str, float]:
    """
    y_true: [N], 0/1
    y_score: [N], probability for positive class
    """
    save_dir = Path(save_dir)
    _ensure_dir(save_dir)

    metrics = {}

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    metrics["auc"] = float(roc_auc)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{prefix.upper()} ROC")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}_roc.png", dpi=200, bbox_inches="tight")
    plt.close()

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    metrics["pr_auc"] = float(pr_auc)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR AUC={pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{prefix.upper()} PR")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}_pr.png", dpi=200, bbox_inches="tight")
    plt.close()

    return metrics


def plot_multiclass_roc_pr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Sequence[str],
    save_dir: str,
    prefix: str = "multiclass",
) -> Dict[str, float]:
    """
    y_true: [N], int labels
    y_prob: [N, C], class probabilities
    """
    save_dir = Path(save_dir)
    _ensure_dir(save_dir)

    num_classes = y_prob.shape[1]
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    # macro ROC AUC
    macro_auc = roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")

    # ROC curves
    plt.figure(figsize=(7, 6))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        class_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={class_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{prefix.upper()} ROC (macro AUC={macro_auc:.4f})")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}_roc.png", dpi=200, bbox_inches="tight")
    plt.close()

    # PR curves
    plt.figure(figsize=(7, 6))
    pr_auc_dict = {}
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        class_pr_auc = auc(recall, precision)
        pr_auc_dict[class_names[i]] = float(class_pr_auc)
        plt.plot(recall, precision, label=f"{class_names[i]} (AUC={class_pr_auc:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{prefix.upper()} PR")
    plt.legend(loc="lower left", fontsize=9)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}_pr.png", dpi=200, bbox_inches="tight")
    plt.close()

    metrics = {
        "macro_auc": float(macro_auc),
    }
    for k, v in pr_auc_dict.items():
        metrics[f"pr_auc_{k}"] = v
    return metrics

def compute_per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)

    metrics = {}
    valid_accs = []

    for class_idx, class_name in enumerate(class_names):
        mask = (y_true == class_idx)
        total = int(mask.sum())

        metrics[f"class_total_{class_name}"] = total

        if total == 0:
            metrics[f"class_acc_{class_name}"] = None
            continue

        acc = float((y_pred[mask] == class_idx).mean())
        metrics[f"class_acc_{class_name}"] = acc
        valid_accs.append(acc)

    if valid_accs:
        metrics["mean_class_acc"] = float(np.mean(valid_accs))

    return metrics


def save_epoch_classification_artifacts(
    save_dir: str,
    tri_y_true: Optional[np.ndarray] = None,
    tri_y_prob: Optional[np.ndarray] = None,
    tri_class_names: Optional[Sequence[str]] = None,
    bin_y_true: Optional[np.ndarray] = None,
    bin_y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    每个 epoch 调用一次：
    - 保存三分类 confusion matrix / ROC / PR
    - 保存二分类 confusion matrix / ROC / PR
    - 保存三个类别各自准确率
    - 保存原始 y_true / y_prob / confusion matrix，便于后续复查
    """
    save_dir = Path(save_dir)
    _ensure_dir(save_dir)

    metrics = {}

    if tri_y_true is not None and tri_y_prob is not None and tri_class_names is not None:
        tri_y_true = np.asarray(tri_y_true).astype(np.int64)
        tri_y_prob = np.asarray(tri_y_prob).astype(np.float64)
        tri_y_pred = np.argmax(tri_y_prob, axis=1)

        np.save(save_dir / "tri_y_true.npy", tri_y_true)
        np.save(save_dir / "tri_y_prob.npy", tri_y_prob)
        np.save(save_dir / "tri_y_pred.npy", tri_y_pred)

        tri_cm_raw = confusion_matrix(
            tri_y_true,
            tri_y_pred,
            labels=list(range(len(tri_class_names))),
        )
        np.save(save_dir / "tri_confusion_matrix_raw.npy", tri_cm_raw)

        plot_confusion_matrix(
            y_true=tri_y_true,
            y_pred=tri_y_pred,
            class_names=tri_class_names,
            save_path=str(save_dir / "tri_confusion_matrix.png"),
            normalize=True,
            title="3-Class Confusion Matrix",
        )

        tri_metrics = plot_multiclass_roc_pr(
            y_true=tri_y_true,
            y_prob=tri_y_prob,
            class_names=tri_class_names,
            save_dir=str(save_dir),
            prefix="tri",
        )

        class_acc_metrics = compute_per_class_accuracy(
            y_true=tri_y_true,
            y_pred=tri_y_pred,
            class_names=tri_class_names,
        )

        tri_all_metrics = {**tri_metrics, **class_acc_metrics}
        metrics.update(tri_all_metrics)

        with open(save_dir / "tri_metrics.json", "w", encoding="utf-8") as f:
            json.dump(tri_all_metrics, f, ensure_ascii=False, indent=2)

    if bin_y_true is not None and bin_y_prob is not None:
        bin_y_true = np.asarray(bin_y_true).astype(np.int64)
        bin_y_prob = np.asarray(bin_y_prob).astype(np.float64)
        bin_y_pred = (bin_y_prob >= 0.5).astype(np.int64)

        np.save(save_dir / "bin_y_true.npy", bin_y_true)
        np.save(save_dir / "bin_y_prob.npy", bin_y_prob)
        np.save(save_dir / "bin_y_pred.npy", bin_y_pred)

        bin_cm_raw = confusion_matrix(bin_y_true, bin_y_pred, labels=[0, 1])
        np.save(save_dir / "bin_confusion_matrix_raw.npy", bin_cm_raw)

        plot_confusion_matrix(
            y_true=bin_y_true,
            y_pred=bin_y_pred,
            class_names=["negative", "positive"],
            save_path=str(save_dir / "bin_confusion_matrix.png"),
            normalize=True,
            title="Binary Confusion Matrix",
        )

        bin_metrics = plot_binary_roc_pr(
            y_true=bin_y_true,
            y_score=bin_y_prob,
            save_dir=str(save_dir),
            prefix="bin",
        )

        metrics["binary_auc"] = float(bin_metrics["auc"])
        metrics["binary_pr_auc"] = float(bin_metrics["pr_auc"])

        with open(save_dir / "bin_metrics.json", "w", encoding="utf-8") as f:
            json.dump(bin_metrics, f, ensure_ascii=False, indent=2)

    with open(save_dir / "epoch_eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics