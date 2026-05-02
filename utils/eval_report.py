from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

def setup_matplotlib_chinese():
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _dump_json(path: Path, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@dataclass
class EvaluationReporter:
    save_root: str
    tri_class_names: Sequence[str]
    aux_binary_class_names: Sequence[str] = ("真实图", "AI介入")
    history: Dict[str, List[Dict]] = field(default_factory=dict)

    def __post_init__(self):
        setup_matplotlib_chinese()
        self.save_dir = Path(self.save_root)
        _ensure_dir(self.save_dir)
        self.history = {"train": [], "val": []}

    # =========================
    # 对外接口
    # =========================
    def save_epoch_report(
        self,
        split: str,
        epoch: int,
        tri_y_true: Optional[np.ndarray],
        tri_y_prob: Optional[np.ndarray],
        bin_y_true: Optional[np.ndarray] = None,
        bin_y_prob: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """每个 epoch 调用一次：保存该epoch的metrics.json、将本轮标量指标写入history、不保存confusion matrix"""

        base_dir = self.save_dir / split / f"epoch_{epoch:03d}"
        _ensure_dir(base_dir)

        flat_metrics: Dict[str, float] = {}

        if tri_y_true is not None and tri_y_prob is not None:
            tri_dir = base_dir / "tri_head"
            _ensure_dir(tri_dir)
            tri_metrics = self._save_multiclass_report(
                save_dir=tri_dir,
                y_true=np.asarray(tri_y_true).astype(np.int64),
                y_prob=np.asarray(tri_y_prob).astype(np.float64),
                class_names=self.tri_class_names,
                prefix="tri",
                save_confusion_matrix=False,
            )
            flat_metrics.update(tri_metrics)

        if bin_y_true is not None and bin_y_prob is not None:
            aux_dir = base_dir / "aux_binary_head"
            _ensure_dir(aux_dir)
            aux_metrics = self._save_binary_report(
                save_dir=aux_dir,
                y_true=np.asarray(bin_y_true).astype(np.int64),
                y_prob=np.asarray(bin_y_prob).astype(np.float64),
                class_names=self.aux_binary_class_names,
                prefix="aux",
                save_confusion_matrix=False,
            )
            flat_metrics.update(aux_metrics)

        summary = {
            "meta": {"split": split, "epoch": int(epoch)},
            "metrics": flat_metrics,
        }
        _dump_json(base_dir / "summary.json", summary)

        self._append_history(split=split, epoch=epoch, metrics=flat_metrics)
        return flat_metrics

    def save_best_report(
        self,
        split: str,
        epoch: int,
        best_metric_name: str,
        best_metric_value: float,
        tri_y_true: Optional[np.ndarray],
        tri_y_prob: Optional[np.ndarray],
        bin_y_true: Optional[np.ndarray] = None,
        bin_y_prob: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        仅在 best epoch 更新时调用：
        - 保存 best 的 confusion matrix
        - 保存 best 的 metrics.json / summary.json
        """
        base_dir = self.save_dir / "best" / split
        _ensure_dir(base_dir)

        flat_metrics: Dict[str, float] = {}

        if tri_y_true is not None and tri_y_prob is not None:
            tri_dir = base_dir / "tri_head"
            _ensure_dir(tri_dir)
            tri_metrics = self._save_multiclass_report(
                save_dir=tri_dir,
                y_true=np.asarray(tri_y_true).astype(np.int64),
                y_prob=np.asarray(tri_y_prob).astype(np.float64),
                class_names=self.tri_class_names,
                prefix="tri",
                save_confusion_matrix=True,
            )
            flat_metrics.update(tri_metrics)

        if bin_y_true is not None and bin_y_prob is not None:
            aux_dir = base_dir / "aux_binary_head"
            _ensure_dir(aux_dir)
            aux_metrics = self._save_binary_report(
                save_dir=aux_dir,
                y_true=np.asarray(bin_y_true).astype(np.int64),
                y_prob=np.asarray(bin_y_prob).astype(np.float64),
                class_names=self.aux_binary_class_names,
                prefix="aux",
                save_confusion_matrix=True,
            )
            flat_metrics.update(aux_metrics)

        summary = {
            "meta": {
                "split": split,
                "best_epoch": int(epoch),
                "best_metric_name": best_metric_name,
                "best_metric_value": float(best_metric_value),
            },
            "metrics": flat_metrics,
        }
        _dump_json(base_dir / "summary.json", summary)
        return flat_metrics

    def finalize(self) -> None:
        """
        训练结束后调用：
        - 保存 train/val 的 history.json 和 history.csv
        - 画 F1 / AUC 历史曲线，并高亮 best 点
        """
        for split in ["train", "val"]:
            self._save_history_files(split)

            self._plot_metric_history(
                split=split,
                metric_name="tri_f1_macro",
                title=f"{split.upper()} Tri-Head Macro F1 vs Epoch",
                save_name="tri_f1_macro_history.png",
                higher_better=True,
            )

            self._plot_metric_history(
                split=split,
                metric_name="macro_auc",
                title=f"{split.upper()} Tri-Head Macro AUC vs Epoch",
                save_name="tri_macro_auc_history.png",
                higher_better=True,
            )

            self._plot_metric_history(
                split=split,
                metric_name="aux_f1_macro",
                title=f"{split.upper()} Aux-Head Macro F1 vs Epoch",
                save_name="aux_f1_macro_history.png",
                higher_better=True,
            )

            self._plot_metric_history(
                split=split,
                metric_name="binary_auc",
                title=f"{split.upper()} Aux-Head AUC vs Epoch",
                save_name="aux_binary_auc_history.png",
                higher_better=True,
            )

    # =========================
    # history 相关
    # =========================
    def _append_history(self, split: str, epoch: int, metrics: Dict[str, float]) -> None:
        row = {"epoch": int(epoch)}
        for k, v in metrics.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                row[k] = float(v)
        self.history.setdefault(split, []).append(row)

    def _save_history_files(self, split: str) -> None:
        rows = self.history.get(split, [])
        if not rows:
            return

        split_dir = self.save_dir / split
        _ensure_dir(split_dir)

        with open(split_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

        columns = {"epoch"}
        for row in rows:
            columns.update(row.keys())
        columns.discard("epoch")
        fieldnames = ["epoch"] + sorted(columns)

        with open(split_dir / "history.csv", "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _plot_metric_history(
        self,
        split: str,
        metric_name: str,
        title: str,
        save_name: str,
        higher_better: bool = True,
    ) -> None:
        rows = [row for row in self.history.get(split, []) if metric_name in row]
        if not rows:
            return

        epochs = [row["epoch"] for row in rows]
        values = [row[metric_name] for row in rows]

        best_idx = int(np.argmax(values) if higher_better else np.argmin(values))
        best_epoch = epochs[best_idx]
        best_value = values[best_idx]

        split_dir = self.save_dir / split
        _ensure_dir(split_dir)

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, values, marker="o", linewidth=2, label=metric_name)
        plt.scatter(
            [best_epoch],
            [best_value],
            marker="*",
            s=220,
            label=f"best @ epoch {best_epoch}: {best_value:.4f}",
        )
        plt.annotate(
            f"{best_value:.4f}",
            xy=(best_epoch, best_value),
            xytext=(best_epoch, best_value),
        )

        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(split_dir / save_name, dpi=200, bbox_inches="tight")
        plt.close()

    # =========================
    # 三分类 head
    # =========================
    def _save_multiclass_report(
        self,
        save_dir: Path,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        class_names: Sequence[str],
        prefix: str,
        save_confusion_matrix: bool,
    ) -> Dict[str, float]:
        y_pred = np.argmax(y_prob, axis=1)
        flat_metrics, full_report = self._multiclass_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            class_names=class_names,
            prefix=prefix,
        )

        _dump_json(save_dir / "metrics.json", full_report)

        if save_confusion_matrix:
            self._plot_confusion_matrix(
                y_true=y_true,
                y_pred=y_pred,
                class_names=class_names,
                save_path=save_dir / "confusion_matrix.png",
                title="三分类任务混淆矩阵",
            )
            np.save(save_dir / "y_true.npy", y_true)
            np.save(save_dir / "y_prob.npy", y_prob)
            np.save(save_dir / "y_pred.npy", y_pred)

        return flat_metrics

    def _multiclass_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        class_names: Sequence[str],
        prefix: str,
    ) -> Tuple[Dict[str, float], Dict]:
        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=list(class_names),
            output_dict=True,
            zero_division=0,
        )

        flat = {
            f"{prefix}_overall_acc": float(accuracy_score(y_true, y_pred)),
            f"{prefix}_precision_macro": float(report["macro avg"]["precision"]),
            f"{prefix}_recall_macro": float(report["macro avg"]["recall"]),
            f"{prefix}_f1_macro": float(report["macro avg"]["f1-score"]),
            f"{prefix}_precision_weighted": float(report["weighted avg"]["precision"]),
            f"{prefix}_recall_weighted": float(report["weighted avg"]["recall"]),
            f"{prefix}_f1_weighted": float(report["weighted avg"]["f1-score"]),
        }

        y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
        try:
            flat["macro_auc"] = float(
                roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
            )
        except ValueError:
            pass

        for idx, class_name in enumerate(class_names):
            class_key = str(class_name)

            flat[f"{prefix}_precision_{class_key}"] = float(report[class_key]["precision"])
            flat[f"{prefix}_recall_{class_key}"] = float(report[class_key]["recall"])
            flat[f"{prefix}_f1_{class_key}"] = float(report[class_key]["f1-score"])
            flat[f"{prefix}_support_{class_key}"] = int(report[class_key]["support"])

            # one-vs-rest accuracy，避免你之前把 recall 当成 class_acc
            ovr_true = (y_true == idx).astype(np.int64)
            ovr_pred = (y_pred == idx).astype(np.int64)
            flat[f"{prefix}_ovr_acc_{class_key}"] = float((ovr_true == ovr_pred).mean())

        full_report = {
            "flat_metrics": flat,
            "classification_report": report,
        }
        return flat, full_report

    # =========================
    # 辅助二分类 head
    # =========================
    def _save_binary_report(
        self,
        save_dir: Path,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        class_names: Sequence[str],
        prefix: str,
        save_confusion_matrix: bool,
    ) -> Dict[str, float]:
        y_pred = (y_prob >= 0.5).astype(np.int64)
        flat_metrics, full_report = self._binary_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            class_names=class_names,
            prefix=prefix,
        )

        _dump_json(save_dir / "metrics.json", full_report)

        if save_confusion_matrix:
            self._plot_confusion_matrix(
                y_true=y_true,
                y_pred=y_pred,
                class_names=class_names,
                save_path=save_dir / "confusion_matrix.png",
                title="Best Aux-Head Confusion Matrix",
            )
            np.save(save_dir / "y_true.npy", y_true)
            np.save(save_dir / "y_prob.npy", y_prob)
            np.save(save_dir / "y_pred.npy", y_pred)

        return flat_metrics

    def _binary_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        class_names: Sequence[str],
        prefix: str,
    ) -> Tuple[Dict[str, float], Dict]:
        report = classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=list(class_names),
            output_dict=True,
            zero_division=0,
        )

        flat = {
            f"{prefix}_overall_acc": float(accuracy_score(y_true, y_pred)),
            f"{prefix}_precision_macro": float(report["macro avg"]["precision"]),
            f"{prefix}_recall_macro": float(report["macro avg"]["recall"]),
            f"{prefix}_f1_macro": float(report["macro avg"]["f1-score"]),
            f"{prefix}_precision_weighted": float(report["weighted avg"]["precision"]),
            f"{prefix}_recall_weighted": float(report["weighted avg"]["recall"]),
            f"{prefix}_f1_weighted": float(report["weighted avg"]["f1-score"]),
        }

        try:
            flat["binary_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            pass

        for idx, class_name in enumerate(class_names):
            class_key = str(class_name)

            flat[f"{prefix}_precision_{class_key}"] = float(report[class_key]["precision"])
            flat[f"{prefix}_recall_{class_key}"] = float(report[class_key]["recall"])
            flat[f"{prefix}_f1_{class_key}"] = float(report[class_key]["f1-score"])
            flat[f"{prefix}_support_{class_key}"] = int(report[class_key]["support"])

            ovr_true = (y_true == idx).astype(np.int64)
            ovr_pred = (y_pred == idx).astype(np.int64)
            flat[f"{prefix}_ovr_acc_{class_key}"] = float((ovr_true == ovr_pred).mean())

        full_report = {
            "flat_metrics": flat,
            "classification_report": report,
        }
        return flat, full_report

    # =========================
    # confusion matrix
    # =========================
    def _plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Sequence[str],
        save_path: Path,
        title: str,
    ) -> None:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        cm_norm = cm.astype(np.float64) / row_sum

        plt.figure(figsize=(7, 6))
        plt.imshow(cm_norm, interpolation="nearest")
        plt.title(title)
        plt.colorbar()

        ticks = np.arange(len(class_names))
        plt.xticks(ticks, class_names, rotation=45)
        plt.yticks(ticks, class_names)

        thresh = cm_norm.max() / 2.0 if cm_norm.size > 0 else 0.5
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                plt.text(
                    j,
                    i,
                    f"{cm_norm[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if cm_norm[i, j] > thresh else "black",
                )

        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()