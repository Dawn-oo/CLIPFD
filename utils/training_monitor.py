from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


class TrainingMonitor:
    """
    用于记录训练过程中的指标，并在训练结束后统一保存曲线图。

    使用方式：
        monitor = TrainingMonitor(save_root="./training_vis")

        for epoch in range(num_epochs):
            train_metrics = trainer.train_one_epoch(...)
            val_metrics = trainer.evaluate(...)

            monitor.update(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
            )

        monitor.finalize()

    记录格式示例：
        train_metrics = {"loss": 0.52, "tri_acc": 0.81}
        val_metrics   = {"loss": 0.47, "tri_acc": 0.84, "macro_auc": 0.91}
    """

    def __init__(
        self,
        save_root: str = "./training_vis",
        run_name: Optional[str] = None,
    ):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_name is None:
            run_name = f"run_{timestamp}"

        self.save_dir = Path(save_root) / run_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.records: List[Dict] = []

    @staticmethod
    def _to_float_dict(metrics: Optional[Dict]) -> Dict[str, float]:
        if metrics is None:
            return {}

        out = {}
        for k, v in metrics.items():
            if v is None:
                continue

            # 支持 tensor / numpy scalar / float / int
            if hasattr(v, "item"):
                v = v.item()

            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                # 非标量就跳过
                continue

        return out

    def update(
        self,
        epoch: int,
        train_metrics: Optional[Dict] = None,
        val_metrics: Optional[Dict] = None,
    ):
        train_metrics = self._to_float_dict(train_metrics)
        val_metrics = self._to_float_dict(val_metrics)

        row = {"epoch": int(epoch)}

        for k, v in train_metrics.items():
            row[f"train_{k}"] = v

        for k, v in val_metrics.items():
            row[f"val_{k}"] = v

        self.records.append(row)

    def _get_all_columns(self) -> List[str]:
        cols = {"epoch"}
        for row in self.records:
            cols.update(row.keys())

        # epoch 放第一列，其余按字母序
        cols = list(cols)
        cols.remove("epoch")
        cols = ["epoch"] + sorted(cols)
        return cols

    def save_history_json(self):
        save_path = self.save_dir / "history.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)

    def save_history_csv(self):
        if not self.records:
            return

        columns = self._get_all_columns()
        save_path = self.save_dir / "history.csv"

        with open(save_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in self.records:
                writer.writerow(row)

    def _collect_base_metric_names(self) -> List[str]:
        """
        从 train_loss / val_loss 中提取出 base metric: loss
        """
        metric_names = set()

        for row in self.records:
            for k in row.keys():
                if k == "epoch":
                    continue
                if k.startswith("train_"):
                    metric_names.add(k[len("train_"):])
                elif k.startswith("val_"):
                    metric_names.add(k[len("val_"):])

        return sorted(metric_names)

    def _plot_single_metric(self, metric_name: str):
        epochs = [row["epoch"] for row in self.records]

        train_key = f"train_{metric_name}"
        val_key = f"val_{metric_name}"

        train_vals = [row.get(train_key, None) for row in self.records]
        val_vals = [row.get(val_key, None) for row in self.records]

        has_train = any(v is not None for v in train_vals)
        has_val = any(v is not None for v in val_vals)

        if not has_train and not has_val:
            return

        plt.figure(figsize=(8, 5))

        if has_train:
            x_train = [e for e, v in zip(epochs, train_vals) if v is not None]
            y_train = [v for v in train_vals if v is not None]
            plt.plot(x_train, y_train, marker="o", label=f"train_{metric_name}")

        if has_val:
            x_val = [e for e, v in zip(epochs, val_vals) if v is not None]
            y_val = [v for v in val_vals if v is not None]
            plt.plot(x_val, y_val, marker="o", label=f"val_{metric_name}")

        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} vs Epoch")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        save_path = self.save_dir / f"{metric_name}.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

    def _build_summary(self) -> str:
        if not self.records:
            return "No records."

        lines = []
        lines.append(f"Total epochs recorded: {len(self.records)}")

        # 常见的“越大越好”指标
        larger_better = ["tri_acc", "bin_acc", "macro_auc", "binary_auc", "auc", "f1", "ap"]

        # 常见的“越小越好”指标
        smaller_better = ["loss"]

        base_metrics = self._collect_base_metric_names()

        for metric in base_metrics:
            val_key = f"val_{metric}"
            train_key = f"train_{metric}"

            val_series = [(row["epoch"], row[val_key]) for row in self.records if val_key in row]
            train_series = [(row["epoch"], row[train_key]) for row in self.records if train_key in row]

            if metric in smaller_better:
                if val_series:
                    best_epoch, best_val = min(val_series, key=lambda x: x[1])
                    lines.append(f"Best val_{metric}: {best_val:.6f} @ epoch {best_epoch}")
                elif train_series:
                    best_epoch, best_val = min(train_series, key=lambda x: x[1])
                    lines.append(f"Best train_{metric}: {best_val:.6f} @ epoch {best_epoch}")
            elif metric in larger_better:
                if val_series:
                    best_epoch, best_val = max(val_series, key=lambda x: x[1])
                    lines.append(f"Best val_{metric}: {best_val:.6f} @ epoch {best_epoch}")
                elif train_series:
                    best_epoch, best_val = max(train_series, key=lambda x: x[1])
                    lines.append(f"Best train_{metric}: {best_val:.6f} @ epoch {best_epoch}")

        return "\n".join(lines)

    def save_summary(self):
        summary = self._build_summary()
        save_path = self.save_dir / "summary.txt"
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(summary)

    def finalize(self):
        """
        训练结束后统一保存：
        - history.json
        - history.csv
        - 各指标曲线图
        - summary.txt
        """
        if not self.records:
            print("[TrainingMonitor] No records to save.")
            return

        self.save_history_json()
        self.save_history_csv()

        for metric_name in self._collect_base_metric_names():
            self._plot_single_metric(metric_name)

        self.save_summary()

        print(f"[TrainingMonitor] All training curves saved to: {self.save_dir}")