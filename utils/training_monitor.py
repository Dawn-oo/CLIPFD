from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from .eval_report import setup_matplotlib_chinese
import matplotlib.pyplot as plt

setup_matplotlib_chinese()

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


def _metric_mode(metric_name: str) -> Optional[str]:
    smaller_better = {"loss", "loss_tri", "loss_bin", "optim_loss"}
    larger_better = {
        "tri_acc", "bin_acc", "macro_auc", "binary_auc",
        "tri_overall_acc", "aux_overall_acc",
        "tri_precision_macro", "tri_recall_macro", "tri_f1_macro",
        "aux_precision_macro", "aux_recall_macro", "aux_f1_macro",
    }

    if metric_name in smaller_better:
        return "min"
    if metric_name in larger_better:
        return "max"
    if metric_name.startswith(("tri_precision_", "tri_recall_", "tri_f1_", "tri_ovr_acc_")):
        return "max"
    if metric_name.startswith(("aux_precision_", "aux_recall_", "aux_f1_", "aux_ovr_acc_")):
        return "max"
    return None


@dataclass
class TrainingVisualizer:
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

        for prefix, metrics in [("train", train_metrics or {}), ("val", val_metrics or {})]:
            for k, v in metrics.items():
                fv = _to_float(v)
                if fv is not None:
                    row[f"{prefix}_{k}"] = fv

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
        print("本轮训练数据指标成功保存为CSV文件")

    def save_history_json(self) -> None:
        if not self.records:
            return
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

        lines = [f"Total epochs recorded: {len(self.records)}"]

        for metric_name in self._base_metric_names():
            mode = _metric_mode(metric_name)
            if mode is None:
                continue

            val_key = f"val_{metric_name}"
            train_key = f"train_{metric_name}"

            val_series = [(row["epoch"], row[val_key]) for row in self.records if val_key in row]
            train_series = [(row["epoch"], row[train_key]) for row in self.records if train_key in row]

            series = val_series if val_series else train_series
            if not series:
                continue

            if mode == "min":
                best_epoch, best_val = min(series, key=lambda x: x[1])
            else:
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