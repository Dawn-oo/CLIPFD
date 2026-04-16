from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Trainer:
    """
    面向当前 CLIPFDModel 的训练器

    约定：
    - batch 来自新的 data_deal/datasets.py
    - batch["image"]        : [B, 3, H, W]
    - batch["binary_label"] : [B]，float，0/1
    - batch["multi_label"]  : [B]，long，0/1/2

    模型输出：
    - outputs["logits"]         : [B, 3]
    - outputs["global_logits"]  : [B, 1]，可选
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        optimizer_type: str = "adamw",
        aux_loss_weight: float = 0.3,
        label_smoothing: float = 0.0,
        use_amp: bool = True,
        grad_clip_norm: Optional[float] = None,
        save_dir: str = "./checkpoints",
    ):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.model = model.to(self.device)

        self.aux_loss_weight = aux_loss_weight
        self.use_amp = use_amp and self.device.type == "cuda"
        self.grad_clip_norm = grad_clip_norm
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        params = [p for p in self.model.parameters() if p.requires_grad]
        if len(params) == 0:
            raise RuntimeError("No trainable parameters found in model.")

        optimizer_type = optimizer_type.lower()
        if optimizer_type == "adamw":
            self.optimizer = torch.optim.AdamW(
                params,
                lr=lr,
                weight_decay=weight_decay,
            )
        elif optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(
                params,
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer_type: {optimizer_type}")

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def _move_batch_to_device(self, batch: Dict) -> Dict:
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device, non_blocking=True)
            else:
                out[k] = v
        return out

    def compute_losses(self, outputs: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        if "logits" not in outputs:
            raise KeyError("Model outputs must contain 'logits' for final classification.")

        if "multi_label" not in batch:
            raise KeyError("Batch must contain 'multi_label' for final 3-class loss.")

        logits = outputs["logits"]                 # [B, 3]
        multi_label = batch["multi_label"].long() # [B]

        loss_tri = self.ce_loss(logits, multi_label)
        total_loss = loss_tri

        loss_dict = {
            "loss": total_loss,
            "loss_tri": loss_tri.detach(),
        }

        if "global_logits" in outputs and "binary_label" in batch and self.aux_loss_weight > 0:
            global_logits = outputs["global_logits"]  # [B, 1] or [B]
            binary_label = batch["binary_label"].float()

            if global_logits.dim() == 2 and global_logits.size(1) == 1:
                global_logits = global_logits.squeeze(1)

            loss_bin = self.bce_loss(global_logits, binary_label)
            total_loss = total_loss + self.aux_loss_weight * loss_bin

            loss_dict["loss"] = total_loss
            loss_dict["loss_bin"] = loss_bin.detach()

        return loss_dict

    def _compute_batch_metrics(self, outputs: Dict, batch: Dict) -> Dict[str, float]:
        metrics = {}

        # 三分类准确率
        if "logits" in outputs and "multi_label" in batch:
            preds = outputs["logits"].argmax(dim=1)
            target = batch["multi_label"]
            tri_acc = (preds == target).float().mean().item()
            metrics["tri_acc"] = tri_acc

        # 全局辅助二分类准确率
        if "global_logits" in outputs and "binary_label" in batch:
            g = outputs["global_logits"]
            if g.dim() == 2 and g.size(1) == 1:
                g = g.squeeze(1)
            pred_bin = (torch.sigmoid(g) >= 0.5).long()
            target_bin = batch["binary_label"].long()
            bin_acc = (pred_bin == target_bin).float().mean().item()
            metrics["bin_acc"] = bin_acc

        return metrics

    def train_one_epoch(self, loader, epoch: int = 0, log_interval: int = 50):
        self.model.train()

        running = {
            "loss": 0.0,
            "loss_tri": 0.0,
            "loss_bin": 0.0,
            "tri_acc": 0.0,
            "bin_acc": 0.0,
        }
        num_steps = 0

        for step, batch in enumerate(loader, start=1):
            batch = self._move_batch_to_device(batch)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(
                    batch["image"],
                    return_aux=True,
                    return_features=False,
                )
                loss_dict = self.compute_losses(outputs, batch)
                loss = loss_dict["loss"]

            self.scaler.scale(loss).backward()

            if self.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            metrics = self._compute_batch_metrics(outputs, batch)

            running["loss"] += loss_dict["loss"].item()
            running["loss_tri"] += loss_dict["loss_tri"].item()
            if "loss_bin" in loss_dict:
                running["loss_bin"] += loss_dict["loss_bin"].item()

            running["tri_acc"] += metrics.get("tri_acc", 0.0)
            running["bin_acc"] += metrics.get("bin_acc", 0.0)

            num_steps += 1

            if step % log_interval == 0:
                print(
                    f"[Train] epoch={epoch} step={step}/{len(loader)} "
                    f"loss={loss_dict['loss'].item():.4f} "
                    f"tri_acc={metrics.get('tri_acc', 0.0):.4f} "
                    f"bin_acc={metrics.get('bin_acc', 0.0):.4f}"
                )

        current_lr = self.optimizer.param_groups[0]["lr"]

        result = {
            "loss": running["loss"] / max(num_steps, 1),
            "loss_tri": running["loss_tri"] / max(num_steps, 1),
            "tri_acc": running["tri_acc"] / max(num_steps, 1),
            "lr": current_lr,
        }

        if running["loss_bin"] > 0:
            result["loss_bin"] = running["loss_bin"] / max(num_steps, 1)
        if running["bin_acc"] > 0:
            result["bin_acc"] = running["bin_acc"] / max(num_steps, 1)

        return result

    @torch.no_grad()
    def evaluate(self, loader, epoch: int = 0):
        self.model.eval()

        running = {
            "loss": 0.0,
            "loss_tri": 0.0,
            "loss_bin": 0.0,
            "tri_acc": 0.0,
            "bin_acc": 0.0,
        }
        num_steps = 0

        all_tri_probs = []
        all_tri_targets = []

        all_bin_probs = []
        all_bin_targets = []

        for batch in loader:
            batch = self._move_batch_to_device(batch)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(
                    batch["image"],
                    return_aux=True,
                    return_features=False,
                )
                loss_dict = self.compute_losses(outputs, batch)

            metrics = self._compute_batch_metrics(outputs, batch)

            running["loss"] += loss_dict["loss"].item()
            running["loss_tri"] += loss_dict["loss_tri"].item()
            if "loss_bin" in loss_dict:
                running["loss_bin"] += loss_dict["loss_bin"].item()

            running["tri_acc"] += metrics.get("tri_acc", 0.0)
            running["bin_acc"] += metrics.get("bin_acc", 0.0)

            num_steps += 1

            tri_probs = torch.softmax(outputs["logits"], dim=1)
            tri_targets = batch["multi_label"]
            all_tri_probs.append(tri_probs.detach().cpu())
            all_tri_targets.append(tri_targets.detach().cpu())

            if "global_logits" in outputs and "binary_label" in batch:
                g = outputs["global_logits"].squeeze(1)
                bin_probs = torch.sigmoid(g)
                bin_targets = batch["binary_label"]
                all_bin_probs.append(bin_probs.detach().cpu())
                all_bin_targets.append(bin_targets.detach().cpu())

        result = {
            "loss": running["loss"] / max(num_steps, 1),
            "loss_tri": running["loss_tri"] / max(num_steps, 1),
            "tri_acc": running["tri_acc"] / max(num_steps, 1),
        }

        if running["loss_bin"] > 0:
            result["loss_bin"] = running["loss_bin"] / max(num_steps, 1)
        if running["bin_acc"] > 0:
            result["bin_acc"] = running["bin_acc"] / max(num_steps, 1)

        # ===== 这里补 AUC =====
        if len(all_tri_probs) > 0:
            tri_probs = torch.cat(all_tri_probs, dim=0).numpy()
            tri_targets = torch.cat(all_tri_targets, dim=0)
            tri_targets_onehot = F.one_hot(tri_targets, num_classes=tri_probs.shape[1]).numpy()

            try:
                from sklearn.metrics import roc_auc_score
                macro_auc = roc_auc_score(
                    tri_targets_onehot,
                    tri_probs,
                    multi_class="ovr",
                    average="macro",
                )
                result["macro_auc"] = float(macro_auc)
            except ValueError:
                pass

        if len(all_bin_probs) > 0:
            bin_probs = torch.cat(all_bin_probs, dim=0).numpy()
            bin_targets = torch.cat(all_bin_targets, dim=0).numpy()

            try:
                from sklearn.metrics import roc_auc_score
                binary_auc = roc_auc_score(bin_targets, bin_probs)
                result["binary_auc"] = float(binary_auc)
            except ValueError:
                pass

        return result

    @torch.no_grad()
    def predict(self, loader):
        self.model.eval()

        all_results = []
        for batch in loader:
            batch = self._move_batch_to_device(batch)
            outputs = self.model(
                batch["image"],
                return_aux=True,
                return_features=False,
            )

            probs = torch.softmax(outputs["logits"], dim=1)
            preds = probs.argmax(dim=1)

            for i in range(len(batch["sample_id"])):
                item = {
                    "sample_id": batch["sample_id"][i],
                    "image_path": batch["image_path"][i],
                    "pred_class": int(preds[i].item()),
                    "probabilities": probs[i].detach().cpu().tolist(),
                }

                if "global_logits" in outputs:
                    g = outputs["global_logits"]
                    if g.dim() == 2 and g.size(1) == 1:
                        g = g.squeeze(1)
                    item["global_ai_prob"] = float(torch.sigmoid(g[i]).item())

                all_results.append(item)

        return all_results

    def save_checkpoint(self, filename: str, epoch: int, extra: Optional[Dict] = None):
        ckpt = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
        }
        if extra is not None:
            ckpt["extra"] = extra

        save_path = self.save_dir / filename
        torch.save(ckpt, save_path)
        print(f"Checkpoint saved to: {save_path}")

    def load_checkpoint(self, checkpoint_path: str, strict: bool = True):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model"], strict=strict)
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint and self.use_amp:
            self.scaler.load_state_dict(checkpoint["scaler"])

        epoch = checkpoint.get("epoch", 0)
        extra = checkpoint.get("extra", None)

        print(f"Checkpoint loaded from: {checkpoint_path}")
        return epoch, extra

