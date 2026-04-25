# import os
# from pathlib import Path
# from typing import Optional, Sequence, Union
#
# import numpy as np
# import torch
# from PIL import Image
#
#
# def _ensure_dir(save_dir: Union[str, Path]) -> Path:
#     """
#     确保保存目录存在
#     """
#     save_dir = Path(save_dir)
#     save_dir.mkdir(parents=True, exist_ok=True)
#     return save_dir
#
#
# def _to_numpy_heatmap(heatmap: torch.Tensor) -> np.ndarray:
#     """
#     将单张热图 tensor 转成 [H, W] 的 numpy 数组，并归一化到 [0, 255]
#
#     支持输入:
#     - [1, H, W]
#     - [H, W]
#     """
#     if not isinstance(heatmap, torch.Tensor):
#         raise TypeError(f"heatmap must be torch.Tensor, but got {type(heatmap)}")
#
#     if heatmap.dim() == 3:
#         if heatmap.size(0) != 1:
#             raise ValueError(f"Expected heatmap shape [1,H,W], but got {tuple(heatmap.shape)}")
#         heatmap = heatmap.squeeze(0)
#     elif heatmap.dim() != 2:
#         raise ValueError(f"Expected heatmap shape [H,W] or [1,H,W], but got {tuple(heatmap.shape)}")
#
#     heatmap = heatmap.detach().float().cpu()
#
#     if not torch.isfinite(heatmap).all():
#         raise ValueError("heatmap contains NaN or Inf")
#
#     heatmap = heatmap.numpy()
#
#     # 归一化到 [0, 1]
#     min_v = float(heatmap.min())
#     max_v = float(heatmap.max())
#
#     if max_v - min_v < 1e-8:
#         heatmap = np.zeros_like(heatmap, dtype=np.float32)
#     else:
#         heatmap = (heatmap - min_v) / (max_v - min_v)
#
#     heatmap = (heatmap * 255.0).clip(0, 255).astype(np.uint8)
#     return heatmap
#
#
# def _gray_to_colormap(gray_map: np.ndarray) -> np.ndarray:
#     """
#     将灰度热图映射成伪彩色图。
#     输入:
#     - gray_map: [H, W], uint8
#
#     输出:
#     - color_map: [H, W, 3], uint8
#     """
#     if gray_map.ndim != 2:
#         raise ValueError(f"gray_map must be [H,W], but got {gray_map.shape}")
#
#     x = gray_map.astype(np.float32) / 255.0
#
#     # 简单暖色图：低值偏黑，高值偏红黄
#     r = np.clip(1.8 * x, 0, 1)
#     g = np.clip(1.8 * x - 0.6, 0, 1)
#     b = np.clip(1.2 * x - 1.0, 0, 1)
#
#     color = np.stack([r, g, b], axis=-1)
#     color = (color * 255.0).astype(np.uint8)
#     return color
#
#
# def _to_numpy_image(image_tensor: torch.Tensor) -> np.ndarray:
#     """
#     将输入图 tensor 转成可保存的 RGB 图像。
#
#     支持输入:
#     - [3, H, W]
#     注意:
#     这里默认输入已经是 0~1 或者大致可视化范围。
#     如果你的输入做过 normalize，建议先在外部反归一化再传进来。
#     """
#     if not isinstance(image_tensor, torch.Tensor):
#         raise TypeError(f"image_tensor must be torch.Tensor, but got {type(image_tensor)}")
#
#     if image_tensor.dim() != 3 or image_tensor.size(0) != 3:
#         raise ValueError(f"Expected image tensor shape [3,H,W], but got {tuple(image_tensor.shape)}")
#
#     image = image_tensor.detach().float().cpu()
#
#     if not torch.isfinite(image).all():
#         raise ValueError("image_tensor contains NaN or Inf")
#
#     image = image.permute(1, 2, 0).numpy()  # [H,W,3]
#
#     # 尝试裁剪到可视范围
#     if image.min() < 0 or image.max() > 1:
#         image = image - image.min()
#         max_v = image.max()
#         if max_v > 1e-8:
#             image = image / max_v
#
#     image = (image * 255.0).clip(0, 255).astype(np.uint8)
#     return image
#
#
# def save_local_heatmaps(
#     local_heatmap: torch.Tensor,
#     save_dir: Union[str, Path],
#     max_save: int = 8,
#     file_prefix: str = "heatmap",
#     image_format: str = "png",
#     input_images: Optional[torch.Tensor] = None,
#     overlay: bool = False,
#     alpha: float = 0.45,
#     file_names: Optional[Sequence[str]] = None,
# ) -> list[str]:
#     """
#     保存局部热图到指定目录。
#
#     参数:
#     - local_heatmap:
#         [B, 1, H, W] 或 [B, H, W]
#     - save_dir:
#         保存目录，例如 "outputs/heatmaps"
#     - max_save:
#         最多保存多少张
#     - file_prefix:
#         默认文件名前缀
#     - image_format:
#         "png" / "jpg" / "jpeg"
#     - input_images:
#         可选，[B, 3, H, W]
#         若提供并且 overlay=True，则保存叠加图
#     - overlay:
#         是否叠加到输入图上
#     - alpha:
#         叠加透明度
#     - file_names:
#         可选，自定义每张图的文件名（不含后缀）
#
#     返回:
#     - 保存的文件路径列表
#     """
#     if not isinstance(local_heatmap, torch.Tensor):
#         raise TypeError(f"local_heatmap must be torch.Tensor, but got {type(local_heatmap)}")
#
#     if local_heatmap.dim() == 4:
#         if local_heatmap.size(1) != 1:
#             raise ValueError(f"Expected local_heatmap shape [B,1,H,W], but got {tuple(local_heatmap.shape)}")
#         heatmaps = local_heatmap
#     elif local_heatmap.dim() == 3:
#         heatmaps = local_heatmap.unsqueeze(1)
#     else:
#         raise ValueError(f"Expected local_heatmap shape [B,1,H,W] or [B,H,W], but got {tuple(local_heatmap.shape)}")
#
#     batch_size = heatmaps.size(0)
#
#     if max_save <= 0:
#         return []
#
#     num_save = min(batch_size, max_save)
#
#     fmt = image_format.lower()
#     if fmt == "jpg":
#         fmt = "jpeg"
#     if fmt not in {"png", "jpeg"}:
#         raise ValueError(f"image_format must be 'png' or 'jpg/jpeg', but got {image_format}")
#
#     if input_images is not None:
#         if not isinstance(input_images, torch.Tensor):
#             raise TypeError(f"input_images must be torch.Tensor, but got {type(input_images)}")
#         if input_images.dim() != 4 or input_images.size(0) != batch_size or input_images.size(1) != 3:
#             raise ValueError(
#                 f"Expected input_images shape [B,3,H,W], but got {tuple(input_images.shape)}"
#             )
#
#     if file_names is not None and len(file_names) < num_save:
#         raise ValueError("file_names length must be >= number of saved images")
#
#     save_dir = _ensure_dir(save_dir)
#     saved_paths = []
#
#     for i in range(num_save):
#         gray_map = _to_numpy_heatmap(heatmaps[i])           # [H,W]
#         color_map = _gray_to_colormap(gray_map)             # [H,W,3]
#
#         if overlay:
#             if input_images is None:
#                 raise ValueError("overlay=True requires input_images")
#             image_np = _to_numpy_image(input_images[i])     # [H,W,3]
#
#             if image_np.shape[:2] != color_map.shape[:2]:
#                 raise ValueError(
#                     f"Image size {image_np.shape[:2]} and heatmap size {color_map.shape[:2]} do not match"
#                 )
#
#             blended = ((1 - alpha) * image_np + alpha * color_map).clip(0, 255).astype(np.uint8)
#             out_img = Image.fromarray(blended)
#         else:
#             out_img = Image.fromarray(color_map)
#
#         if file_names is not None:
#             stem = file_names[i]
#         else:
#             stem = f"{file_prefix}_{i:03d}"
#
#         file_path = save_dir / f"{stem}.{fmt}"
#         out_img.save(file_path)
#         saved_paths.append(str(file_path))
#
#     return saved_paths