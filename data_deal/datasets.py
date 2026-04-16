import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile

from utils.enchance import TrainImageTransform, EvalImageTransform


ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def recursively_read_images(
    rootdir: Union[str, Path],
    exts: Optional[Sequence[str]] = None,
) -> List[Path]:
    """
    递归读取目录下所有图片
    """
    rootdir = Path(rootdir)
    if not rootdir.exists():
        raise FileNotFoundError(f"image root does not exist: {rootdir}")

    if exts is None:
        exts = IMAGE_EXTENSIONS
    exts = {e.lower() for e in exts}

    image_paths = []
    for p in rootdir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            image_paths.append(p)

    return sorted(image_paths)


def load_label_index(label_json_path: Union[str, Path]) -> Dict[str, Dict[str, int]]:
    """
    读取固定格式的标签 json，并转成：
    {
        "000001": {"binary_label": 1, "multi_label": 2},
        ...
    }

    当前输入格式要求为：
    {
        "000001": {"是否有ai介入": 1, "具体类别": 2},
        "000002": {"是否有ai介入": 0, "具体类别": 0}
    }
    """
    label_json_path = Path(label_json_path)
    if not label_json_path.exists():
        raise FileNotFoundError(f"label json not found: {label_json_path}")

    with open(label_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError("label json must be a dict like {'000001': {...}, ...}")

    label_index = {}

    for sample_id, item in raw.items():
        if not isinstance(item, dict):
            raise ValueError(f"label info of sample '{sample_id}' must be a dict")

        for key in ["是否有AI介入", "具体类别"]:
            if key not in item:
                raise KeyError(f"missing key '{key}' in label json item: {sample_id} -> {item}")

        sample_id = str(sample_id).strip()
        binary_label = int(item["是否有AI介入"])
        multi_label = int(item["具体类别"])

        if binary_label not in (0, 1):
            raise ValueError(f"binary label must be 0/1, got {binary_label} in sample '{sample_id}'")
        if multi_label not in (0, 1, 2):
            raise ValueError(f"multi label must be 0/1/2, got {multi_label} in sample '{sample_id}'")

        label_index[sample_id] = {
            "binary_label": binary_label,
            "multi_label": multi_label,
        }

    return label_index


class TrainImageJsonDataset(Dataset):
    """
    训练集：
    - 必须有标签
    - 使用训练增强
    """

    def __init__(
        self,
        image_root: Union[str, Path],
        label_json_path: Union[str, Path],
        transform=None,
    ):
        self.image_root = Path(image_root)
        self.image_paths = recursively_read_images(self.image_root)
        if len(self.image_paths) == 0:
            raise RuntimeError(f"no images found in {self.image_root}")

        self.label_index = load_label_index(label_json_path)
        self.transform = transform if transform is not None else TrainImageTransform()
        self.samples = self._build_samples()

    def _build_samples(self) -> List[Dict]:
        samples = []

        for img_path in self.image_paths:
            sample_id = img_path.stem  # 例如 000001.jpg -> 000001

            if sample_id not in self.label_index:
                raise KeyError(f"image id '{sample_id}' not found in label json")

            label_info = self.label_index[sample_id]

            samples.append(
                {
                    "sample_id": sample_id,
                    "image_path": str(img_path),
                    "binary_label": label_info["binary_label"],
                    "multi_label": label_info["multi_label"],
                }
            )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        item = self.samples[index]
        img = Image.open(item["image_path"]).convert("RGB")
        img = self.transform(img)

        return {
            "image": img,
            "binary_label": torch.tensor(item["binary_label"], dtype=torch.float32),
            "multi_label": torch.tensor(item["multi_label"], dtype=torch.long),
            "sample_id": item["sample_id"],
            "image_path": item["image_path"],
        }


class TestImageJsonDataset(Dataset):
    """
    验证 / 测试集：
    - 可带标签，也可不带标签
    - 使用测试预处理，不做训练增强
    """

    def __init__(
        self,
        image_root: Union[str, Path],
        label_json_path: Optional[Union[str, Path]] = None,
        transform=None,
    ):
        self.image_root = Path(image_root)
        self.image_paths = recursively_read_images(self.image_root)
        if len(self.image_paths) == 0:
            raise RuntimeError(f"no images found in {self.image_root}")

        self.label_index = load_label_index(label_json_path) if label_json_path is not None else None
        self.transform = transform if transform is not None else EvalImageTransform()
        self.samples = self._build_samples()

    def _build_samples(self) -> List[Dict]:
        samples = []

        for img_path in self.image_paths:
            sample_id = img_path.stem
            item = {
                "sample_id": sample_id,
                "image_path": str(img_path),
            }

            if self.label_index is not None:
                if sample_id not in self.label_index:
                    raise KeyError(f"image id '{sample_id}' not found in label json")

                label_info = self.label_index[sample_id]
                item["binary_label"] = label_info["binary_label"]
                item["multi_label"] = label_info["multi_label"]

            samples.append(item)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        item = self.samples[index]
        img = Image.open(item["image_path"]).convert("RGB")
        img = self.transform(img)

        output = {
            "image": img,
            "sample_id": item["sample_id"],
            "image_path": item["image_path"],
        }

        if "binary_label" in item:
            output["binary_label"] = torch.tensor(item["binary_label"], dtype=torch.float32)
        if "multi_label" in item:
            output["multi_label"] = torch.tensor(item["multi_label"], dtype=torch.long)

        return output


def build_train_loader(
    image_root: Union[str, Path],
    label_json_path: Union[str, Path],
    batch_size: int,
    image_size: int = 224,
    load_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    no_crop: bool = False,
    no_flip: bool = False,
    blur_prob: float = 0.0,
    blur_radius: Tuple[float, float] = (0.1, 1.5),
    jpg_prob: float = 0.0,
    jpg_quality: Tuple[int, int] = (65, 95),
):
    transform = TrainImageTransform(
        image_size=image_size,
        load_size=load_size,
        no_crop=no_crop,
        no_flip=no_flip,
        blur_prob=blur_prob,
        blur_radius=blur_radius,
        jpg_prob=jpg_prob,
        jpg_quality=jpg_quality,
    )

    dataset = TrainImageJsonDataset(
        image_root=image_root,
        label_json_path=label_json_path,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
    )

    return dataset, loader


def build_test_loader(
    image_root: Union[str, Path],
    batch_size: int,
    label_json_path: Optional[Union[str, Path]] = None,
    image_size: int = 224,
    load_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    no_crop: bool = False,
):
    transform = EvalImageTransform(
        image_size=image_size,
        load_size=load_size,
        no_crop=no_crop,
    )

    dataset = TestImageJsonDataset(
        image_root=image_root,
        label_json_path=label_json_path,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
    )

    return dataset, loader