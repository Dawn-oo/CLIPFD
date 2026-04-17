from io import BytesIO
from typing import Tuple

import torch
from PIL import Image, ImageFilter
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


# CLIP 官方归一化参数
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def pil_jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    """使用PIL对图片进行JPEG压缩"""
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    out = Image.open(buffer).convert("RGB")
    return out


def random_blur(img: Image.Image,blur_radius: Tuple[float, float] = (0.1, 1.5)) -> Image.Image:
    """随机高斯模糊"""
    low, high = blur_radius
    radius = low + (high - low) * torch.rand(1).item()
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def random_jpeg(img: Image.Image,jpg_quality: Tuple[int, int] = (65, 95)) -> Image.Image:
    """随机JPEG压缩"""
    low, high = jpg_quality
    quality = int(torch.randint(low, high + 1, (1,)).item())
    return pil_jpeg_compress(img, quality=quality)


class TrainImageTransform:
    """
    训练阶段图像增强
    流程：
    resize -> blur/jpeg增强 -> random crop -> flip -> to tensor -> normalize
    """

    def __init__(
        self,
        image_size: int = 224,
        load_size: int = 256,
        no_crop: bool = False,
        no_flip: bool = False,
        blur_prob: float = 0.0,
        blur_radius: Tuple[float, float] = (0.1, 1.5),
        jpg_prob: float = 0.0,
        jpg_quality: Tuple[int, int] = (65, 95),
    ):
        self.image_size = image_size
        self.load_size = load_size
        self.no_crop = no_crop
        self.no_flip = no_flip
        self.blur_prob = blur_prob
        self.blur_radius = blur_radius
        self.jpg_prob = jpg_prob
        self.jpg_quality = jpg_quality

        self.resize = transforms.Resize((load_size, load_size),interpolation=transforms.InterpolationMode.BICUBIC)
        self.crop = (transforms.Lambda(lambda x: x)if no_crop else transforms.RandomCrop(image_size))
        self.flip = (transforms.Lambda(lambda x: x)if no_flip else transforms.RandomHorizontalFlip(p=0.5))
        self.normalize = transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)

    def __call__(self, img: Image.Image):
        img = img.convert("RGB")
        img = self.resize(img)

        if self.blur_prob > 0 and torch.rand(1).item() < self.blur_prob:
            img = random_blur(img, self.blur_radius)

        if self.jpg_prob > 0 and torch.rand(1).item() < self.jpg_prob:
            img = random_jpeg(img, self.jpg_quality)

        img = self.crop(img)
        img = self.flip(img)
        img = TF.to_tensor(img)
        img = self.normalize(img)
        return img


class EvalImageTransform:
    """
    验证 / 测试阶段预处理
    流程：
    resize -> center crop -> to tensor -> normalize
    """

    def __init__(
        self,
        image_size: int = 224,
        load_size: int = 256,
        no_crop: bool = False,
    ):
        self.resize = transforms.Resize(
            (load_size, load_size),
            interpolation=transforms.InterpolationMode.BICUBIC
        )
        self.crop = (
            transforms.Lambda(lambda x: x)
            if no_crop else transforms.CenterCrop(image_size)
        )
        self.normalize = transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)

    def __call__(self, img: Image.Image):
        img = img.convert("RGB")
        img = self.resize(img)
        img = self.crop(img)
        img = TF.to_tensor(img)
        img = self.normalize(img)
        return img