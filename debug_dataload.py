from data_deal import build_train_loader
import sys
from pathlib import Path
from utils.log import Tee

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_FILE = PROJECT_ROOT / "log.txt"

sys.stdout = Tee(LOG_FILE, sys.stdout)
sys.stderr = Tee(LOG_FILE, sys.stderr)

def main():
    train_image_root = r".\datasets\train_images"
    train_label_json = r".\datasets\train_labels.json"

    dataset, loader = build_train_loader(
        image_root=train_image_root,
        label_json_path=train_label_json,
        batch_size=4,
        image_size=224,
        load_size=256,
        num_workers=0,          # 先用 0，方便排错
        pin_memory=False,
        persistent_workers=False,
        no_crop=False,
        no_flip=False,
        blur_prob=0.1,
        jpg_prob=0.1,
    )

    print("=" * 60)
    print(f"dataset size: {len(dataset)}")
    print(f"loader steps: {len(loader)}")
    print("=" * 60)

    # 看单个样本
    sample = dataset[0]
    print("single sample keys:", sample.keys())
    print("single image shape:", sample["image"].shape)
    print("single image dtype:", sample["image"].dtype)
    print("single binary_label:", sample["binary_label"], sample["binary_label"].dtype)
    print("single multi_label:", sample["multi_label"], sample["multi_label"].dtype)
    print("single sample_id:", sample["sample_id"])
    print("single image_path:", sample["image_path"])
    print("=" * 60)

    # 看一个 batch
    batch = next(iter(loader))
    print("batch keys:", batch.keys())
    print("batch image shape:", batch["image"].shape)
    print("batch image dtype:", batch["image"].dtype)
    print("batch binary_label shape:", batch["binary_label"].shape)
    print("batch binary_label:", batch["binary_label"])
    print("batch multi_label shape:", batch["multi_label"].shape)
    print("batch multi_label:", batch["multi_label"])
    print("batch sample_id:", batch["sample_id"])
    print("batch image_path:", batch["image_path"])

    # 做一些最基础断言
    assert batch["image"].dim() == 4, "image batch should be [B, C, H, W]"
    assert batch["image"].shape[1] == 3, "image channel should be 3"
    assert batch["image"].shape[2] == 224 and batch["image"].shape[3] == 224, "image size should be 224x224"
    assert batch["binary_label"].dim() == 1, "binary_label should be [B]"
    assert batch["multi_label"].dim() == 1, "multi_label should be [B]"

    print("=" * 60)
    print("Dataloader smoke test passed.")


if __name__ == "__main__":
    main()