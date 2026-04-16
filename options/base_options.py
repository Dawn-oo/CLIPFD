import argparse
import os
from pathlib import Path
import torch


class BaseOptions:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # ===== experiment =====
        parser.add_argument("--name", type=str, default="clipfd_exp", help="experiment name")
        parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="checkpoint save root")
        parser.add_argument("--gpu_ids", type=str, default="0", help="gpu ids, e.g. 0 or 0,1 or -1 for cpu")

        # ===== data_deal =====
        parser.add_argument("--train_image_root", type=str, default=None, help="train image folder")
        parser.add_argument("--train_label_json", type=str, default=None, help="train label json")
        parser.add_argument("--val_image_root", type=str, default=None, help="val image folder")
        parser.add_argument("--val_label_json", type=str, default=None, help="val label json")
        parser.add_argument("--test_image_root", type=str, default=None, help="test image folder")
        parser.add_argument("--test_label_json", type=str, default=None, help="test label json")

        # ===== dataloader =====
        parser.add_argument("--batch_size", type=int, default=16, help="batch size")
        parser.add_argument("--num_workers", type=int, default=4, help="dataloader workers")
        parser.add_argument("--pin_memory", action="store_true", help="use pin_memory")
        parser.add_argument("--persistent_workers", action="store_true", help="use persistent_workers")

        # ===== image preprocess =====
        parser.add_argument("--load_size", type=int, default=256, help="resize size before crop")
        parser.add_argument("--image_size", type=int, default=224, help="final image size")
        parser.add_argument("--no_crop", action="store_true", help="disable crop")
        parser.add_argument("--no_flip", action="store_true", help="disable random flip in train")

        # ===== model =====
        parser.add_argument("--backbone_name", type=str, default='../models/parameters/ViT-L-14.pt', help="CLIP backbone name")
        parser.add_argument("--freeze_backbone", action="store_true", help="freeze CLIP backbone")
        parser.add_argument("--use_global_aux_head", action="store_true", help="enable global auxiliary binary head")

        parser.add_argument("--final_num_classes", type=int, default=3, help="final fusion classifier classes")
        parser.add_argument("--aux_num_classes", type=int, default=1, help="global auxiliary head output dim")

        parser.add_argument("--local_hidden_dim", type=int, default=256)
        parser.add_argument("--local_out_dim", type=int, default=768)
        parser.add_argument("--local_num_blocks", type=int, default=2)
        parser.add_argument("--proj_dropout", type=float, default=0.1)
        parser.add_argument("--block_dropout", type=float, default=0.0)
        parser.add_argument("--gn_groups", type=int, default=8)
        parser.add_argument("--fusion_dropout", type=float, default=0.1)

        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            parser = self.initialize(parser)

        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        message = "----------------- Options ---------------\n"
        for k, v in sorted(vars(opt).items()):
            default = self.parser.get_default(k)
            comment = "" if v == default else f"\t[default: {default}]"
            message += f"{str(k):>25}: {str(v):<30}{comment}\n"
        message += "----------------- End -------------------"
        print(message)

        save_dir = Path(opt.checkpoints_dir) / opt.name
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "opt.txt", "w", encoding="utf-8") as f:
            f.write(message + "\n")

    def parse(self, print_options=True):
        opt = self.gather_options()
        opt.isTrain = self.isTrain

        # gpu ids
        str_ids = opt.gpu_ids.split(",")
        opt.gpu_ids = []
        for s in str_ids:
            gid = int(s)
            if gid >= 0:
                opt.gpu_ids.append(gid)

        if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
            torch.cuda.set_device(opt.gpu_ids[0])

        if print_options:
            self.print_options(opt)

        self.opt = opt
        return self.opt
