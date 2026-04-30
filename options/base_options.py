import argparse
from pathlib import Path
import torch


class BaseOptions:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # ===== 实验相关参数 =====
        parser.add_argument("--name", type=str, default="clipfd_exp", help="实验名，方便保存不同类型的训练文件")
        parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="checkpoint保存的根目录")
        parser.add_argument("--gpu_ids", type=str, default="0", help="gpu ids, e.g. 0 or 0,1 or -1 for cpu")

        # 路径参数
        project_root = Path(__file__).resolve().parents[1] # 解析项目路径，根据当前文件位置反推计算项目文件位置
        dataset_root = project_root / "datasets"
        default_clip_ckpt = project_root / "models" / "parameters" / "ViT-L-14.pt"

        # ===== 数据读取路径参数 =====
        parser.add_argument("--train_image_root",type=str,default=str(dataset_root / "train_images"),help="训练集图片路径")
        parser.add_argument("--train_label_json",type=str,default=str(dataset_root / "train_labels.json"),help="训练集图片标签文件")
        parser.add_argument("--val_image_root",type=str,default=str(dataset_root / "val_images"),help="验证集图片路径")
        parser.add_argument("--val_label_json",type=str,default=str(dataset_root / "val_labels.json"),help="验证集图片标签文件")
        parser.add_argument("--test_image_root",type=str,default=str(dataset_root / "test_images"),help="测试集图片路径")
        parser.add_argument("--test_label_json",type=str,default=str(dataset_root / "test_labels.json"),help="测试集图片标签文件")

        # ===== 数据加载参数配置 =====
        parser.add_argument("--batch_size", type=int, default=16, help="读取图片批次大小")
        parser.add_argument("--num_workers", type=int, default=4, help="在进行数据加载时，使用的读取照片的子进程数量")
        parser.add_argument("--pin_memory", action="store_true", help="用于控制数据加载时的内存锁定功能，提高数据加载效率")
        parser.add_argument("--persistent_workers", action="store_true", help="子进程持久化，不需要每一轮训练都创建")
        # action参数的作用就是它不在命令行出现就默认为False，只要出现了就是True

        # ===== 图片处理参数配置 =====
        parser.add_argument("--load_size", type=int, default=256, help="在图片裁剪之前进行尺寸调整")
        parser.add_argument("--image_size", type=int, default=224, help="输入模型的图片大小，根据选取的主干模型确定")
        parser.add_argument("--no_crop", action="store_true", help="默认允许使用裁剪")
        parser.add_argument("--no_flip", action="store_true", help="默认使用水平翻转")

        # ===== 模型参数配置 =====
        # 模型主体参数
        parser.add_argument("--backbone_name", type=str, default=r"E:\Project\CLIPFD\models\parameters\ViT-L-14.pt", help="模型主体参数配置文件")
        parser.add_argument("--freeze_backbone",action="store_true",default=True,help="冻结主体模型参数不做训练")
        parser.add_argument("--unfreeze_backbone",action="store_false",dest="freeze_backbone",help="不冻结模型主干参数，参与训练的更新")
        parser.add_argument("--use_global_aux_head", action="store_true", help="是否使用全局特征进行二分类")
        parser.add_argument("--final_num_classes", type=int, default=3, help="最后使用融合特征进行分类的种数")
        parser.add_argument("--aux_num_classes", type=int, default=1, help="如果启用全局辅助头，它的输出维度是多少，默认为二分类")

        parser.add_argument("--local_hidden_dim", type=int, default=256, help="局部分支内部的隐藏特征维度")
        parser.add_argument("--local_out_dim", type=int, default=768, help="局部分支在融合前的输出维度")
        parser.add_argument("--local_num_blocks", type=int, default=2, help="局部分支堆叠层数")
        parser.add_argument("--proj_dropout", type=float, default=0.1, help = "局部特征向量到特征图映射的随机失活比例")
        parser.add_argument("--block_dropout", type=float, default=0.0, help="局部block内部使用的dropout比例，默认不使用")
        parser.add_argument("--gn_groups", type=int, default=8, help="把通道分成多少组来做归一化")
        parser.add_argument("--fusion_dropout", type=float, default=0.1, help="融合模块里的dropout比例")
        parser.add_argument("--use_global_adapter", action="store_true", default=True,help="是否在全局特征后增加共享适配层")
        parser.add_argument("--disable_global_adapter", action="store_false", dest="use_global_adapter",help="关闭全局特征适配层")
        parser.add_argument("--global_adapter_dropout", type=float, default=0.1, help="全局适配层中的dropout比例")

        self.initialized = True
        return parser

    # 参数收集方法，没有参数就调用初始化进行参数配置，创建一个参数配置对象
    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        self.parser = parser
        return parser.parse_args()

    # 打印和保存相关的参数配置的方法
    def print_options(self, opt):
        message = "----------------- Options ---------------\n"
        for k, v in sorted(vars(opt).items()):
            default = self.parser.get_default(k)
            comment = "" if v == default else f"\t[default: {default}]" # 这里主要用于打印提示与默认的基础参数配置不同，哪个参数被在命令行指定修改了
            message += f"{str(k):>25}: {str(v):<30}{comment}\n"
        message += "----------------- End -------------------"
        print(message)

        save_dir = Path(opt.checkpoints_dir) / opt.name
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "opt.txt", "w", encoding="utf-8") as f:
            f.write(message + "\n")

    # 参数对象构建的总控方法
    def parse(self, print_options=True):
        opt = self.gather_options() # 读取命令行的输入并将其解析为一个对象
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

if __name__ == '__main__':

    print(Path(__file__).resolve().parents[1])