from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        # ===== 训练参数配置 =====
        parser.add_argument("--epochs", type=int, default=12, help="总训练轮次，默认15轮")
        parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
        parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减系数，L2正则化")
        parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
        parser.add_argument("--label_smoothing", type=float, default=0.0, help="交叉熵损失的标签平滑系数")
        parser.add_argument("--aux_loss_weight", type=float, default=0.3, help="辅助二分类损失（全局辅助头）在总损失中的权重")
        parser.add_argument("--use_amp", action="store_true", help="是否启用自动混合精度，可加速训练并减少显存占用")
        parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="梯度裁剪的最大L2范数。当梯度的范数超过该值时，会将其缩放到此阈值，防止梯度爆炸")
        parser.add_argument("--log_interval", type=int, default=20, help="每训练多少次打印一次日志")
        parser.add_argument("--save_epoch_freq", type=int, default=1, help="每隔多少个训练轮次保留一个检查点")
        parser.add_argument("--aux_loss_weight_end",type=float,default=0.05,help="辅助二分类损失权重在训练结束时的目标值")
        parser.add_argument("--aux_weight_schedule",type=str,default="cosine_decay",choices=["constant", "cosine_decay"],help="辅助损失权重调度方式")
        parser.add_argument("--scheduler_type",type=str,default="cosine",choices=["constant", "cosine"],help="学习率调度方式：constant表示固定学习率，cosine表示余弦退火")
        parser.add_argument("--min_lr",type=float,default=1e-6,help="余弦退火学习率的最小值")

        # ===== 训练阶段图像处理配置参数 =====
        parser.add_argument("--blur_prob", type=float, default=0.1, help="对输入图像应用高斯模糊增强的概率")
        parser.add_argument("--blur_radius", type=str, default="0.1,1.5", help="高斯模糊半径范围")
        parser.add_argument("--jpg_prob", type=float, default=0.1, help="对图像应用JPEG压缩伪像增强的概率")
        parser.add_argument("--jpg_quality", type=str, default="65,95", help="JPEG 质量参数范围（最小值,最大值）；质量越低，压缩伪像越明显")

        self.isTrain = True
        return parser

    def parse(self, print_options=True):
        opt = super().parse(print_options=print_options)

        # parse range strings
        opt.blur_radius = tuple(float(x) for x in opt.blur_radius.split(","))
        opt.jpg_quality = tuple(int(x) for x in opt.jpg_quality.split(","))

        return opt