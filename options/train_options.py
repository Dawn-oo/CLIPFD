from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        # ===== training =====
        parser.add_argument("--epochs", type=int, default=20, help="total training epochs")
        parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
        parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
        parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
        parser.add_argument("--label_smoothing", type=float, default=0.0, help="cross entropy label smoothing")
        parser.add_argument("--aux_loss_weight", type=float, default=0.3, help="weight for auxiliary binary loss")
        parser.add_argument("--use_amp", action="store_true", help="use automatic mixed precision")
        parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="gradient clip norm")
        parser.add_argument("--log_interval", type=int, default=20, help="print log every N steps")
        parser.add_argument("--save_epoch_freq", type=int, default=1, help="save checkpoint every N epochs")

        # ===== train augment =====
        parser.add_argument("--blur_prob", type=float, default=0.1, help="probability of blur augmentation")
        parser.add_argument("--blur_radius", type=str, default="0.1,1.5", help="blur radius range, e.g. 0.1,1.5")
        parser.add_argument("--jpg_prob", type=float, default=0.1, help="probability of jpeg augmentation")
        parser.add_argument("--jpg_quality", type=str, default="65,95", help="jpeg quality range, e.g. 65,95")

        self.isTrain = True
        return parser

    def parse(self, print_options=True):
        opt = super().parse(print_options=print_options)

        # parse range strings
        opt.blur_radius = tuple(float(x) for x in opt.blur_radius.split(","))
        opt.jpg_quality = tuple(int(x) for x in opt.jpg_quality.split(","))

        return opt