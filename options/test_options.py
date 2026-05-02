from pathlib import Path
from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        project_root = Path(__file__).resolve().parents[1]
        default_ckpt = project_root / "pretrained_weights" / "best.pth"

        # ===== 测试相关参数 =====
        parser.add_argument(
            "--ckpt_path",
            type=str,
            default=str(default_ckpt),
            help="测试时加载的模型权重路径"
        )
        parser.add_argument(
            "--save_predictions",
            action="store_true",
            help="是否保存测试集预测结果CSV"
        )
        parser.add_argument(
            "--prediction_csv_name",
            type=str,
            default="test_predictions.csv",
            help="测试预测结果CSV文件名"
        )
        parser.add_argument(
            "--eval_split_name",
            type=str,
            default="test",
            help="评估数据集名称，用于日志与结果文件命名"
        )

        # 测试阶段一般不需要训练增强
        self.isTrain = False
        return parser