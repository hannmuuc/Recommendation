from .base_options import BaseOptions
import argparse

class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument('--dataroot_train', type=str, default="training.txt")
        parser.add_argument('--dataroot_test', type=str,default="test.txt")
        parser.add_argument('--shuffle', type=str, default="true")
        parser.add_argument('--random_state', type=int, default="42")
        parser.add_argument('--divide', type=str, default="true")
        parser.add_argument('--divide_num',type=float, default=0.2)
        parser.add_argument('--recommender',type=str, default='Fusion', choices='(Fusion,SelfCF,LightGCN,NeuFM,FM,MF,ItemCf,UserCf,Random)')
        parser.add_argument('--evalution_interval',type=int, default=10)

        self.isTrain = True
        return parser
