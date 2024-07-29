import argparse
import os
import torch
import data
import recommenders

class BaseOptions():


    def __init__(self):
        self.initialized = False

    def initialize(self, parser):

        # basic parameters
        parser.add_argument('--dataroot',type=str,default="./data")
        parser.add_argument('--name', type=str, default='experiment_name')
        parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')

        self.initialized = True
        return parser

    def gather_options(self):

        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        recommender_name = opt.recommender
        recommender_option_setter = recommenders.get_option_setter(recommender_name)
        parser = recommender_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def parse(self):

        opt = self.gather_options()
        opt.isTrain = self.isTrain
        opt.dataroot_train = opt.dataroot+'/'+opt.dataroot_train
        opt.dataroot_test = opt.dataroot + '/' + opt.dataroot_test

        if opt.shuffle in ["True", "true"]:
            opt.random_state = None

        if opt.divide not in ["True", "true"]:
            opt.divide_num = 0


        self.opt = opt
        return self.opt
