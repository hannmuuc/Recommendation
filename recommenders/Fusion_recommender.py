import torch
import torch.nn as nn
import pandas
import itertools
import scipy.sparse as sp
import numpy as np
from recommenders.BPRData import BPRData
import torch.utils.data as data
import torch.optim as optim
import os
import recommenders

'''
0  top10 pred accurate:0.1365010799136069,top10 true pred:1264,top10 sum pred:9260
1  top10 pred accurate:0.13801295896328294,top10 true pred:1278,top10 sum pred:9260 104
   top10 pred accurate:0.13995680345572353,top10 true pred:1296,top10 sum pred:9260
2  top10 pred accurate:0.13736501079913607,top10 true pred:1272,top10 sum pred:9260 92
3  top10 pred accurate:0.13434125269978403,top10 true pred:1244,top10 sum pred:9260
4 0.131
5 0.132


0:
top10 pred accurate:0.13563714902807775,top10 true pred:1256,top10 sum pred:9260

1: 
80

100
top10 pred accurate:0.1365010799136069,top10 true pred:1264,top10 sum pred:9260
top10 pred accurate:0.133585313174946,top10 true pred:1237,top10 sum pred:9260
top10 pred accurate:0.13120950323974082,top10 true pred:1215,top10 sum pred:9260
top10 pred accurate:0.13228941684665227,top10 true pred:1225,top10 sum pred:9260

0

1
add rating0.0019543802349470518
add rating0.001174735289469531
add rating0.0002844650476742663
add rating0.0011958067744824645
add rating0.001206342516988877
add rating0.0008560290786492396
add rating0.0017305615550756493
add rating0.00021058315334767652
add rating0.0006182505399567661
add rating0.0016646473160196443
2
add rating0.0029832613390928576
add rating0.0007694384449243641
add rating0.003239740820734348
add rating0.0011826370963493728
add rating0.0014144234314912947
add rating0.0012488231710694009
add rating0.002586524785334233
add rating0.0025142603976297286
add rating0.003563714902807824
add rating0.0015803613759680403
3
add rating0.000885002370542015
add rating0.001248485487014744
'''


class FusionRecommender(nn.Module):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--use_model', type=str, default='true', choices='(true)')
        parser.add_argument('--gcn_model', type=str, default='true', choices='(true,false)')
        parser.add_argument('--use_bpr', type=str, default='true', choices='(true,false)')

        # recommender 1 参数
        parser.add_argument('--first_recommender', type=str, default='ItemCf')
        parser.add_argument('--numknn', type=int, default='15')

        # recommender 2 参数
        parser.add_argument('--second_recommender', type=str, default='LightGCN')
        parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
        parser.add_argument("--lamda", type=float, default=0.001, help="model regularization rate")
        parser.add_argument("--l2", type=float, default=0.0001, help="model l2 rate")
        parser.add_argument("--batch_size", type=int, default=4096, help="batch size for training")
        parser.add_argument("--epochs", type=int, default=100, help="training epoches")
        parser.add_argument("--no_eval_epochs", type=int, default=0, help="training epoches")
        parser.add_argument("--top_k", type=int, default=10, help="compute metrics@top_k")
        parser.add_argument("--num_ng", type=int, default=4, help="sample negative items for training")
        parser.add_argument("--test_num_ng", type=int, default=99, help="sample part of negative items for testing")
        parser.add_argument("--out", default=True, help="save model or not")
        parser.add_argument("--model_path", type=str, default='./models/Fusion/')
        parser.add_argument("--gpu", type=str, default="0", help="gpu card ID")
        parser.add_argument("--test", type=str, default='false', choices='(true,false)')
        parser.add_argument("--feat_sizes", type=int, default=2)
        parser.add_argument("--embedding_size", type=int, default=32)
        parser.add_argument("--n_layers", type=str, default="1")

        return parser

    def __init__(self, opt):
        super(FusionRecommender, self).__init__()
        self.opt = opt
        self.topNpred_Second = []
        self.topNpred_fused = []
        # 第一个推荐模型
        first_recommender_class = recommenders.find_recommender_using_name(opt.first_recommender)
        self.FirstRecommender = first_recommender_class(opt)
        self.FirstRecommenderPredFlag = False
        # 第二个推荐模型
        second_recommender_class = recommenders.find_recommender_using_name(opt.second_recommender)
        self.SecondRecommender = second_recommender_class(opt)

    def preprocess(self, train_data, test_data, user_id, item_id):
        # 初始化模型
        self.FirstRecommender.preprocess(train_data, user_id, item_id)
        self.SecondRecommender.preprocess(train_data, test_data, user_id, item_id)

        # 初始化数据
        # 初始化数据
        self.test_data = self.SecondRecommender.test_data
        self.train_data = self.SecondRecommender.train_data
        self.user_id = self.SecondRecommender.user_id
        self.item_id = self.SecondRecommender.item_id
        self.user_num = self.SecondRecommender.user_num
        self.item_num = self.SecondRecommender.item_num

        # 需要进行训练
        self.SecondRecommender.cuda()

        return None

    def forward(self, user, item_i, item_j):
        return self.SecondRecommender(user, item_i, item_j)

    def backward_model(self, user, item_i, item_j):
        return self.SecondRecommender.backward_model(user, item_i, item_j)

    def recommend_users(self, users):
        # 第一个recommend只需要训练一次
        if self.FirstRecommenderPredFlag is False:
            self.FirstRecommenderPred = self.FirstRecommender.recommend_users(users)
            self.FirstRecommenderPredFlag = True

        self.SecondRecommenderPred = self.SecondRecommender.recommend_users(users)

        recommend_list = self.fuse_pred_list(self.FirstRecommenderPred, self.SecondRecommenderPred)

        return recommend_list

    def fuse_pred_list(self, FirstPred, SecondPred):
        '''
        对两种不同推荐结果进行融合
        '''
        fusion_num = 0
        recommend_list = []
        for i in range(0, len(FirstPred)):
            first_user, first_df = FirstPred[i]
            second_user, second_df = SecondPred[i]
            # 判断异常
            if first_user != second_user:
                print("error df!")
                return None
            # 进行融合
            first_partition_num = 2
            fusion_num_flag = 0
            new_df = first_df.head(first_partition_num)
            first_partition_item = set(new_df['item'].unique())
            for index, row in second_df.iterrows():
                if row['item'] in first_partition_item:
                    fusion_num_flag += 1
                    continue
                else:
                    new_df = new_df._append(row, ignore_index=True)
                if len(new_df) >= 10:
                    break
            fusion_num += first_partition_num - fusion_num_flag
            new_df = new_df.sort_values(by=['rank'], ascending=[True])

            # 加入新的推荐列表
            recommend_list.append((first_user, new_df))
        print(f"fusion num:{fusion_num}")
        return recommend_list

    def get_all_items(self):
        return self.SecondRecommender.get_all_items()

    def get_user_items(self, user):
        return self.SecondRecommender.get_user_items(user)

    def get_item_users(self, item):
        return self.SecondRecommender.get_item_users(item)

    def load_model(self):
        # 加载预训练的参数
        load_path = self.opt.model_path + "net.pth"
        print(load_path)
        pretrained_state_dict = torch.load(load_path)
        self.load_state_dict(pretrained_state_dict)
        print("预训练参数已加载。")

    def users_check(self, user_pred_list, dataset):
        # print("----------------recommender----------------")
        # self.topNpred_Second.append(self.SecondRecommender.users_check(self.SecondRecommenderPred,dataset))
        pred_fused = self.SecondRecommender.users_check(user_pred_list, dataset)
        # self.topNpred_fused.append(pred_fused)
        # rating = (sum(self.topNpred_fused)-sum(self.topNpred_Second))/ len(self.topNpred_fused)
        # print(f"add rating{rating}")

        return pred_fused

    def ouput_topN_recommender(self, opt, recommend_list):
        pred_list = []
        for user, df in recommend_list:
            user_items = self.get_user_items(user)
            user_pred_str = f"{user}:"
            if df.shape[0] != 10:
                print("top10 recommender error!")
                return None
            for index, row in df.iterrows():
                item = int(row['item'])
                if item in user_items:
                    print("top10 recommender error!")
                    return None
                if index == 0:
                    user_pred_str = user_pred_str + f"{item}"
                else:
                    user_pred_str = user_pred_str + f",{item}"
            pred_list.append(user_pred_str)

        path = opt.dataroot + "/Fusion_result.txt"
        with open(path, 'w') as file:
            for string in pred_list:
                file.write(string + '\n')
        return None
