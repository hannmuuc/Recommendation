import torch
import torch.nn as nn
import pandas
import itertools
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from recommenders.LightGCN_recommender import LightGCNEncoder


# SelfCFEncoder类参考自https://github.com/Coder-Yu/QRec代码
class SelfCFEncoder(nn.Module):
    def __init__(self, norm_adj,user_num,item_num, emb_size,momentum, n_layers):
        super(SelfCFEncoder, self).__init__()
        self.user_count = user_num
        self.item_count = item_num
        self.latent_size = emb_size
        self.momentum = momentum
        self.online_encoder = LightGCNEncoder(norm_adj,user_num,item_num, emb_size, n_layers)
        self.predictor = nn.Linear(self.latent_size, self.latent_size)
        self.u_target_his = torch.randn((self.user_count, self.latent_size), requires_grad=False).cuda()
        self.i_target_his = torch.randn((self.item_count, self.latent_size), requires_grad=False).cuda()

    def forward(self, users, items):
        u_online, i_online = self.online_encoder()

        with torch.no_grad():
            u_target, i_target = self.u_target_his.clone()[users], self.i_target_his.clone()[items]
            u_target.detach()
            i_target.detach()
            # 使用Historical embedding来进行对比学习
            u_target = u_target * self.momentum + u_online[users].data * (1. - self.momentum)
            i_target = i_target * self.momentum + i_online[items].data * (1. - self.momentum)
            # 将结果保存下来
            self.u_target_his[users, :] = u_online[users].clone()
            self.i_target_his[items, :] = i_online[items].clone()
        return self.predictor(u_online[users]), u_target, self.predictor(i_online[items]), i_target

        # return u_online[users], u_target, i_online[items], i_target

    @torch.no_grad()
    def get_embedding(self):
        u_online, i_online = self.online_encoder.forward()
        return self.predictor(u_online), u_online, self.predictor(i_online), i_online

    def loss_fn(self, p, z):

        return 1 - F.cosine_similarity(p, z.detach(), dim=-1).mean()

    def get_loss(self, output):
        u_online, u_target, i_online, i_target = output
        loss_ui = self.loss_fn(u_online, i_target) / 2
        loss_iu = self.loss_fn(i_online, u_target) / 2
        return loss_ui + loss_iu


class SelfCFRecommender(nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--use_model', type=str, default='true', choices='(true)')
        parser.add_argument('--gcn_model', type=str, default='true', choices='(true,false)')
        parser.add_argument('--use_bpr', type=str, default='false', choices='(true,false)')
        parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
        parser.add_argument("--lamda", type=float, default=0.001, help="model regularization rate")
        parser.add_argument("--l2", type=float, default=0.0001, help="model l2 rate")
        parser.add_argument("--momentum",type=float,default=0.05, help="use for Historical embedding")
        parser.add_argument("--batch_size", type=int, default=4096, help="batch size for training")
        parser.add_argument("--epochs", type=int, default=100, help="training epoches")
        parser.add_argument("--no_eval_epochs", type=int, default=0, help="training epoches")
        parser.add_argument("--num_ng", type=int, default=1, help="sample negative items for training")
        parser.add_argument("--test_num_ng", type=int, default=99, help="sample part of negative items for testing")
        parser.add_argument("--top_k", type=int, default=10, help="compute metrics@top_k")
        parser.add_argument("--out", default=True, help="save model or not")
        parser.add_argument("--model_path", type=str, default='./models/LightGCN/')
        parser.add_argument("--gpu", type=str, default="0", help="gpu card ID")
        parser.add_argument("--test", type=str, default='false', choices='(true,false)')
        parser.add_argument("--feat_sizes", type=int, default=2)
        parser.add_argument("--embedding_size", type=int, default=32)
        parser.add_argument("--n_layers", type=str, default="1")

        return parser

    def __init__(self, opt):
        super(SelfCFRecommender, self).__init__()
        self.opt = opt
        self.gpu = opt.gpu
        self.feat_sizes = opt.feat_sizes
        self.embedding_size = opt.embedding_size
        self.n_layers = int(opt.n_layers)
        self.momentum = opt.momentum
        self.device = torch.device('cuda:{}'.format(self.gpu[0])) if self.gpu else torch.device('cpu')
        self.recommend_prior_flag = False
        self.batch_size = opt.batch_size

    def forward(self, user, item):
        output = self.GCNmodel(user, item)
        return output

    def backward_model(self, user, item , item_extra=None):
        # 进行训练的user和item 不使用item_extra只为了方便输出
        output = self.GCNmodel(user, item)
        loss = self.GCNmodel.get_loss(output)
        return loss

    def get_prediction_i(self,all_users_cuda,all_items_cuda):
        p_u_online, u_online, p_i_online, i_online = self.GCNmodel.get_embedding()

        p_users_features = p_u_online[all_users_cuda]
        users_features = u_online[all_users_cuda]
        p_items_features = p_i_online[all_items_cuda]
        items_features = i_online[all_items_cuda]

        score_ui = torch.mm(p_users_features, items_features.T)

        score_iu = torch.mm(users_features, p_items_features.T)

        prediction_i = (score_ui + score_iu).cpu()
        return prediction_i

    def recommend_users(self, users):
        all_users = list(users[self.user_id].unique())
        all_items = self.get_all_items()
        users_position_map = {num: i for i, num in enumerate(all_users)}
        items_position_map = {num: i for i, num in enumerate(all_items)}

        all_users_cuda = torch.IntTensor(all_users).cuda()
        all_items_cuda = torch.IntTensor(all_items).cuda()

        prediction_i = self.get_prediction_i(all_users_cuda,all_items_cuda)

        min_nums = prediction_i.min()

        for index, row in self.train_data.iterrows():
            user = row['user']
            item = row['item']
            if user in users_position_map and item in items_position_map:
                user_index = users_position_map[user]
                item_index = items_position_map[item]
                prediction_i[user_index, item_index] = min_nums - 1

        topk_pred, topk_index = torch.topk(prediction_i, 10, dim=1)
        topk_pred = topk_pred.detach().numpy()

        recommend_list = []
        for i in range(0, topk_pred.shape[0]):
            columns = ['user', 'item', 'score', 'rank']
            df = pandas.DataFrame(columns=columns)
            for j in range(0, topk_pred.shape[1]):
                df.loc[len(df)] = [all_users[i], all_items[topk_index[i][j]], topk_pred[i][j], j + 1]

            recommend_list.append((all_users[i], df))

        return recommend_list

    def users_check(self, user_pred_list, dataset):
        top1_true_pred = 0
        top1_sum_pred = 0
        top5_true_pred = 0
        top5_sum_pred = 0
        top10_true_pred = 0
        top10_sum_pred = 0
        all_pred_num = 0
        for user, df in user_pred_list:
            user_test_data = dataset[dataset[self.user_id] == user]
            user_test_items = set(user_test_data[self.item_id].unique())
            df_item = df[self.item_id].unique()
            all_pred_num = all_pred_num + len(df_item)

            top1_sum_pred = top1_sum_pred + 1
            top5_sum_pred = top5_sum_pred + 5
            top10_sum_pred = top10_sum_pred + 10
            for index, row in df.iterrows():
                item = row['item']
                if item in user_test_items:
                    if index == 0:
                        top1_true_pred = top1_true_pred + 1
                    if index < 5:
                        top5_true_pred = top5_true_pred + 1
                    if index < 10:
                        top10_true_pred = top10_true_pred + 1

        top10_accurate = float(top10_true_pred) / float(top10_sum_pred)
        top1_accurate = float(top1_true_pred) / float(top1_sum_pred)
        top5_accurate = float(top5_true_pred) / float(top5_sum_pred)
        print(f"user num:{len(user_pred_list)},pred item num:{all_pred_num}")
        print(f"top1 pred accurate:{top1_accurate},top1 true pred:{top1_true_pred},top1 sum pred:{top1_sum_pred}")
        print(f"top5 pred accurate:{top5_accurate},top5 true pred:{top5_true_pred},top5 sum pred:{top5_sum_pred}")
        print(f"top10 pred accurate:{top10_accurate},top10 true pred:{top10_true_pred},top10 sum pred:{top10_sum_pred}")
        return top10_accurate

    def preprocess(self, train_data, test_data, user_id, item_id):
        # 初始化数据
        self.test_data = test_data
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

        # 保存索引
        self.all_users = self.get_all_users()
        self.all_items = self.get_all_items()

        self.user_num = max(self.all_users) + 1
        self.item_num = max(self.all_items) + 1

        # 数据类型sp
        self.ui_adj = self.__create_bipartite_adjacency()
        self.norm_adj = self.normalize_graph_mat(self.ui_adj)

        # 构建模型
        self.GCNmodel = SelfCFEncoder(self.norm_adj, self.user_num, self.item_num, self.embedding_size,self.momentum, self.n_layers)

        return None

    def __create_bipartite_adjacency(self, self_connection=False):
        '''
        创建一个邻接矩阵(user_num+item_num,user_num+item_num)
        '''
        n_nodes = self.user_num + self.item_num
        row_idx = self.train_data[self.user_id]
        col_idx = self.train_data[self.item_id]
        user_np = np.array(row_idx)
        item_np = np.array(col_idx)
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.user_num)), shape=(n_nodes, n_nodes),
                                dtype=np.float32)
        adj_mat = tmp_adj + tmp_adj.T
        if self_connection:
            adj_mat += sp.eye(n_nodes)
        return adj_mat

    def normalize_graph_mat(self, adj_mat):
        '''
        对邻接矩阵进行归一化
        '''
        shape = adj_mat.get_shape()
        rowsum = np.array(adj_mat.sum(1))
        if shape[0] == shape[1]:
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
        else:
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_mat = d_mat_inv.dot(adj_mat)
        return norm_adj_mat



    def get_all_items(self):
        all_items = list(self.train_data[self.item_id].unique())
        return all_items

    def get_all_users(self):
        all_user = list(self.train_data[self.user_id].unique())
        return all_user

    def get_user_items(self, user):
        user_data = self.train_data[self.train_data[self.user_id] == user]
        user_items = list(user_data[self.item_id].unique())
        return user_items

    def get_item_users(self, item):
        item_data = self.train_data[self.train_data[self.item_id] == item]
        item_users = list(item_data[self.user_id].unique())
        return item_users

    def load_model(self):
        # 加载预训练的参数
        load_path = self.opt.model_path + "net.pth"
        print(load_path)
        pretrained_state_dict = torch.load(load_path)
        self.load_state_dict(pretrained_state_dict)
        print("预训练参数已加载。")

    # 输出预测
    def ouput_topN_recommender(self,opt, recommend_list):
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

        path = opt.dataroot + "/LGCN_result.txt"
        with open(path, 'w') as file:
            for string in pred_list:
                file.write(string + '\n')
        return None

