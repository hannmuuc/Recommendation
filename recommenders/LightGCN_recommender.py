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

def l2_reg_loss(reg, *args):
    emb_loss = 0
    for emb in args:
        emb_loss += torch.norm(emb, p=2)/emb.shape[0]
    return emb_loss * reg

# 图神经网络 LightGCNEncoder://github.com/Coder-Yu/QRec代码
class LightGCNEncoder(nn.Module):
    def __init__(self, norm_adj,user_num,item_num, emb_size, n_layers):
        super(LightGCNEncoder, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = self.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.item_num, self.latent_size))),
        })
        return embedding_dict

    def convert_sparse_mat_to_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor(np.array([coo.row, coo.col]))
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings = all_embeddings[:self.user_num]
        item_all_embeddings = all_embeddings[self.user_num:]
        return user_all_embeddings, item_all_embeddings




class LightGCNRecommender(nn.Module):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--use_model', type=str, default='true', choices='(true)')
        parser.add_argument('--gcn_model', type=str, default='true', choices='(true,false)')
        parser.add_argument('--use_bpr', type=str, default='true', choices='(true,false)')
        parser.add_argument("--lr",type=float,default=0.01, help="learning rate")
        parser.add_argument("--lamda", type=float, default=0.001, help="model regularization rate")
        parser.add_argument("--l2", type=float, default=0.0001, help="model l2 rate")
        parser.add_argument("--batch_size", type=int, default=4096,  help="batch size for training")
        parser.add_argument("--epochs", type=int, default=100,  help="training epoches")
        parser.add_argument("--no_eval_epochs", type=int, default=0, help="training epoches")
        parser.add_argument("--top_k", type=int,  default=10,  help="compute metrics@top_k")
        parser.add_argument("--num_ng", type=int, default=4, help="sample negative items for training")
        parser.add_argument("--test_num_ng",  type=int,  default=99, help="sample part of negative items for testing")
        parser.add_argument("--out",  default=True,  help="save model or not")
        parser.add_argument("--model_path",type=str, default='./models/LightGCN/')
        parser.add_argument("--gpu",  type=str, default="0",  help="gpu card ID")
        parser.add_argument("--test", type=str,default='false', choices='(true,false)')
        parser.add_argument("--feat_sizes", type=int, default=2)
        parser.add_argument("--embedding_size", type=int, default=32)
        parser.add_argument("--n_layers",type=str, default="1")

        return parser

    def __init__(self,opt):
        super(LightGCNRecommender, self).__init__()
        self.opt = opt
        self.gpu = opt.gpu
        self.feat_sizes = opt.feat_sizes
        self.embedding_size = opt.embedding_size
        self.n_layers = int(opt.n_layers)
        self.device = torch.device('cuda:{}'.format(self.gpu[0])) if self.gpu else torch.device('cpu')
        self.recommend_prior_flag = False
        self.batch_size = opt.batch_size

    def forward(self, user, item_i, item_j):
        all_user_emb, all_item_emb = self.GCNmodel()
        user_emb, pos_item_emb, neg_item_emb = all_user_emb[user], all_item_emb[item_i], all_item_emb[item_j]

        return user_emb, pos_item_emb,neg_item_emb

    def backward_model(self,user,item_i,item_j):
        user_emb, pos_item_emb,neg_item_emb = self.forward(user,item_i,item_j)
        prediction_i = (user_emb * pos_item_emb).sum(dim=-1)
        prediction_j = (user_emb * neg_item_emb).sum(dim=-1)
        loss = - (prediction_i - prediction_j).sigmoid().log().sum()
        # bprloss = - (prediction_i - prediction_j).sigmoid().log().mean()
        # l2loss = l2_reg_loss(self.opt.l2, user_emb, pos_item_emb, neg_item_emb) / self.batch_size
        # loss = bprloss + l2loss
        return loss

    def recommend_users(self, users):
        all_users = list(users[self.user_id].unique())
        all_items = self.get_all_items()
        users_position_map = {num: i for i, num in enumerate(all_users)}
        items_position_map = {num: i for i, num in enumerate(all_items)}

        all_users_cuda = torch.IntTensor(all_users).cuda()
        all_items_cuda = torch.IntTensor(all_items).cuda()

        all_user_emb, all_item_emb = self.GCNmodel()

        users_features = all_user_emb[all_users_cuda].cpu()
        item_features = all_item_emb[all_items_cuda].cpu()

        prediction_i = torch.mm(users_features, item_features.T)
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

        #构建模型
        self.GCNmodel = LightGCNEncoder(self.norm_adj,self.user_num,self.item_num, self.embedding_size, self.n_layers)

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

    def normalize_graph_mat(self,adj_mat):
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

