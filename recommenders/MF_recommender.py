import numpy as np
import torch
import torch.nn as nn
import pandas

'''
pred accurate:0.12937365010799137,true pred:1198,sum pred:9260
'''

class evaluate:
    @staticmethod
    def hit(gt_item, pred_items):
        if gt_item in pred_items:
            return 1
        return 0

    @staticmethod
    def ndcg(gt_item, pred_items):
        if gt_item in pred_items:
            index = pred_items.index(gt_item)
            return np.reciprocal(np.log2(index+2))
        return 0

    @staticmethod
    def metrics(model, test_loader, top_k):
        HR, NDCG = [], []

        for user, item_i, item_j in test_loader:
            user = user.cuda()
            item_i = item_i.cuda()
            item_j = item_j.cuda() # not useful when testing
            print(user,item_i)
            prediction_i, prediction_j = model(user, item_i, item_j)
            _, indices = torch.topk(prediction_i, top_k)
            recommends = torch.take(item_i, indices).cpu().numpy().tolist()
            print(recommends)

            gt_item = item_i[0].item()
            print(gt_item)
            HR.append(evaluate.hit(gt_item, recommends))
            NDCG.append(evaluate.ndcg(gt_item, recommends))

        return np.mean(HR), np.mean(NDCG)


class MFRecommender(nn.Module):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--use_model', type=str, default='true', choices='(true)')
        parser.add_argument('--gcn_model', type=str, default='false', choices='(true,false)')
        parser.add_argument('--use_bpr', type=str, default='true', choices='(true,false)')
        parser.add_argument("--lr",type=float,default=0.01, help="learning rate")
        parser.add_argument("--lamda", type=float, default=0.001, help="model regularization rate")
        parser.add_argument("--batch_size", type=int, default=4096,  help="batch size for training")
        parser.add_argument("--epochs", type=int, default=50,  help="training epoches")
        parser.add_argument("--no_eval_epochs", type=int, default=0, help="training epoches")
        parser.add_argument("--top_k", type=int,  default=10,  help="compute metrics@top_k")
        parser.add_argument("--factor_num",type=int,  default=32, help="predictive factors numbers in the model")
        parser.add_argument("--num_ng", type=int, default=4, help="sample negative items for training")
        parser.add_argument("--test_num_ng",  type=int,  default=99, help="sample part of negative items for testing")
        parser.add_argument("--out",  default=True,  help="save model or not")
        parser.add_argument("--model_path",type=str, default='./models/MF/')
        parser.add_argument("--gpu",  type=str, default="0",  help="gpu card ID")
        parser.add_argument("--test", type=str,default='false', choices='(true,false)')

        return parser

    def __init__(self,opt):
        super(MFRecommender, self).__init__()
        self.opt = opt
        self.factor_num = opt.factor_num
        self.model_names = ["user","item"]
        self.gpu = opt.gpu
        self.device = torch.device('cuda:{}'.format(self.gpu[0])) if self.gpu else torch.device('cpu')

    def forward(self, user, item_i, item_j):
        user = self.embed_user(user)
        item_i = self.embed_item(item_i)
        item_j = self.embed_item(item_j)

        prediction_i = (user * item_i).sum(dim=-1)
        prediction_j = (user * item_j).sum(dim=-1)
        return prediction_i, prediction_j

    def backward_model(self, user, item_i, item_j):
        prediction_i, prediction_j = self.forward(user, item_i, item_j)
        loss = - (prediction_i - prediction_j).sigmoid().log().sum()
        return loss

    def recommend_users(self, users):
        all_users = list(users[self.user_id].unique())
        all_items = self.get_all_items()
        users_position_map = {num: i for i, num in enumerate(all_users)}
        items_position_map = {num: i for i, num in enumerate(all_items)}

        all_users_cuda = torch.IntTensor(all_users).cuda()
        all_items_cuda = torch.IntTensor(all_items).cuda()

        users_features = self.embed_user(all_users_cuda).cpu()
        item_features = self.embed_item(all_items_cuda).cpu()

        prediction_i = torch.mm(users_features,item_features.T)
        min_nums = prediction_i.min()


        for index, row in self.train_data.iterrows():
            user = row['user']
            item = row['item']
            if user in users_position_map and item in items_position_map:
                user_index = users_position_map[user]
                item_index = items_position_map[item]
                prediction_i[user_index, item_index] = min_nums-1

        topk_pred,topk_index = torch.topk(prediction_i,10,dim=1)
        topk_pred = topk_pred.detach().numpy()

        recommend_list = []
        for i in range(0,topk_pred.shape[0]):
            columns = ['user', 'item', 'score', 'rank']
            df = pandas.DataFrame(columns=columns)
            for j in range(0,topk_pred.shape[1]):

                df.loc[len(df)] = [all_users[i], all_items[topk_index[i][j]], topk_pred[i][j], j+1]

            recommend_list.append((all_users[i], df))

        return recommend_list

    def users_check(self, user_pred_list, dataset):
        top1_true_pred = 0
        top1_sum_pred = 0
        top5_true_pred = 0
        top5_sum_pred = 0
        top10_true_pred = 0
        top10_sum_pred = 0
        for user, df in user_pred_list:
            user_test_data = dataset[dataset[self.user_id] == user]
            user_test_items = set(user_test_data[self.item_id].unique())
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
        print(f"user num:{len(user_pred_list)}")
        print(f"top1 pred accurate:{top1_accurate},top1 true pred:{top1_true_pred},top1 sum pred:{top1_sum_pred}")
        print(f"top5 pred accurate:{top5_accurate},top5 true pred:{top5_true_pred},top5 sum pred:{top5_sum_pred}")
        print(f"top10 pred accurate:{top10_accurate},top10 true pred:{top10_true_pred},top10 sum pred:{top10_sum_pred}")
        return top10_accurate

    def preprocess(self, train_data,test_data, user_id, item_id):

        # 初始化数据
        self.test_data = test_data
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

        #保存索引
        self.all_users = self.get_all_users()
        self.all_items = self.get_all_items()

        # 矩阵分解
        self.user_num = max(self.all_users)+ 1
        self.item_num = max(self.all_items)+ 1

        self.embed_user = nn.Embedding(self.user_num, self.factor_num)
        self.embed_item = nn.Embedding(self.item_num, self.factor_num)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

        return None

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
        load_path = self.opt.model_path + "net.pth"
        print(load_path)
        pretrained_state_dict = torch.load(load_path)
        self.load_state_dict(pretrained_state_dict)
        print("预训练参数已加载。")

    def ouput_topN_recommender(self, opt, recommend_list):
        pred_list = []
        for user, df in recommend_list:
            user_items = self.get_user_items(user)
            user_pred_str = f"{user}:"
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

        path = opt.dataroot + "/2024110671_result.txt"
        with open(path, 'w') as file:
            for string in pred_list:
                file.write(string + '\n')
        return None

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

        path = opt.dataroot + "/MF_result.txt"
        with open(path, 'w') as file:
            for string in pred_list:
                file.write(string + '\n')
        return None