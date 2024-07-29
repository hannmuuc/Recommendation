import torch
import torch.nn as nn
import pandas
import itertools
'''
pred accurate:0.13563714902807775,true pred:1256,sum pred:9260
first pred accurate:0.1857451403887689,first true pred:172,first sum pred:926
'''
class FMRecommender(nn.Module):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--use_model', type=str, default='true', choices='(true)')
        parser.add_argument('--gcn_model', type=str, default='false', choices='(true,false)')
        parser.add_argument('--use_bpr', type=str, default='true', choices='(true,false)')
        parser.add_argument("--lr",type=float,default=0.005, help="learning rate")
        parser.add_argument("--lamda", type=float, default=0.001, help="model regularization rate")
        parser.add_argument("--batch_size", type=int, default=4096,  help="batch size for training")
        parser.add_argument("--epochs", type=int, default=90,  help="training epoches")
        parser.add_argument("--no_eval_epochs", type=int, default=0,  help="training epoches")
        parser.add_argument("--top_k", type=int,  default=10,  help="compute metrics@top_k")
        parser.add_argument("--num_ng", type=int, default=4, help="sample negative items for training")
        parser.add_argument("--test_num_ng",  type=int,  default=99, help="sample part of negative items for testing")
        parser.add_argument("--out",  default=True,  help="save model or not")
        parser.add_argument("--model_path",type=str, default='./models/FM/')
        parser.add_argument("--gpu",  type=str, default="0",  help="gpu card ID")
        parser.add_argument("--test", type=str,default='false', choices='(true,false)')
        parser.add_argument("--feat_sizes", type=int, default=2)
        parser.add_argument("--embedding_size", type=int, default=16)



        return parser

    def __init__(self,opt):
        super(FMRecommender, self).__init__()
        self.opt = opt
        self.model_names = ["user","item"]
        self.gpu = opt.gpu
        self.feat_sizes = opt.feat_sizes
        self.embedding_size = opt.embedding_size
        self.device = torch.device('cuda:{}'.format(self.gpu[0])) if self.gpu else torch.device('cpu')

        self.recommend_prior_flag = False

    def forward(self, user, item_i, item_j):

        linear_part_user = self.linear_part_user.to(user.device)
        linear_part_item = self.linear_part_item.to(user.device)
        linear_user = linear_part_user[user]
        linear_item_i = linear_part_item[item_i]
        linear_item_j = linear_part_item[item_j]
        linear_i = torch.cat([linear_user,linear_item_i],dim=1)
        linear_j = torch.cat([linear_user,linear_item_j], dim=1)
        linear_i = self.linear(linear_i).squeeze()
        linear_j = self.linear(linear_j).squeeze()

        # 计算交叉
        user = self.embed_user(user)
        item_i = self.embed_item(item_i)
        item_j = self.embed_item(item_j)
        prediction_i = (user * item_i).sum(dim=-1)
        prediction_j = (user * item_j).sum(dim=-1)

        return prediction_i+0.3*linear_i, prediction_j+0.3*linear_j

    def backward_model(self, user, item_i, item_j):
        prediction_i, prediction_j = self.forward(user, item_i, item_j)
        loss = - (prediction_i - prediction_j).sigmoid().log().sum()
        return loss

    def recommend_prior(self,users):
        if self.recommend_prior_flag is True:
            return None

        self.topK_recommender_users = list(users[self.user_id].unique())

        full_connect = list(itertools.product(self.topK_recommender_users, self.all_items))
        self.full_connect_users = torch.IntTensor([user for user, item in full_connect]).reshape(len(self.topK_recommender_users),
                                                                                                 len(self.all_items))
        self.full_connect_items = torch.IntTensor([item for user, item in full_connect]).reshape(len(self.topK_recommender_users),
                                                                                                 len(self.all_items))
        self.recommend_pred_matrix = torch.zeros((len(self.topK_recommender_users),len(self.all_items)))
        self.recommend_prior_flag = True
        return None

    def recommend_one_user(self,index):
        user_i = self.full_connect_users[index].cuda()
        item_i = self.full_connect_items[index].cuda()
        full_connect_pred, _ = self.forward(user_i,item_i,item_i)
        full_connect_pred = torch.squeeze(full_connect_pred.cpu())
        self.recommend_pred_matrix[index] = full_connect_pred
        return None

    def recommend_users(self, users):


        self.recommend_prior(users)

        all_users = self.topK_recommender_users
        all_items = self.all_items
        users_position_map = {num: i for i, num in enumerate(all_users)}
        items_position_map = {num: i for i, num in enumerate(all_items)}

        with torch.no_grad():
            for i in range(len(all_users)):
                self.recommend_one_user(i)

        prediction_i = self.recommend_pred_matrix

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
        self.linear_part_user = torch.eye(self.user_num)
        self.linear_part_item = torch.eye(self.item_num)


        # 模型的偏置
        self.linear = nn.Linear(self.user_num+self.item_num, 1, bias=False)
        nn.init.normal_(self.linear.weight, std=0.01)

        # 模型参数
        self.embed_user = nn.Embedding(self.user_num, self.embedding_size)
        self.embed_item = nn.Embedding(self.item_num, self.embedding_size)
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
        # 加载预训练的参数
        load_path = self.opt.model_path +"net.pth"
        print(load_path)
        pretrained_state_dict = torch.load(load_path)
        self.load_state_dict(pretrained_state_dict)
        print("预训练参数已加载。")

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

        path = opt.dataroot + "/FM_result.txt"
        with open(path, 'w') as file:
            for string in pred_list:
                file.write(string + '\n')
        return None