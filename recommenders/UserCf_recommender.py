import numpy as np
import pandas

'''
score_sum               
10: accurate:0.1088
15: accurate:0.1146
20: accurate:0.1276

num_sum
5:  accurate:0.0897
15: accurate:0.1137

'''
class UserCfRecommender:

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.add_argument('--use_model', type=str, default='false', choices='(false)')
        parser.add_argument('--gcn_model', type=str, default='false', choices='(true,false)')
        parser.add_argument('--sim', type=str, default='cos', choices='(cos)')
        parser.add_argument('--numknn', type=int, default='20')
        parser.add_argument('--criterion', type=str, default='score_sum', choices='(score_sum,score_ave,num_sum)')

        return parser

    def __init__(self, opt):

        self.opt = opt

    def preprocess(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

        return None

    def recommend_users(self, users):
        # 生成推荐

        all_users = self.get_all_users()
        print(f"unique all users:{len(all_users)}")
        all_items = self.get_all_items()
        print(f"unique all items:{len(all_items)}")

        users_position_map = {num: i for i, num in enumerate(all_users)}
        items_position_map = {num: i for i, num in enumerate(all_items)}

        cooccurence = self.construct_cooccurence_matrix(all_users,all_items,users_position_map,items_position_map)

        print(f"item recommend num:{self.opt.numknn}")

        # 生成推荐
        recommend_list = []
        for index, row in users.iterrows():
            user = row['user']
            df = self.generate_user_top_recommendation(user,cooccurence,users_position_map,all_users,self.opt.numknn)
            recommend_list.append((user, df))

        return recommend_list

    def construct_cooccurence_matrix(self, all_users, all_items,users_position_map,items_position_map):
        connect = np.zeros((len(all_users),len(all_items)))

        for index,row in self.train_data.iterrows():
            user = row['user']
            item = row['item']
            user_index = users_position_map[user]
            item_index = items_position_map[item]
            connect[user_index,item_index] = 1.0

        row_sums = np.sqrt(connect.sum(axis=1)).reshape(-1,1)
        cooccurence_sum = np.dot(row_sums,row_sums.T)
        cooccurence = np.dot(connect, connect.T)

        cooccurence = cooccurence/(cooccurence_sum + 1e-6)

        return cooccurence


    def recommend(self, user):
        all_users = self.get_all_users()
        print(f"unique all users:{len(all_users)}")
        all_items = self.get_all_items()
        print(f"unique all items:{len(all_items)}")

        users_position_map = {num: i for i, num in enumerate(all_users)}
        items_position_map = {num: i for i, num in enumerate(all_items)}

        cooccurence = self.construct_cooccurence_matrix(all_users, all_items, users_position_map, items_position_map)

        print(f"item recommend num:{self.opt.numknn}")
        df = self.generate_user_top_recommendation(user, cooccurence, users_position_map, all_users, self.opt.numknn)

        return df

    def generate_user_top_recommendation(self,user,cooccurence_matrix,position_map,all_users,N):

        columns = ['user', 'item', 'score', 'rank']
        df = pandas.DataFrame(columns=columns)
        # 对user进行聚类
        index = position_map[user]
        item_vector = cooccurence_matrix[index,:]
        sorted_indices = np.argsort(item_vector)[::-1]

        # 对聚类之后的商品进行统计
        user_items = self.get_user_items(user)
        items_list = {}
        num = 1
        for i in range(0, len(sorted_indices)):
            index = sorted_indices[i]
            user_sim = item_vector[index]
            check_user = all_users[index]

            if np.isnan(user_sim):
                continue
            if check_user == user:
                continue

            num = num + 1
            check_user_items = self.get_user_items(check_user)
            # 对商品进行检查
            for check_user_item in check_user_items:
                if check_user_item in user_items:
                    continue
                if check_user_item in items_list:
                    sim_sum,num_sum = items_list[check_user_item]
                    items_list[check_user_item] = (sim_sum+user_sim, num_sum+1)
                else:
                    items_list[check_user_item] = (user_sim, 1)

            if num <= N or N == -1:
                continue

            if len(items_list) >= 10:
                break

        # 进行评分
        user_item_scores = []
        user_item_index = []

        for key, value in items_list.items():
            sim_scores,sim_num = value
            user_item_index.append(key)
            if self.opt.criterion in ['score_sum']:
                user_item_scores.append(sim_scores)
            elif self.opt.criterion in ['score_ave']:
                user_item_scores.append(sim_scores/sim_num)
            elif self.opt.criterion in ['num_sum']:
                user_item_scores.append(sim_num)

        # 排序
        user_item_sorted_index = np.argsort(user_item_scores)[::-1]

        rank = 1
        for i in range(0,len(user_item_sorted_index)):
            if rank<= 10:
                tool_index = user_item_sorted_index[i]
                df.loc[len(df)] = [user,user_item_index[tool_index], user_item_scores[tool_index], rank]
                rank = rank + 1

        return df

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

    def check(self, user, item, dataset):
        user = int(user)
        item = int(item)
        user_train_data = self.train_data[self.train_data[self.user_id] ==user]
        users_train_items = set(user_train_data[self.item_id].unique())
        user_test_data = dataset[dataset[self.user_id] == user]
        user_test_items = list(user_test_data[self.item_id].unique())

        if item in users_train_items:
            print("recommender wrong")

        if item in user_test_items:
            return True
        else:
            return False

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

        path = opt.dataroot + "/UserCf_result.txt"
        with open(path, 'w') as file:
            for string in pred_list:
                file.write(string + '\n')
        return None