import numpy as np
import pandas


'''
KnnNum
1   accurate:0.0820     11   accurate:0.1113
2   accurate:0.0966     12   accurate:0.1146
3   accurate:0.1096     15   accurate:0.1166
4   accurate:0.1112     20   accurate:0.1191
5   accurate:0.1090     30   accurate:0.1179
6   accurate:0.1100   
7   accurate:0.1057  
8   accurate:0.1061
9   accurate:0.1134
10  accurate:0.1110
-1  accurate:0.1178
'''

class ItemCfRecommender:

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.add_argument('--use_model', type=str, default='false', choices='(false)')
        parser.add_argument('--gcn_model', type=str, default='false', choices='(true,false)')
        # parser.add_argument('--sim', type=str, default='jaccard', choices='(jaccard,pearson)')
        parser.add_argument('--numknn',type=int,default='20')

        return parser

    def __init__(self, opt):

        self.opt = opt

    def preprocess(self,train_data,user_id,item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id


        return None

    def recommend_users(self,users):
        all_items = self.get_all_items()
        print(f"unique all items:{len(all_items)}")

        position_map = {num: i for i, num in enumerate(all_items)}

        # 生成矩阵
        cooccurence_matrix = np.matrix(np.zeros(shape=(len(all_items), len(all_items))), float)

        user_list = []
        for i in range(0,len(all_items)):
            item_i_data = self.train_data[self.train_data[self.item_id] == all_items[i]]
            user_list.append(set(item_i_data[self.user_id].unique()))

        for i in range(0, len(all_items)):
            for j in range(0, len(all_items)):
                users_i = user_list[i]
                users_j = user_list[j]

                users_intersection = users_i.intersection(users_j)

                if len(users_intersection) != 0:

                    users_union = users_i.union(users_j)
                    cooccurence_matrix[j, i] = float(len(users_intersection)) / float(len(users_union))
                else:
                    cooccurence_matrix[j, i] = 0

        print(f"item recommend num:{self.opt.numknn}")
        # 生成推荐
        recommend_list = []
        for index,row in users.iterrows():
            user = row['user']
            df = self.generate_user_top_recommendation(user,cooccurence_matrix,position_map,all_items,self.opt.numknn)

            recommend_list.append((user,df))

        return recommend_list


    def generate_user_top_recommendation(self,user,cooccurence_matrix,position_map,all_items,N):
        user_items = self.get_user_items(user)

        columns = ['user', 'item', 'score', 'rank']
        df = pandas.DataFrame(columns=columns)

        index = []
        for item in user_items:
            index.append(position_map[item])

        if len(index) ==0:
            return df

        item_matrix = cooccurence_matrix[index,:]

        sorted_matrix = np.sort(item_matrix, axis=0)[::-1]  # 对矩阵按列降序排序

        # 数量大小
        if N == -1:
            num = len(index)
        else:
            num = min(N,len(index))

        # 取出前 N 大的值 并且 计算平均值
        top_n_values = sorted_matrix[:num]
        average_values = np.mean(top_n_values, axis=0)
        user_sim_scores = np.array(average_values)[0].tolist()

        # 进行排序
        sort_index = sorted(((e, i) for i, e in enumerate(list(user_sim_scores))), reverse=True)

        # top10推荐
        rank = 1
        for i in range(0, len(sort_index)):
            if np.isnan(sort_index[i][0]):
                continue
            if all_items[sort_index[i][1]] not in user_items and rank <= 10:
                df.loc[len(df)] = [user, all_items[sort_index[i][1]], sort_index[i][0], rank]
                rank = rank + 1

        return df


    def recommend(self,user):

        user_items = self.get_user_items(user)
        # print(f"unique items for the user:{len(user_items)}")

        all_items = self.get_all_items()
        # print(f"unique all items:{len(all_items)}")

        cooccurence_matrix = self.construct_cooccurence_matrix(user_items,all_items)

        recommend_item = self.generate_top_recommendation(user,cooccurence_matrix,all_items ,user_items)


        return recommend_item


    def generate_top_recommendation(self,user,cooccurence_matrix, all_items,user_items):
        user_sim_scores = cooccurence_matrix.sum(axis=0)/float(cooccurence_matrix.shape[0])
        user_sim_scores = np.array(user_sim_scores)[0].tolist()


        sort_index = sorted(((e,i) for i,e in enumerate(list(user_sim_scores))), reverse=True)

        columns = ['user','item','score','rank']

        df = pandas.DataFrame(columns=columns)

        rank = 1

        for i in range(0,len(sort_index)):
            if np.isnan(sort_index[i][0]):
                continue
            if all_items[sort_index[i][1]] not in user_items and rank<=10:
                df.loc[len(df)] = [user,all_items[sort_index[i][1]], sort_index[i][0], rank]
                rank = rank+1

        if df.shape[0] ==0:
            print("no recommender item!")

        return df



    def construct_cooccurence_matrix(self,user_items,all_items):

        user_items_users=[]
        for i in range(0,len(user_items)):
            user_items_users.append(self.get_item_users(user_items[i]))


        cooccurence_matrix = np.matrix(np.zeros(shape=(len(user_items), len(all_items))), float)

        for i in range(0,len(all_items)):

            item_i_data = self.train_data[self.train_data[self.item_id] == all_items[i]]
            users_i = set(item_i_data[self.user_id].unique())

            for j in range(0,len(user_items)):

                user_j = user_items_users[j]


                users_intersection = users_i.intersection(user_j)

                if len(users_intersection) != 0 :

                    users_union = users_i.union(user_j)
                    cooccurence_matrix[j, i] = float(len(users_intersection))/float(len(users_union))
                else:
                    cooccurence_matrix[j, i] = 0

        return cooccurence_matrix

    def get_all_items(self):
        all_items = list(self.train_data[self.item_id].unique())
        return all_items

    def get_user_items(self,user):
        user_data = self.train_data[self.train_data[self.user_id]==user]
        user_items = list(user_data[self.item_id].unique())
        return user_items


    def get_item_users(self,item):
        item_data = self.train_data[self.train_data[self.item_id]==item]
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
                    if index <5:
                        top5_true_pred = top5_true_pred +1
                    if index <10:
                        top10_true_pred = top10_true_pred + 1

        top10_accurate = float(top10_true_pred) / float(top10_sum_pred)
        top1_accurate = float(top1_true_pred) / float(top1_sum_pred)
        top5_accurate = float(top5_true_pred) / float(top5_sum_pred)
        print(f"user num:{len(user_pred_list)}")
        print(f"top1 pred accurate:{top1_accurate},top1 true pred:{top1_true_pred},top1 sum pred:{top1_sum_pred}")
        print(f"top5 pred accurate:{top5_accurate},top5 true pred:{top5_true_pred},top5 sum pred:{top5_sum_pred}")
        print(f"top10 pred accurate:{top10_accurate},top10 true pred:{top10_true_pred},top10 sum pred:{top10_sum_pred}")
        return top10_accurate


    def check(self,user,item,dataset):
        user = int(user)
        item = int(item)
        # user_train_data = self.train_data[self.train_data[self.user_id] ==user]
        # users_train = set(user_train_data[self.item_id].unique())
        user_test_data = dataset[dataset[self.user_id] == user]
        user_test_items = list(user_test_data[self.item_id].unique())

        # users_intersection = users_train.intersection(user_test_items)

        # print(len(users_intersection))
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

        path = opt.dataroot + "/ItemCf_result.txt"
        with open(path, 'w') as file:
            for string in pred_list:
                file.write(string + '\n')
        return None
