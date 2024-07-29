import pandas
import random

class RandomRecommender:

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.add_argument('--use_model', type=str, default='false', choices='(false)')
        parser.add_argument('--gcn_model', type=str, default='false', choices='(true,false)')

        return parser

    def __init__(self, opt):

        self.opt = opt

    def preprocess(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

        return None

    def recommend_users(self,users):
        # 生成推荐
        recommend_list = []
        for index, row in users.iterrows():
            user = row['user']
            df = self.recommend(user)
            recommend_list.append((user, df))

        return recommend_list

    def recommend(self, user):
        user_items = self.get_user_items(user)
        # print(f"unique items for the user:{len(user_items)}")

        all_items = self.get_all_items()
        # print(f"unique all items:{len(all_items)}")

        set1 = set(user_items)

        sub = [element for element in all_items if element not in set1]
        recommend_item = random.sample(sub, 10)

        columns = ['user', 'item', 'score', 'rank']
        df = pandas.DataFrame(columns=columns)

        score = 1.0 / float(len(sub))
        rank = 1

        for element in recommend_item:
            df.loc[len(df)] = [user, element , score, rank]
            rank = rank+1

        return df


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

        true_pred = 0
        sum_pred = 0
        for user, df in user_pred_list:
            user_test_data = dataset[dataset[self.user_id] == user]
            user_test_items = set(user_test_data[self.item_id].unique())
            for index, row in df.iterrows():
                item = row['item']
                if item in user_test_items:
                    true_pred = true_pred + 1
                sum_pred = sum_pred + 1

        accurate = float(true_pred) / float(sum_pred)
        print(f"user num:{len(user_pred_list)}")
        print(f"pred accurate:{accurate},true pred:{true_pred},sum pred:{sum_pred}")
        return accurate

    def check(self, user, item, dataset):
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

        path = opt.dataroot + "/Random_result.txt"
        with open(path, 'w') as file:
            for string in pred_list:
                file.write(string + '\n')
        return None