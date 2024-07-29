import torch
from options.train_options import TrainOptions
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from recommenders import create_recommender
from recommenders.BPRData import BPRData
import torch.utils.data as data
import torch.optim as optim
import os


# 模型训练
def bpr_model_train(model, opt, pried_user):
    # 测试数据
    test_data = []
    for index, row in model.test_data.iterrows():
        user = row[model.user_id]
        item = row[model.item_id]
        test_data.append([user, item])

    # 训练数据
    train_data = []
    for index, row in model.train_data.iterrows():
        user = row[model.user_id]
        item = row[model.item_id]
        train_data.append([user, item])

    train_mat = sp.dok_matrix((model.user_num, model.item_num), dtype=np.float32)
    for x in train_data:
        train_mat[x[0], x[1]] = 1.0

    # 构建数据集
    train_dataset = BPRData(train_data, model.item_num, train_mat, opt.num_ng, opt.use_bpr)
    test_dataset = BPRData(test_data, model.item_num, train_mat, 0, False)
    train_loader = data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
    test_loader = data.DataLoader(test_dataset, batch_size=opt.test_num_ng + 1, shuffle=False, num_workers=0)

    model.cuda()

    if opt.recommender in ['Fusion', 'LightGCN', 'NeuFM', 'FM', 'MF']:
        optimizer = optim.SGD(model.parameters(), lr=opt.lr, weight_decay=opt.lamda)
    else:
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    # 训练
    count, best_hr = 0, 0
    best_precision = 0.0
    for epoch in range(opt.epochs):

        print(f"epoch:{epoch}")

        model.train()
        train_loader.dataset.ng_sample()

        for user, item_i, item_j in train_loader:
            user = user.cuda()
            item_i = item_i.cuda()
            item_j = item_j.cuda()

            model.zero_grad()
            loss = model.backward_model(user, item_i, item_j)
            loss.backward()
            optimizer.step()

            count += 1
        print(f"epoch:{epoch},loss:{loss}")
        # 进行训练的model和不进行测试epochs
        if epoch < opt.no_eval_epochs and epoch+1 != opt.epochs:
            continue

        if (epoch+1)%opt.evalution_interval == 0:
            print(f"eval epoch:{epoch}")
            model.eval()

            recommend_list = model.recommend_users(pried_user)
            precision = model.users_check(recommend_list, model.test_data)
            if precision > best_precision:
                best_precision = precision
                if opt.out:
                    if not os.path.exists(opt.model_path):
                        os.mkdir(opt.model_path)
                    torch.save(model.cpu().state_dict(), '{}net.pth'.format(opt.model_path))
                    model.cuda()

    return None


def model_output_pred(model, opt, pried_user):
    '''
    不划分测试数据则输出预测结果
    '''
    # 训练数据
    train_data = []
    for index, row in model.train_data.iterrows():
        user = row[model.user_id]
        item = row[model.item_id]
        train_data.append([user, item])

    train_mat = sp.dok_matrix((model.user_num, model.item_num), dtype=np.float32)
    for x in train_data:
        train_mat[x[0], x[1]] = 1.0

    # 构建数据集
    train_dataset = BPRData(train_data, model.item_num, train_mat, opt.num_ng, opt.use_bpr)
    train_loader = data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
    model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, weight_decay=opt.lamda)

    for epoch in range(opt.epochs):

        print(f"epoch:{epoch}")

        model.train()
        train_loader.dataset.ng_sample()
        for user, item_i, item_j in train_loader:
            user = user.cuda()
            item_i = item_i.cuda()
            item_j = item_j.cuda()
            model.zero_grad()
            loss = model.backward_model(user, item_i, item_j)
            loss.backward()
            optimizer.step()
        print(f"epoch:{epoch},loss:{loss}")

    recommend_list = model.recommend_users(pried_user)
    model.ouput_topN_recommender(opt, recommend_list)
    return None


def train():
    opt = TrainOptions().parse()  # get training options

    triplet_dataset = pd.read_csv(filepath_or_buffer=opt.dataroot_train, sep=' ', header=None,
                                  names=['user', 'item', 'click_count'])
    pried_user = pd.read_csv(filepath_or_buffer=opt.dataroot_test, sep=' ', header=None, names=['user'])

    # 统计数量

    if opt.divide in ["True", "true"]:
        train_data, test_data = train_test_split(triplet_dataset, test_size=opt.divide_num,
                                                 random_state=opt.random_state)
    else:
        train_data = triplet_dataset.sample(frac=1, random_state=opt.random_state)
        test_data = None

    print(f"pred user:{pried_user.shape[0]}")
    if test_data is None:
        print(f"train dataset:{train_data.shape},test dataset:None")
    else:
        print(f"train dataset:{train_data.shape[0]},test dataset:{test_data.shape[0]},shuffle:{opt.shuffle}")

    # 模型的预测
    if opt.use_model in ['false']:
        # 不需要训练的模型
        recommender = create_recommender(opt)
        recommender.preprocess(train_data, 'user', 'item')
        recommend_list = recommender.recommend_users(pried_user)
        if opt.divide in ["True", "true"]:
            # 数据集划分则进行模型测试
            recommender.users_check(recommend_list, test_data)
        else:
            recommender.ouput_topN_recommender(opt, recommend_list)

    elif opt.use_model in ['true']:
        # 需要训练的模型
        recommender = create_recommender(opt)
        recommender.preprocess(train_data, test_data, 'user', 'item')

        if opt.divide in ["True", "true"]:
            # 数据集划分则进行模型测试
            print(opt.use_bpr)
            if opt.test in ['false']:
                bpr_model_train(recommender, opt, pried_user)
            recommender.load_model()
            recommender.cuda()
            recommend_list = recommender.recommend_users(pried_user)
            recommender.users_check(recommend_list, test_data)
        else:
            # 未划分数据集直接输出预测结果
            model_output_pred(recommender, opt, pried_user)

if __name__ == '__main__':
    train()
