实现了常见的推荐系统算法，如:LightGCN、SelfCF、ItemCf等

<h2>Usage</h2>

```
python train.py --recommender fusion 
```

--recommender 实现了不同的推荐算法

'Fusion': ItemCF和LightGCN混合推荐算法  
'LightGCN': LightGCN推荐算法  
'SelfCF': 'SelfCF'推荐算法  
'NeuFM': Neural Factorization Machines 推荐算法  
'FM': Factorization Machine 推荐算法  
'MF': 矩阵分解协同过滤算法  
'ItemCf': 基于商品的协同过滤算法   
'UserCf': 基于用户的协同过滤算法   
'Random': 随机进行商品推荐  

--dataroot 数据集的路径 默认在./data路径下  

```
python train.py --recommender fusion --divide false
```
--divide 将top10推荐结果输出到data文件夹下

<h2>dataset</h2>

./data/training.txt   
训练集（training.txt）包含了用于模型训练/计算的数据，内含 942 个用户，1412 个商品项目，44,234 条点击信息.   
文件内每一行包含三个字段，分别为：user_id, item_id, click

./data/test.txt  
测试集 （test.txt）包含了需要预测 top-10 推荐列表的 926 个用户，每个用户均在训练集中出现过.  
文件内每一行分别代表一个用户。

