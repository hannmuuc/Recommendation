
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

```
python train.py --recommender fusion --divide false
```
--divide 将top10推荐结果输出到data文件夹下
