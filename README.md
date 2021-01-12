# 降维方法

### 数据集

```
1 'avila': r'..\data\avila.npy',
2 'credit_card': r'..\data\credit_card.npy',
3 'glass': r'..\data\glass.npy',
4 'spambase': r'..\data\spambase.npy',
5 'wdbc': r'..\data\wdbc.npy',
6 'wine': r'..\data\wine.npy'
```

共6个数据集

### 降维方法

```
'PCA': 主成分分析(Principal Component Analysis, PCA)
'KPCA': 核主成分分析(Kernal Principal Component Analysis, KPCA)
'LDA': 线性判别分析(Linear Discrimination Analysis, LDA)
'LLE': 局部线性嵌入(Locally Linear Embedding, LLE)
'LE': 拉普拉斯特征映射(Laplacian Eigenmaps, LE)
'MDS': 多尺度变换(Multidimensional scaling, MDS)
'Isomap': 等度量映射(Isometric Mapping, Isomap)
'T-SNE': t分布-随机邻近嵌入(t-distributed stochastic neighbor embedding, t-SNE)
```

### 分类方法

```
无监督算法：
'Kmeans':K均值聚类
'Meanshift':均值漂移
'DBSCAN':密度聚类(Density-Based Spatial Clustering of Applications with Noise, DBSCAN)
'SC':谱聚类(Spectral Clustering, SC)
'HC':层次聚类(Hierarchical Clustering, HC)

有监督算法：
'Adaboost':集成学习算法
'KSVM':核化支撑向量机(Kernel Support Vector Machine, KSVM)
'RF':随机森林(Random Forest, RF)
'NN':神经网络(Neural Network, NN)
'NBM':朴素贝叶斯(Naive Bayes Model, NBM)
```



### 调用方法

数据降维：在dim_reduct.py中

```python
if __name__ == '__main__':
    data_name = 'wine'  # 数据集
    method_now = 'LDA'  # 降维方法
    out_dim_num = 2		# 输出维度
	
    # done_data为降维后的数据
    done_data = red_data(data_name, method_now, out_dim_num)

```

数据分类: 在main.py中

```python
if __name__ == '__main__':
    my_func='my_kmeans' #分类算法
	path=r'..\datafile\avila\Isomap\data_2.npy' #调用数据(data_2表示降到2维)
    X=np.load(path,allow_pickle=True)
    dict['my_kmeans_2']=my_classifier(X,my_func) #调用分类算法
    np.save(savepath, dict) #讲分类结果保存到指定路径savepath
```



### 文件说明

```
数据处理：
data_process.py: 一些数据的处理过程

降维过程：
dim_reduct_function.py:降维算法
dim_reduct.py: 调用降维算法并保存降维后的数据

分类过程：
clssifier.py: 分类算法
main.py (main_supervised、main_unsupervised函数): 调用分类算法并保存结果

结果展示：
plot_tools.py: 降维后数据的可视化，数据分类后结果的展示
main.py (plot、plot_un函数)：调用plot_tools.py中的函数绘图
```


