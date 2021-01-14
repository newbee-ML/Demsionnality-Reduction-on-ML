# 降维方法

### 数据集（6个）

```
1 'avila': r'..\data\avila.npy',
2 'credit_card': r'..\data\credit_card.npy',
3 'glass': r'..\data\glass.npy',
4 'spambase': r'..\data\spambase.npy',
5 'wdbc': r'..\data\wdbc.npy',
6 'wine': r'..\data\wine.npy'
```

### 降维方法（8种）

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

### 分类方法（10种）

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



### 实验结果

#### 降维结果图
以下展示glass数据集在2、3维的低维表示：
* iosmap
![iosmap_dim=2](https://github.com/newbee-ML/Demsionnality-Reduction-on-ML/blob/main/dim_reduction_figures/Glass/Isomap_outdim%3D2.png)
![iosmap_dim=3](https://github.com/newbee-ML/Demsionnality-Reduction-on-ML/blob/main/dim_reduction_figures/Glass/Isomap_outdim%3D3.png)
* LE
![LE_dim=2](https://github.com/newbee-ML/Demsionnality-Reduction-on-ML/blob/main/dim_reduction_figures/Glass/LE_outdim%3D2.png)
![LE_dim=3](https://github.com/newbee-ML/Demsionnality-Reduction-on-ML/blob/main/dim_reduction_figures/Glass/LE_outdim%3D3.png)
* LLE
![LLE_dim=2](https://github.com/newbee-ML/Demsionnality-Reduction-on-ML/blob/main/dim_reduction_figures/Glass/LLE_outdim%3D2.png)
![LLE_dim=3](https://github.com/newbee-ML/Demsionnality-Reduction-on-ML/blob/main/dim_reduction_figures/Glass/LLE_outdim%3D3.png)
* MDS
![MDS_dim=2](https://github.com/newbee-ML/Demsionnality-Reduction-on-ML/blob/main/dim_reduction_figures/Glass/MDS_outdim%3D2.png)
![MDS_dim=3](https://github.com/newbee-ML/Demsionnality-Reduction-on-ML/blob/main/dim_reduction_figures/Glass/MDS_outdim%3D3.png)
* T-SNE
![T-SNE_dim=2](https://github.com/newbee-ML/Demsionnality-Reduction-on-ML/blob/main/dim_reduction_figures/Glass/T-SNE_outdim%3D2.png)
![T-SNE_dim=3](https://github.com/newbee-ML/Demsionnality-Reduction-on-ML/blob/main/dim_reduction_figures/Glass/T-SNE_outdim%3D3.png)

#### 非监督学习结果

总获得240组结果，部分展示如下：

* avila在LDA下：

  |    Clustering Method    | original |  10   |     5     |     3     |     2     |
  | :---------------------: | :------: | :---: | :-------: | :-------: | :-------: |
  |         K-means         |  0.173   | 0.212 | **0.306** |   0.297   |   0.259   |
  |        Meanshift        |  0.095   | 0.087 |   0.067   | **0.239** |   0.237   |
  |         DBSCAN          |  0.004   |   0   |     0     |     0     |   0.004   |
  |   Spectral clustering   |  0.048   | 0.004 |   0.031   |   0.014   | **0.087** |
  | Hierarchical clustering |   0.16   | 0.209 | **0.301** |   0.27    |   0.231   |

* creditcard在T-SNE下;

  |    Clustering Method    | original |    10     |   5   |     3     |     2     |
  | :---------------------: | :------: | :-------: | :---: | :-------: | :-------: |
  |         K-means         |    0     |   0.005   | 0.005 | **0.006** |   0.004   |
  |        Meanshift        |  0.001   | **0.008** | 0.004 |   0.004   |   0.003   |
  |         DBSCAN          |  0.005   |     0     | 0.001 |   0.001   | **0.008** |
  |   Spectral clustering   |    0     | **0.003** |   0   |   0.002   |     0     |
  | Hierarchical clustering |  0.002   |   0.007   | 0.006 | **0.007** |   0.003   |

* glass在LDA下：

  |    Clustering Method    | original |  10   |   5   |     3     |    2     |
  | :---------------------: | :------: | :---: | :---: | :-------: | :------: |
  |         K-means         |  0.737   | 0.757 | 0.747 | **0.765** |  0.699   |
  |        Meanshift        |  0.716   | 0.747 | 0.747 |   0.643   | **0.79** |
  |         DBSCAN          |    0     | 0.468 | 0.468 | **0.601** |  0.431   |
  |   Spectral clustering   |  0.354   | 0.741 | 0.741 | **0.813** |  0.692   |
  | Hierarchical clustering |  0.746   | 0.699 | 0.699 | **0.769** |  0.696   |

* spambase在LE下：

  |    Clustering Method    | original |  10   |   5   |   3   |   2   |
  | :---------------------: | :------: | :---: | :---: | :---: | :---: |
  |         K-means         |   0.04   | 0.025 | 0.07  | 0.093 | 0.073 |
  |        Meanshift        |  0.061   | 0.045 | 0.058 | 0.101 | 0.099 |
  |         DBSCAN          |  0.001   |   0   |   0   |   0   |   0   |
  | Hierarchical clustering |  0.061   | 0.093 | 0.068 | 0.094 | 0.063 |

#### 监督学习结果

实验共得到961组数据，展示部分如下：

|  学习器  | 数据集 | 降维方法 | 降维后维度 | 原始精度 | 降维后精度 |  变化  |
| :------: | :----: | :------: | :--------: | :------: | :--------: | :----: |
|  NBayes  | glass  |    LE    |     10     | 0.395349 |  0.930233  | 135.29 |
|  NBayes  | glass  |    LE    |     5      | 0.395349 |  0.930233  | 135.29 |
|  NBayes  | glass  |    LE    |     2      | 0.395349 |  0.906977  | 129.41 |
|  NBayes  | glass  |    LE    |     3      | 0.395349 |  0.906977  | 129.41 |
|  NBayes  | glass  |   LLE    |     2      | 0.395349 |  0.906977  | 129.41 |
|  NBayes  | glass  |   LLE    |     3      | 0.395349 |  0.906977  | 129.41 |
|  NBayes  | glass  |   MDS    |     10     | 0.395349 |  0.906977  | 129.41 |
|  NBayes  | glass  |   MDS    |     2      | 0.395349 |  0.906977  | 129.41 |
|  NBayes  | glass  |   MDS    |     3      | 0.395349 |  0.906977  | 129.41 |
|  NBayes  | glass  |  T-SNE   |     2      | 0.395349 |  0.906977  | 129.41 |
|  NBayes  | glass  |  Isomap  |     2      | 0.395349 |  0.883721  | 123.53 |
|  NBayes  | glass  |   LLE    |     10     | 0.395349 |  0.883721  | 123.53 |
|  NBayes  | glass  |   MDS    |     5      | 0.395349 |  0.883721  | 123.53 |
|  NBayes  | glass  |   PCA    |     2      | 0.395349 |  0.883721  | 123.53 |
|  NBayes  | glass  |  T-SNE   |     10     | 0.395349 |  0.883721  | 123.53 |
|  NBayes  | glass  |  T-SNE   |     3      | 0.395349 |  0.883721  | 123.53 |
|  NBayes  | glass  |  T-SNE   |     5      | 0.395349 |  0.883721  | 123.53 |
|  NBayes  | glass  |  Isomap  |     3      | 0.395349 |  0.837209  | 111.76 |
|  NBayes  | glass  |   LDA    |     10     | 0.395349 |  0.837209  | 111.76 |
|  NBayes  | glass  |   LDA    |     3      | 0.395349 |  0.837209  | 111.76 |
|  NBayes  | glass  |   LDA    |     5      | 0.395349 |  0.837209  | 111.76 |
|  NBayes  | glass  |   PCA    |     3      | 0.395349 |  0.837209  | 111.76 |
|  NBayes  | glass  |  Isomap  |     5      | 0.395349 |  0.813953  | 105.88 |
|  NBayes  | glass  |   LLE    |     5      | 0.395349 |  0.813953  | 105.88 |
|  NBayes  | glass  |  Isomap  |     10     | 0.395349 |  0.790698  | 100.00 |
|  NBayes  | glass  |   LDA    |     2      | 0.395349 |  0.790698  | 100.00 |
|  NBayes  | glass  |   PCA    |     5      | 0.395349 |  0.767442  | 94.12  |
|  NBayes  | glass  |   PCA    |     10     | 0.395349 |  0.72093   | 82.35  |
|   RFC    | glass  |  Isomap  |     10     | 0.511628 |  0.930233  | 81.82  |
|   RFC    | glass  |   PCA    |     2      | 0.511628 |  0.906977  | 77.27  |
|   RFC    | glass  |   LDA    |     3      | 0.511628 |  0.883721  | 72.73  |
|   RFC    | glass  |   LDA    |     2      | 0.511628 |  0.860465  | 68.18  |
|    BP    | avila  |   PCA    |     10     | 0.460925 |  0.754386  | 63.67  |
|   RFC    | glass  |    LE    |     10     | 0.511628 |  0.837209  | 63.64  |
|   RFC    | glass  |   PCA    |     3      | 0.511628 |  0.837209  | 63.64  |
|   RFC    | glass  |    LE    |     3      | 0.511628 |  0.813953  | 59.09  |
| Adaboost | glass  |  Isomap  |     10     | 0.604651 |  0.953488  | 57.69  |
| Adaboost | glass  |  Isomap  |     5      | 0.604651 |  0.953488  | 57.69  |
| Adaboost | glass  |    LE    |     2      | 0.604651 |  0.953488  | 57.69  |
|    BP    | avila  |   LDA    |     10     | 0.460925 |  0.714514  | 55.02  |
|   RFC    | glass  |  Isomap  |     2      | 0.511628 |  0.790698  | 54.55  |
|   RFC    | glass  |  Isomap  |     3      | 0.511628 |  0.790698  | 54.55  |
|   RFC    | glass  |  Isomap  |     5      | 0.511628 |  0.790698  | 54.55  |
|   RFC    | glass  |   LDA    |     10     | 0.511628 |  0.790698  | 54.55  |
|   RFC    | glass  |   LDA    |     5      | 0.511628 |  0.790698  | 54.55  |
|   RFC    | glass  |    LE    |     5      | 0.511628 |  0.790698  | 54.55  |
|   RFC    | glass  |   LLE    |     10     | 0.511628 |  0.790698  | 54.55  |
|   RFC    | glass  |   PCA    |     5      | 0.511628 |  0.790698  | 54.55  |
|   RFC    | glass  |  T-SNE   |     10     | 0.511628 |  0.790698  | 54.55  |
|   RFC    | glass  |  T-SNE   |     2      | 0.511628 |  0.790698  | 54.55  |
| Adaboost | glass  |  Isomap  |     2      | 0.604651 |  0.930233  | 53.85  |
| Adaboost | glass  |  Isomap  |     3      | 0.604651 |  0.930233  | 53.85  |
| Adaboost | glass  |   LLE    |     3      | 0.604651 |  0.930233  | 53.85  |
| Adaboost | glass  |   MDS    |     2      | 0.604651 |  0.930233  | 53.85  |
| Adaboost | glass  |  T-SNE   |     2      | 0.604651 |  0.930233  | 53.85  |
| Adaboost | glass  |  T-SNE   |     3      | 0.604651 |  0.930233  | 53.85  |
|   SVM    | glass  |  Isomap  |     5      | 0.627907 |  0.953488  | 51.85  |
|   SVM    | glass  |   MDS    |     10     | 0.627907 |  0.953488  | 51.85  |
|   SVM    | glass  |   MDS    |     5      | 0.627907 |  0.953488  | 51.85  |
|   SVM    | glass  |  T-SNE   |     10     | 0.627907 |  0.953488  | 51.85  |
|   SVM    | glass  |  T-SNE   |     2      | 0.627907 |  0.953488  | 51.85  |
|   SVM    | glass  |  T-SNE   |     3      | 0.627907 |  0.953488  | 51.85  |
|   RFC    | glass  |    LE    |     2      | 0.511628 |  0.767442  | 50.00  |
|   RFC    | glass  |   LLE    |     3      | 0.511628 |  0.767442  | 50.00  |
|   RFC    | glass  |   LLE    |     5      | 0.511628 |  0.767442  | 50.00  |
|   RFC    | glass  |   MDS    |     10     | 0.511628 |  0.767442  | 50.00  |
|   RFC    | glass  |   MDS    |     2      | 0.511628 |  0.767442  | 50.00  |
|   RFC    | glass  |   MDS    |     3      | 0.511628 |  0.767442  | 50.00  |
|   RFC    | glass  |  T-SNE   |     3      | 0.511628 |  0.767442  | 50.00  |
| Adaboost | glass  |   LLE    |     2      | 0.604651 |  0.906977  | 50.00  |
| Adaboost | glass  |   LLE    |     5      | 0.604651 |  0.906977  | 50.00  |
| Adaboost | glass  |   MDS    |     10     | 0.604651 |  0.906977  | 50.00  |
| Adaboost | glass  |   MDS    |     5      | 0.604651 |  0.906977  | 50.00  |
| Adaboost | glass  |   PCA    |     10     | 0.604651 |  0.906977  | 50.00  |
| Adaboost | glass  |   PCA    |     2      | 0.604651 |  0.906977  | 50.00  |
| Adaboost | glass  |   PCA    |     3      | 0.604651 |  0.906977  | 50.00  |
| Adaboost | glass  |   PCA    |     5      | 0.604651 |  0.906977  | 50.00  |
|   SVM    | glass  |    LE    |     10     | 0.627907 |  0.930233  | 48.15  |
|   SVM    | glass  |   LLE    |     10     | 0.627907 |  0.930233  | 48.15  |
|   SVM    | glass  |   MDS    |     2      | 0.627907 |  0.930233  | 48.15  |
|   SVM    | glass  |  T-SNE   |     5      | 0.627907 |  0.930233  | 48.15  |
| Adaboost | glass  |   MDS    |     3      | 0.604651 |  0.883721  | 46.15  |
|    BP    | avila  |   MDS    |     10     | 0.460925 |  0.671451  | 45.67  |
|   RFC    | glass  |   MDS    |     5      | 0.511628 |  0.744186  | 45.45  |
|   RFC    | glass  |   PCA    |     10     | 0.511628 |  0.744186  | 45.45  |
|   RFC    | glass  |  T-SNE   |     5      | 0.511628 |  0.744186  | 45.45  |
|   SVM    | glass  |  Isomap  |     10     | 0.627907 |  0.906977  | 44.44  |
|   SVM    | glass  |  Isomap  |     2      | 0.627907 |  0.906977  | 44.44  |
|   SVM    | glass  |   LDA    |     10     | 0.627907 |  0.906977  | 44.44  |
|   SVM    | glass  |   LDA    |     3      | 0.627907 |  0.906977  | 44.44  |
|   SVM    | glass  |   LDA    |     5      | 0.627907 |  0.906977  | 44.44  |
|   SVM    | glass  |    LE    |     2      | 0.627907 |  0.906977  | 44.44  |
|   SVM    | glass  |    LE    |     5      | 0.627907 |  0.906977  | 44.44  |
|   SVM    | glass  |   LLE    |     3      | 0.627907 |  0.906977  | 44.44  |
|   SVM    | glass  |   PCA    |     2      | 0.627907 |  0.906977  | 44.44  |
|   SVM    | glass  |   PCA    |     5      | 0.627907 |  0.906977  | 44.44  |
| Adaboost | glass  |   LDA    |     10     | 0.604651 |  0.860465  | 42.31  |
| Adaboost | glass  |   LDA    |     5      | 0.604651 |  0.860465  | 42.31  |
|   RFC    | glass  |   LLE    |     2      | 0.511628 |  0.72093   | 40.91  |

### 结论

**非监督学习：**

数据在新的流形空间中，数据由完全不可分到可分

**监督学习：**

1）对于强学习器，样本量充足时，数据降维的降噪作用不明显，甚至造成信息丢失，导致分类效果变差；样本量不足时，对于强学习器，数据降维有一定的降噪作用

2）对于弱学习器，数据降维到合适的维度，数据特征的增强与信息的丢失可以达到一个“均衡”状态。



### 讨论

**子空间学习的思想嵌入到机器学习模型中：**

1.聚类方法：映射到低维流形空间

2.监督学习：

 ①“多模态”特征提取

 ② 嵌入损失函数中： 

Loss = Loss(raw data) + Loss(trans data)

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
