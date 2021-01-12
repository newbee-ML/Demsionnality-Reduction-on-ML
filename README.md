# Demsionnality-Reduction-on-ML
数据降维在非监督学习与监督学习中的表现

### 数据集

```
1 'glass': r'..\data\glass.npy',
2 'leaf': r'..\data\leaf.npy',
3 'spambase': r'..\data\spambase.npy',
4 'wdbc': r'..\data\wdbc.npy',
5 'wine': r'..\data\wine.npy'
```

共5个数据集

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



### 调用方法

在dim_reduct.py中

```python
if __name__ == '__main__':
    data_name = 'wine'  # 数据集
    method_now = 'LDA'  # 降维方法
    out_dim_num = 2		# 输出维度
	
    # done_data为降维后的数据
    done_data = red_data(data_name, method_now, out_dim_num)

```


