import numpy as np
from classifier import *
from plot_tools import *
import os
import pandas as pd
# x=np.load('../reductfile/glass/Isomap/result.npy',allow_pickle=True)
# print(x)


# path= '../datafile/avila/Isomap/data_2.npy'
# path='../datafile/credit_card.npy'
# X=np.load(path,allow_pickle=True)
# print(X)
# print(my_network(X))




# def plot():
#     # funcname = ['my_kmeans','my_meanshift','my_dbscan','my_spectral','my_hie']
#     num=['or','2','3','5','10']
#     path = '../datafile/'
#     for i in os.listdir(path):
#         path1=os.path.join(path,i)
#         if os.path.isdir(path1):
#             if i =='spambase':
#                 funcname = ['my_kmeans','my_meanshift','my_dbscan','my_hie']
#             else:
#                 funcname = ['my_kmeans', 'my_meanshift', 'my_dbscan', 'my_spectral', 'my_hie']
#             for j in os.listdir(path1):
#                 if j != 'KPCA':
#                     path2=os.path.join(path1,j)
#                     print(path2)
#                     plot_unsupervised(path2,funcname,num)


reductname=['Isomap','KPCA','LDA','LE','LLE','MDS','PCA','T-SNE']
dataname=['avila','creditcard','glass','wdbc','wine','spambase']
funcname=['ababoost','bp','nb','rfc','svm']
funcname_num={
    'ababoost':0,
    'bp':1,
    'nb':2,
    'rfc':3,
    'svm':4
}

# import csv
# alldict={}
# for i in dataname:
#     path1='../supervised/图片/' + i
#     os.makedirs(path1)
#     for j in reductname:
#         path2='../supervised/图片/'+i+'/'+j
#         os.makedirs(path2)
#         dict={}
#         with open('../supervised/表格/totalacc.csv', 'r') as f:
#             reader = csv.reader(f)
#             # print(type(reader))
#             for row in reader:
#                 # print(row[1])
#                 if i in row[0] and j in row[0]:
#                     dict[(row[0]).split('.')[0]]=row[1:]
#         np.save('../supervised/图片/'+i+'/'+j+'/'+'supervised.npy',dict)

# from plot_tools import *
# # X=np.load('../supervised/图片/avila/Isomap/supervised.npy',allow_pickle=True)
# # print(X)
# path=r'..\supervised\图片'
# for i in os.listdir(path):
#     path1=os.path.join(path,i)
#     for j in os.listdir(path1):
#         path2=os.path.join(path1,j)
#         print(path2)
#         plot_supervised(path2,funcname,funcname_num,num)


my_func_unsupervised=['my_kmeans','my_meanshift','my_dbscan','my_spectral','my_hie']
num=['or','10','5','3','2']
data=np.load('../datafile/wine/LDA/unsupervised.npy',allow_pickle=True).item()
print(data)
dict={}
for j in num:
    list=[]
    for i in my_func_unsupervised:
        list.append(round(data[i+'_'+j][1],3))
    dict[j]=list
print(dict)

df = pd.DataFrame(dict,index=['Kmeans','Meanshift','DBSCAN','Spectral clustering','Hierarchical clustering'])
# print(df)

writer = pd.ExcelWriter('../unsupervised/wine_LDA.xlsx',index=False)

df.to_excel(writer, sheet_name='Sheet1')
writer.save()



