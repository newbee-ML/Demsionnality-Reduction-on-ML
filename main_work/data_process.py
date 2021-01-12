import os
import numpy as np

# path='../data/'
#
# X=np.load(os.path.join(path,'avila.npy'))
# y=X[:,0]
# num=list(set(y))
# print(type(num))
# for j in range(len(num)):
#     print(num[j])
#     X[X[:,0]==num[j],0]=j
# np.save('../datafile/'+'avila.npy',X)
# path='../reductfile/glass.npy'
# X=np.load(path)
# print(X[:,0])

# path='../data/avila.npy'
# X=np.load(path)
# print(sum(X[:,0]==1))
# length=X.shape[0]
# print(length)
# a=np.random.choice(range(length),5000)
# new_X=X[a,:]
# print(sum(new_X[:,0]==1))
# print(new_X[new_X[:,0]==1,:])
# print(set(new_X[:,0]))
# print(X.shape[1])
# newX=np.empty(shape=(0,X.shape[1]))
# for i in range(len(set(X[:,0]))):
#     if sum(X[:,0]==i)>200:
#         a=X[X[:,0]==i,:][:200,:]
#         newX = np.vstack((newX, a))
#     elif sum(X[:,0]==i)>50:
#         a=X[X[:,0]==i,:]
#         newX=np.vstack((newX,a))
#
#
# y=newX[:,0]
# num=list(set(y))
# for j in range(len(num)):
#     newX[newX[:,0]==num[j],0]=j
# print(newX)
# np.save('../datafile/avila.npy',newX)

# path='../datafile/'
# total_dict={}
# for i in os.listdir(path):
#     path1=os.path.join(path,i)
#     if os.path.isdir(path1):
#         for j in os.listdir(path1):
#             if j !='KPCA':
#                 path2=os.path.join(path1,j)
#                 dict=np.load(path2+'/'+'unsupervised.npy',allow_pickle=True).item()
#                 for a in dict:
#                     total_dict[i+'_'+j+'_'+a]=dict[a]
# print(np.load('../datafile/result.npy',allow_pickle=True))


import pandas as pd


def export_excel(export):
   #将字典列表转换为DataFrame
   pf = pd.DataFrame(list(export))
   #指定字段顺序
   order = ['dataname','dim']
   pf = pf[order]
   #将列名替换为中文
   columns_map = {
      'dataname':'数据',
      'dim':'维度',
   }
   pf.rename(columns = columns_map,inplace = True)
   #指定生成的Excel表格名称
   file_path = pd.ExcelWriter('name.xlsx')
   #替换空单元格
   pf.fillna(' ',inplace = True)
   #输出
   pf.to_excel(file_path,encoding = 'utf-8',index = False)
   #保存表格
   file_path.save()


X=np.load('../datafile/glass/LDA/unsupervised.npy',allow_pickle=True).item()
print(X)

dict={}
for i in X:
   dict[i]=X[i][4]
print(dict)



funclist = ['my_kmeans', 'my_meanshift', 'my_dbscan', 'my_spectral', 'my_hie']
numlist = ['or','2','3','5','10']

newdict={}
newdict['func']=['Kmeans','Meanshift','DBSCAN','Spectral Clustering','Hierarchical Clustering']
for i in numlist:
   list=[]
   for j in funclist:
      list.append(round(dict[j+'_'+i],3))
   newdict[i]=list


df=pd.DataFrame(newdict)
# df.style.set_caption("data:avila  reduct:LDA")

def highlight_max(s):
   '''
   highlight the maximum in a Series yellow.
   '''
   is_max = s == s.max()
   return ['background-color: yellow' if v else '' for v in is_max]
df.style.highlight_max(axis=0)

df.to_excel('../unsupervised/glass_LDA.xlsx',index=False)
stu = pd.read_excel('../unsupervised/glass_LDA.xlsx',index_col=0)
print(stu)