import numpy as np
from classifier import *
import os
import json
from plot_tools import *
from MyEncoder import MyEncoder




# path='../resultfile/spambase/LLE_2/data_2.npy'
# X=np.load(path)
# print(classifier.my_svm(X))
dataname=['avila','credit_card','glass','wdbc','wine']
original_class_num={
    'glass':6,
    'credit_card':2,
    'spambase':2,
    'wdbc':2,
    'wine':3,
    'avila':11
}

my_func_supervised=['my_Adaboost','my_rfc','my_NBayes','my_svm','my_network']
# my_func_supervised=['my_network']
my_func_unsupervised=['my_kmeans','my_meanshift','my_dbscan','my_spectral','my_hie']

def my_classifier(X, method,**args):
    return eval(method)(X, **args)


def main_unsupervised():
    path='../datafile/'
    # for i in os.listdir(path):
    for i in dataname:
        path1=os.path.join(path,i)
        if os.path.isdir(path1):
            dict1={}
            ordata=np.load('../datafile/'+str(i)+'.npy')
            for m in my_func_unsupervised:
                print(m)
                dict1[m + '_' + 'or'] = my_classifier(ordata, m)
            for j in os.listdir(path1):
                if j!='KPCA':
                    path2=os.path.join(path1,j)
                    savepath = path2 + '/' + 'unsupervised.npy'
                    print(path2)
                    if os.path.exists(savepath):
                        os.remove(savepath)
                    # dict = {}
                    dict = dict1.copy()
                    for k in os.listdir(path2):
                        if k[-4:]=='.npy' and k[:-4]!='unsupervised' and k[:-4]!='supervised':
                            X=np.load(os.path.join(path2,k),allow_pickle=True)
                            for m in my_func_unsupervised:
                                dict[m+'_'+k[5:-4]]=my_classifier(X,m)
                        # np.save(savepath,dict)
                    print(dict)
                    # json_str=json.dumps(data,cls=MyEncoder,indent=4)
                    np.save(savepath, dict)


def main_supervised():
    path='../datafile/'
    # for i in os.listdir(path):
    for i in dataname:
        path1=os.path.join(path,i)
        if os.path.isdir(path1):
            dict1={}
            ordata=np.load('../datafile/'+str(i)+'.npy')
            for m in my_func_supervised:
                print(m)
                dict1[m + '_' + 'or'] = my_classifier(ordata, m)
            for j in os.listdir(path1):
                if j!='KPCA':
                    path2=os.path.join(path1,j)
                    savepath = path2 + '/' + 'supervised.npy'
                    print(path2)
                    if os.path.exists(savepath):
                        os.remove(savepath)
                    # dict = {}
                    dict = dict1.copy()
                    for k in os.listdir(path2):
                        if k[-4:]=='.npy' and k[:-4]!='unsupervised' and k[:-4]!='supervised':
                            X=np.load(os.path.join(path2,k),allow_pickle=True)
                            for m in my_func_supervised:
                                dict[m+'_'+k[5:-4]]=my_classifier(X,m)
                        # np.save(savepath,dict)
                    print(dict)
                    # json_str=json.dumps(data,cls=MyEncoder,indent=4)
                    np.save(savepath, dict)

def plot_un():
    # funcname = ['my_kmeans','my_meanshift','my_dbscan','my_spectral','my_hie']
    num=['or','2','3','5','10']
    path = '../datafile/'
    for i in os.listdir(path):
        path1=os.path.join(path,i)
        if os.path.isdir(path1):
            if i =='spambase':
                funcname = ['my_kmeans','my_meanshift','my_dbscan','my_hie']
            else:
                funcname = ['my_kmeans', 'my_meanshift', 'my_dbscan', 'my_spectral', 'my_hie']
            for j in os.listdir(path1):
                if j != 'KPCA':
                    path2=os.path.join(path1,j)
                    print(path2)
                    plot_unsupervised(path2,funcname,num)

def plot():
    # funcname = ['my_kmeans','my_meanshift','my_dbscan','my_spectral','my_hie']
    num=['or','2','3','5','10']
    path = '../datafile/'
    for i in os.listdir(path):
        path1=os.path.join(path,i)
        if os.path.isdir(path1):
            funcname = ['my_Adaboost','my_rfc','my_NBayes','my_svm','my_network']
            funcname_num = {
                'ababoost': 0,
                'bp': 1,
                'nb': 2,
                'rfc': 3,
                'svm': 4
            }
            for j in os.listdir(path1):
                path2=os.path.join(path1,j)
                print(path2)
                plot_supervised(path2,funcname,num)

main_supervised()







