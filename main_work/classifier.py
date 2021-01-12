from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.svm import SVC
from sklearn.cluster import KMeans,MeanShift,estimate_bandwidth,DBSCAN,SpectralClustering,AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn import metrics
import pandas as pd
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

#层次聚类
def my_hie(X,**args):
    data = X[:, 1:]
    target = X[:, 0]
    ac = AgglomerativeClustering(n_clusters=len(set(target)))
    ac.fit(data)
    labels = ac.labels_
    if len(set(labels)) == 1:
        score1 = -1
    else:
        score1 = metrics.silhouette_score(data, labels)
    score2 = metrics.adjusted_mutual_info_score(target, labels)
    score3 = metrics.homogeneity_score(target, labels)
    score4 = metrics.completeness_score(target, labels)
    score5 = metrics.v_measure_score(target, labels)
    return score1, score2, score3, score4, score5

# DBSCAN聚类
def my_dbscan(X,**args):
    eps_list=[1,10,100,1000]
    score=[]
    for i in eps_list:
        print(my_dbscan_train(X,i))
        score.append(my_dbscan_train(X,i)[0])
    eps=eps_list[np.argmax(score)]
    return my_dbscan_train(X,eps)

def my_dbscan_train(X,eps,**args):
    data = X[:, 1:]
    target = X[:, 0]
    db = DBSCAN(eps=eps, min_samples=10)
    db.fit(data)
    labels = db.labels_
    if len(set(labels)) == 1:
        score1 = -1
    else:
        score1 = metrics.silhouette_score(data, labels)
    score2 = metrics.adjusted_mutual_info_score(target, labels)
    score3 = metrics.homogeneity_score(target, labels)
    score4 = metrics.completeness_score(target, labels)
    score5 = metrics.v_measure_score(target, labels)
    return score1, score2, score3, score4, score5


#谱聚类
def my_spectral(X,**args):
    data = X[:, 1:]
    target = X[:, 0]
    sc = SpectralClustering(n_clusters=len(set(target)))
    a=sc.fit(data)
    print(a)
    labels=sc.labels_
    if len(set(labels)) == 1:
        score1 = -1
    else:
        score1 = metrics.silhouette_score(data, labels)
    score2 = metrics.adjusted_mutual_info_score(target, labels)
    score3 = metrics.homogeneity_score(target, labels)
    score4 = metrics.completeness_score(target, labels)
    score5 = metrics.v_measure_score(target, labels)
    return score1, score2, score3, score4, score5

#Kmeans聚类
def my_kmeans(X,**args):
    data=X[:,1:]
    target=X[:,0]
    k=len(set(target))
    kmeans= KMeans(n_clusters=k)
    labels=kmeans.fit_predict(data)
    if len(set(labels)) == 1:
        score1 = -1
    else:
        score1 = metrics.silhouette_score(data, labels)
    score2 = metrics.adjusted_mutual_info_score(target, labels)
    score3 = metrics.homogeneity_score(target, labels)
    score4 = metrics.completeness_score(target, labels)
    score5 = metrics.v_measure_score(target, labels)
    return score1, score2, score3, score4, score5



#均值漂移
def my_meanshift(X,**args):
    data = X[:, 1:]
    target = X[:, 0]
    bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=int(0.2*len(target)))
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    # ms=MeanShift()
    ms.fit(data)
    y=ms.labels_
    labels= ms.fit_predict(data)
    if len(set(labels)) == 1:
        score1 = -1
    else:
        score1 = metrics.silhouette_score(data, labels)
    score2 = metrics.adjusted_mutual_info_score(target, labels)
    score3 = metrics.homogeneity_score(target, labels)
    score4 = metrics.completeness_score(target, labels)
    score5 = metrics.v_measure_score(target, labels)
    return score1, score2, score3, score4, score5


#Adaboost
def my_Adaboost(X, **args):
    # 加载数据集，切分数据集80%训练，20%测试
    x= X[:, 1:]
    y = X[:, 0].T
    num_class1=len(set(X[:,0]))
    x_train, x_test, y_train, y_test \
        = train_test_split(x, y, test_size=0.2, random_state=0)
    if type(x[0][0]) == np.complex128:
        return [0]
    else:
        model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=2),
                                   algorithm="SAMME",
                                   n_estimators=200, learning_rate=0.8)
        ab_model = Pipeline(steps=[('StandardScaler', StandardScaler()),  # 数据标准化
                                   ('Poly', PolynomialFeatures()),  # 多项式扩展
                                   ('classifier', model)])
        ab_model.fit(x_train, y_train)
        y_pred_ab = ab_model.predict(x_test)
        if num_class1 == 2:
            confusion_m = confusion_matrix(y_test, y_pred_ab)
            ab_confusion_m = pd.DataFrame(confusion_m, columns=['0', '1'], index=['0', '1'])
            ab_confusion_m.index.name = 'Real'
            ab_confusion_m.columns.name = 'Predict'
            # print(ab_confusion_m)
            y_score = ab_model.predict_proba(x_test)
            fpr, tpr, thresholds = roc_curve(y_test, y_score[:, [1]])
            auc_s = auc(fpr, tpr)
            # 准确率
            accuracy_s = accuracy_score(y_test, y_pred_ab)
            # 精准度
            precision_s = precision_score(y_test, y_pred_ab)
            recall_s = recall_score(y_test, y_pred_ab)
            # F1得分
            f1_s = f1_score(y_test, y_pred_ab)
            ab_metrics = pd.DataFrame([[auc_s, accuracy_s, precision_s, recall_s, f1_s]])
            # print(ab_metrics)
            # plt.figure(figsize=(8, 7))
            # plt.plot(fpr, tpr, label='ROC')  # 画出ROC曲线
            # plt.plot([0, 1], [0, 1], linestyle='--', color='k', label='random chance')
            # plt.savefig(name1.split('.')[0] + '_AB')
            # plt.close()
        count = 0
        for i in range(len(y_test)):
            if y_test[i] == y_pred_ab[i]:
                count += 1
        if num_class1 == 2:
            return (count / len(y_pred_ab)), ab_metrics
        else:
            return (count / len(y_pred_ab))


#随机森林
def my_rfc(X, **args):
    # 加载数据集，切分数据集80%训练，20%测试
    x = X[:, 1:]
    y = X[:, 0].T
    num_class1=len(set(X[:,0]))
    x_train, x_test, y_train, y_test \
        = train_test_split(x, y, test_size=0.2, random_state=0)
    if type(x[0][0]) == np.complex128:
        return [0]
    else:
        randomForestClassifier = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
        rfc_model = Pipeline(steps=[('StandardScaler', StandardScaler()),  # 数据标准化
                                    ('Poly', PolynomialFeatures()),  # 多项式扩展
                                    ('classifier', randomForestClassifier)])
        rfc_model.fit(x_train, y_train)
        y_pred_rfc = rfc_model.predict(x_test)
        if num_class1 == 2:
            confusion_m = confusion_matrix(y_test, y_pred_rfc)
            rfc_confusion_m = pd.DataFrame(confusion_m, columns=['0', '1'], index=['0', '1'])
            rfc_confusion_m.index.name = 'Real'
            rfc_confusion_m.columns.name = 'Predict'
            # print(rfc_confusion_m)
            y_score = rfc_model.predict_proba(x_test)
            fpr, tpr, thresholds = roc_curve(y_test, y_score[:, [1]])
            auc_s = auc(fpr, tpr)
            # 准确率
            accuracy_s = accuracy_score(y_test, y_pred_rfc)
            # 精准度
            precision_s = precision_score(y_test, y_pred_rfc)
            recall_s = recall_score(y_test, y_pred_rfc)
            # F1得分
            f1_s = f1_score(y_test, y_pred_rfc)
            rfc_metrics = pd.DataFrame([[auc_s, accuracy_s, precision_s, recall_s, f1_s]])
            # print(rfc_metrics)
            # plt.figure(figsize=(8, 7))
            # plt.plot(fpr, tpr, label='ROC')  # 画出ROC曲线
            # plt.plot([0, 1], [0, 1], linestyle='--', color='k', label='random chance')
            # plt.savefig(name1.split('.')[0] + '_RFC')
            # plt.close()
        count = 0
        for i in range(len(y_test)):
            if y_test[i] == y_pred_rfc[i]:
                count += 1
        if num_class1 == 2:
            return (count / len(y_pred_rfc)), rfc_metrics
        else:
            return (count / len(y_pred_rfc))


#朴素贝叶斯
def my_NBayes(X,**args):
    # 加载数据集，切分数据集80%训练，20%测试
    x = X[:, 1:]
    y = X[:, 0].T
    num_class1 = len(set(X[:, 0]))
    x_train, x_test, y_train, y_test \
        = train_test_split(x, y, test_size=0.2, random_state=0)
    if type(x[0][0]) == np.complex128:
        return [0]
    else:
        nbClassifier = GaussianNB()
        nb_model = Pipeline(steps=[('StandardScaler', StandardScaler()),  # 数据标准化
                                   ('Poly', PolynomialFeatures()),  # 多项式扩展
                                   ('classifier', nbClassifier)])
        nb_model.fit(x_train, y_train)
        y_pred_nb = nb_model.predict(x_test)
        if num_class1 == 2:
            confusion_m = confusion_matrix(y_test, y_pred_nb)
            nb_confusion_m = pd.DataFrame(confusion_m, columns=['0', '1'], index=['0', '1'])
            nb_confusion_m.index.name = 'Real'
            nb_confusion_m.columns.name = 'Predict'
            # print(nb_confusion_m)
            y_score = nb_model.predict_proba(x_test)
            fpr, tpr, thresholds = roc_curve(y_test, y_score[:, [1]])
            auc_s = auc(fpr, tpr)
            # 准确率
            accuracy_s = accuracy_score(y_test, y_pred_nb)
            # 精准度
            precision_s = precision_score(y_test, y_pred_nb)
            recall_s = recall_score(y_test, y_pred_nb)
            # F1得分
            f1_s = f1_score(y_test, y_pred_nb)
            nb_metrics = pd.DataFrame([[auc_s, accuracy_s, precision_s, recall_s, f1_s]])
            # print(nb_metrics)
            # plt.figure(figsize=(8, 7))
            # plt.plot(fpr, tpr, label='ROC')  # 画出ROC曲线
            # plt.plot([0, 1], [0, 1], linestyle='--', color='k', label='random chance')
            # plt.savefig(name1.split('.')[0] + '_NB')
            # plt.close()
        count = 0
        for i in range(len(y_test)):
            if y_test[i] == y_pred_nb[i]:
                count += 1
        if num_class1 == 2:
            return (count / len(y_pred_nb)), nb_metrics
        else:
            return (count / len(y_pred_nb))

#支撑向量机
def my_svm(X,**args):
    x = X[:, 1:]
    y = X[:, 0].T
    num_class1 = len(set(X[:, 0]))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    if type(x[0][0]) == np.complex128:
        return [0]
    else:
        x_train, x_test, y_train, y_test \
            = train_test_split(x, y, test_size=0.2, random_state=0)
        svmClassifier = SVC(kernel='rbf', gamma='auto',probability=True)
        svm_model = Pipeline(steps=[('StandardScaler', StandardScaler()),  # 数据标准化
                                    ('Poly', PolynomialFeatures()),  # 多项式扩展
                                    ('classifier', svmClassifier)])
        svm_model.fit(x_train, y_train)
        y_pred_svm = svm_model.predict(x_test)
        if num_class1 == 2:
            confusion_m = confusion_matrix(y_test, y_pred_svm)
            svm_confusion_m = pd.DataFrame(confusion_m, columns=['0', '1'], index=['0', '1'])
            svm_confusion_m.index.name = 'Real'
            svm_confusion_m.columns.name = 'Predict'
            # print(svm_confusion_m)
            y_score = svm_model.predict_proba(x_test)
            fpr, tpr, thresholds = roc_curve(y_test, y_score[:, [1]])
            auc_s = auc(fpr, tpr)
            # 准确率
            accuracy_s = accuracy_score(y_test, y_pred_svm)
            # 精准度
            precision_s = precision_score(y_test, y_pred_svm)
            recall_s = recall_score(y_test, y_pred_svm)
            # F1得分
            f1_s = f1_score(y_test, y_pred_svm)
            svm_metrics = pd.DataFrame([[auc_s, accuracy_s, precision_s, recall_s, f1_s]])
            # print(svm_metrics)
            # plt.figure(figsize=(8, 7))
            # plt.plot(fpr, tpr, label='ROC')  # 画出ROC曲线
            # plt.plot([0, 1], [0, 1], linestyle='--', color='k', label='random chance')
            # plt.savefig(name1.split('.')[0] + '_SVM')
            # plt.close()
        count = 0
        for i in range(len(y_test)):
            if y_test[i] == y_pred_svm[i]:
                count += 1
        if num_class1 == 2:
            return (count / len(y_pred_svm)), svm_metrics
        else:
            return (count / len(y_pred_svm))

#全连接网络
def my_network(X,**args):
    lr_list=[0.01,0.001,0.0001,0.00001]
    score=[]
    for i in lr_list:
        score.append(my_network_train(X,i))
    lr=lr_list[np.argmax(score)]
    return my_network_train(X,lr)

def my_network_train(X,lr,**args):
    batch_size=32
    data = X[:, 1:]
    target = X[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)
    num_features=data.shape[1]
    num_class=len(set(target))
    model = BP(num_features,num_class)
    train_data = mydata(X_train,y_train)
    trainloader = DataLoader(train_data, batch_size, shuffle=False)
    val_data = mydata(X_test,y_test)
    valloader = DataLoader(val_data, batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    batch_train_results = []
    batch_train_loss = []
    batch_val_results=[]
    batch_val_loss=[]
    best_result=0
    for epoch in range(100):
        running_loss1 = 0
        total1 = 0
        result1 = 0
        running_loss2 = 0
        total2 = 0
        result2 = 0
        model.train()
        for ii, (data, label) in enumerate(trainloader):
            every_loss1 = 0
            input = Variable(data)
            label = Variable(label)
            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, label)
            loss.backward()
            optimizer.step()
            running_loss1 += loss.item()
            every_loss1 = loss.item()
            pro = np.array(score.detach().max(1)[1].tolist())
            label = np.array(label.tolist())
            total1 += pro.shape[0]
            result1 += np.sum(label == pro)
        batch_train_loss.append(float(running_loss1 / total1 * batch_size))
        batch_train_results.append(float(result1 / total1))
        model.eval()
        for ii, (data, label) in enumerate(valloader):
            every_loss2 = 0
            input = Variable(data)
            label = Variable(label)
            # model.zero_grad()
            score = model(input)
            loss = criterion(score, label)
            running_loss2 += loss.item()
            every_loss2 += loss.item()
            pro = np.array(score.detach().max(1)[1].tolist())
            label = np.array(label.tolist())
            total2 += pro.shape[0]
            result2 += np.sum(label == pro)
        batch_val_loss.append(float(running_loss2 / total2 * batch_size))
        batch_val_results.append(float(result2 / total2))
        # torch.save(model, '../modelfile/1.pt')
        # print(batch_train_results[-1],batch_val_results[-1])
        if np.abs(batch_train_results[-1] - batch_val_results[-1]) < 0.05 and best_result < batch_val_results[-1]:
            best_result = batch_val_results[-1]
            torch.save(model, '../modelfile/1.pt')
    model = torch.load('../modelfile/1.pt')
    total = 0
    result = 0
    label_list = []
    pro_list = []
    for ii, (data, label) in enumerate(valloader):
        input = data
        score = model(input)
        pro = np.array(score.detach().max(1)[1].tolist())
        label = np.array(label.tolist())
        total += pro.shape[0]
        result += np.sum(label == pro)
        label_list.append(label)
        pro_list.append(pro)
    score=result / total
    return score

class BP(nn.Module):
    def __init__(self,num_features,num_class):
        super(BP, self).__init__()
        self.classifier=nn.Sequential(
            nn.Linear(num_features,32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16,num_class)
        )
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class mydata(data.Dataset):
    def __init__(self, X,y,train=True):
        self.data=X
        self.target=y
    def __getitem__(self, index):
        data = torch.FloatTensor([self.data[index,:]])
        label = int(self.target[index])
        return data, label
    def __len__(self):
        return len(self.target)

