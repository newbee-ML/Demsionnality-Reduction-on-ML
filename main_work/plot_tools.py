import numpy as np
import matplotlib.pyplot as plt


# plot 3D data
def plot_3d(X, Y, title_n,out_dim, if_save=0, save_path='0'):
    X = np.array(X)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y)
    ax.set_xlabel('$dim_1$')
    ax.set_ylabel('$dim_2$')
    ax.set_zlabel('$dim_3$')
    plt.title(title_n)
    # plt.show()
    if if_save:
        plt.savefig(save_path+str(out_dim)+'.png')
    plt.close()


# plot 2D data
def plot_2d(X, Y, title_n,out_dim,if_save=0, save_path='0'):
    X = np.array(X)
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    plt.xlabel('$dim_1$')
    plt.ylabel('$dim_2$')
    plt.title(title_n)
    # plt.show()

    if if_save:
        plt.savefig(save_path+str(out_dim)+'.png')
    plt.close()

def plot_unsupervised(path,funcname,num):
    dict=np.load(path+'/'+'unsupervised.npy',allow_pickle=True).item()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 可以解释中文无法显示的问题
    plt.rcParams['axes.unicode_minus']=False
    plt.figure(
        figsize=(12,6)
    )
    for i in funcname:
        list1=[]
        list2=[]

        for j in num:
            list1.append(dict[i+'_'+j][0])
            list2.append(dict[i + '_' + j][1])

        plt.subplot(1,2,1)
        plt.plot(num,list1,marker='*',label = i)
        plt.title('轮廓系数')
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(num, list2, marker='*', label=i)
        plt.title('互信息')
        plt.legend()

    plt.savefig(path+'/'+'score_1.png')

    plt.figure(
        figsize=(12, 4)
    )
    for i in funcname:
        list1=[]
        list2=[]
        list3=[]
        for j in num:
            list1.append(dict[i+'_'+j][2])
            list2.append(dict[i + '_' + j][3])
            list3.append(dict[i + '_' + j][4])

        plt.subplot(1,3,1)
        plt.plot(num,list1,marker='*',label = i)
        plt.title('homogeneity_score')
        plt.legend()

        plt.subplot(1,3,2)
        plt.plot(num, list2, marker='*', label=i)
        plt.title('completeness_score')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(num, list3, marker='*', label=i)
        plt.title('v_measure_score')
        plt.legend()

    plt.savefig(path+'/'+'score_2.png')
    plt.close()


def plot_supervised(path,funcname,num):
    dict=np.load(path+'\\'+'supervised.npy',allow_pickle=True).item()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 可以解释中文无法显示的问题
    plt.rcParams['axes.unicode_minus']=False

    # for i in funcname:
    #     list1=[]
    #     for j in num:
    #         a=(path).split('\\')[-2]
    #         b = (path).split('\\')[-1]
    #         list1.append(round(float(dict[a+'_'+b+'_'+j][funcname_num[i]]),3))
    #     plt.plot(num,list1,marker='*',label = i)
    #     plt.title('acc')
    #     plt.legend()

    for i in funcname:
        list1=[]
        for j in num:
            list1.append(dict[i+'_'+j][0])
        plt.plot(num, list1, marker='*', label=i)
        plt.title('acc')
        plt.legend()
    plt.savefig(path+'/'+'score_1.png')
    plt.close()

