from dim_reduct_function import *
from plot_tools import *
import os
"""
load data class
"""


def load_data(dataset_name):
    path = {
        'glass': r'..\datafile\glass.npy',
        'credit_card': r'..\datafile\credit_card.npy',
        'spambase': r'..\datafile\spambase.npy',
        'wdbc': r'..\datafile\wdbc.npy',
        'wine': r'..\datafile\wine.npy',
        'avila': r'..\datafile\avila.npy'
    }
    return np.load(path[dataset_name])


"""
dimensionality reduction method class
"""

method_func_dict = {
    'PCA': my_pca,
    'KPCA': my_kpca,
    'LDA': my_lda,
    'LLE': my_lle,
    'LE': my_le,
    'MDS': my_mds,
    'Isomap': my_isomap,
    'T-SNE': my_tsne
}

method_func = {
    'PCA',
    'KPCA',
    'LDA',
    'LLE',
    'LE',
    'MDS',
    'Isomap',
    'T-SNE'
}

original_class_num={
    'glass':6,
    'credit_card':2,
    'spambase':2,
    'wdbc':2,
    'wine':3,
    'avila':12
}
data={
    # 'glass',
    # 'credit_card',
    # 'spambase',
    # 'wdbc',
    # 'wine',
    'avila'
}
reduct_class={
    2,3,5,10
}


class dim_reduction:
    def __init__(self, method):
        self.method = method

    def fit(self, X, output_dim=3, **args):
        dim_method = method_func_dict[self.method]
        return dim_method(X, output_dim, **args)


"""
main test
"""


def red_data(name, method, out_dim, save_path, save_dim):
    # load data
    data = load_data(name)
    data_x = data[:, 1:]
    data_y = data[:, 0]
    print('class number: ', len(set(data_y)))
    print('sample number: ', data_x.shape[0])
    # dimensionality reduction processing
    drl = dim_reduction(method)
    red_data_x = drl.fit(data_x, out_dim, y=data_y)

    # plot reduction results
    if out_dim == 2:
        plot_2d(red_data_x, data_y, method + ': dim = 2',out_dim,1,save_path,)
    elif out_dim == 3:
        plot_3d(red_data_x, data_y, method + ': dim = 3',out_dim,1,save_path)
    else:
        print('error: your input output dim num is beyond 3')
    data=np.hstack((data_y.reshape(-1,1),red_data_x))
    data=np.real(data)
    np.save(save_path+'data'+'_'+str(save_dim)+'.npy',data)


if __name__ == '__main__':
    # data_name = 'wdbc'
    # method_now = 'LDA'
    # out_dim_num = 2
    # save_path='../resultfile/1/'
    # os.makedirs(save_path)
    # done_data = red_data(data_name, method_now, out_dim_num,save_path)

    # for i in data:
    #     for k in method_func:
    #         for j in reduct_class:
    #             if k=='LDA':
    #                 if j<original_class_num[i]:
    #                     print('dataname', i)
    #                     print('class',j)
    #                     print('original_class',original_class_num[i])
    #                     save_path='../datafile/'+str(i)+'/'+str(k)+'/'
    #                     if not os.path.exists(save_path):
    #                         os.makedirs(save_path)
    #                     print(save_path)
    #                     red_data(i, k, j,save_path,j)
    #                 else:
    #                     print('dataname', i)
    #                     print('class',j)
    #                     print('original_class',original_class_num[i])
    #                     save_path='../datafile/'+str(i)+'/'+str(k)+'/'
    #                     if not os.path.exists(save_path):
    #                         os.makedirs(save_path)
    #                     print(save_path)
    #                     red_data(i, k, original_class_num[i]-1,save_path,j)
    #             else:
    #                 print('dataname', i)
    #                 print('class', j)
    #                 print('original_class', original_class_num[i])
    #                 save_path = '../datafile/' + str(i) + '/' + str(k) + '/'
    #                 if not os.path.exists(save_path):
    #                     os.makedirs(save_path)
    #                 print(save_path)
    #                 red_data(i, k, j, save_path,j)
    i='avila'
    k='Isomap'
    j=2
    save_path = '../datafile/' + str(i) + '/' + str(k) + '/'
    red_data(i, k, j, save_path, j)

