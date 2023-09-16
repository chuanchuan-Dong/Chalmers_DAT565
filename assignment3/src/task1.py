import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.preprocessing import StandardScaler
"""
Read data, Process data
"""
data = pd.read_csv('data/protein-angle-dataset.csv')
# data['phi'] = (data['phi'] - data['phi'].mean()) / data['phi'].std()
# data['psi'] = (data['psi'] - data['psi'].mean()) / data['psi'].std()

def show_residue_name():
    residue_name = data['residue name'].unique()
    for name in residue_name:
        phi, psi = data[data['residue name'] == name]['phi'].values, data[data['residue name'] == name]['psi'].values
        plt.scatter(phi, psi, label=name, s = 0.7)
        
    plt.legend()
    plt.show()


def task2(max_cluster = 8):
    data_array = [[data.iloc[i]['phi'], data.iloc[i]['psi']] for i in range(len(data))]
    data_array = np.array(data_array)
    score_list = []
    for i in range(3, max_cluster+1):        
        k = KMeans(n_clusters=i).fit(data_array)
        score = silhouette_score(data_array, k.predict(data_array))
        score_list.append(score)
        labels = np.array(k.labels_)
        
        plt.subplot(2, 3, i-2)
        plt.xlabel('phi')
        plt.ylabel('psi')
        plt.title(f'k={i}')
        for j in range(i):
            index = np.where(labels==j)[0]
            phi, psi = data.iloc[index]['phi'].values, data.iloc[index]['psi'].values
            plt.scatter(phi, psi, label=i, s=0.7)
            # print(index)
            # exit(0)
        plt.legend(labels=[j for j in range(i)], loc='lower right')

    # plt.scatter(data['phi'], data['psi'], label = k.labels_)
    plt.suptitle("K-means based on different K")
    plt.show()
    plt.title('Silhouette Score')
    plt.xlabel('the number of cluster')
    plt.ylabel('Silhouette Score')
    plt.plot([i for i in range(3, max_cluster + 1)], score_list)
    plt.show()

def task4():
    pro_data = data[data['residue name'] == 'PRO']
    data_array = [[pro_data.iloc[i]['phi'], pro_data.iloc[i]['psi']] for i in range(len(pro_data))]
    data_array = np.array(data_array)
    plt.scatter(data_array[:, 0], data_array[:, 1])
    plt.show()


if __name__ == '__main__':
    # show_residue_name()
    # task2()
    task4()

# for i in data.index:
#     plt.scatter(data['phi'][i],
#                 data["psi"][i], 
#                 label=data['residue name'][i],
#                 s=5)
# print(data['residue name'][0])
# plt.legend()       
# name = data['chain'].unique()
# for i in name:
#     plt.scatter(data[data['chain']==i]['phi'],
#                 data[data['chain']==i]["psi"], 
#                 s=1)
# # fig,(ax1,ax2) = plt.subplots(1,2)
# # ax1.scatter(data['phi'], data["psi"], s=0.5, alpha=0.5 )
# # ax1.set_xlim(-180, 180)  
# # ax1.set_ylim(-180, 180)  
# # ax2.hist2d(data['phi'], data["psi"])
# # ax2.set_xlim(-180, 180)  
# # ax2.set_ylim(-180, 180)  
# plt.show()

# X = data[['phi','psi']].values
# scaler = StandardScaler()
# X_std = scaler.fit_transform(X)



# # draw k-distance 
# k = 3
# k_dis = []
# file_path = 'data/Kdis.npy'  
# if os.path.exists(file_path):
#     print('exist k_dis')
# else:
#     for i in range(X_std.shape[0]):
#         dis =( ((X_std[i] - X_std)**2).sum(axis=1)**0.5 )
#         dis.sort()
#         k_dis.append(dis[k])
#     k_dis.sort(reverse=True)
#     np.save('data/Kdis.npy',k_dis)
#     print('save to data/Kdis.npy')

# k_dis = np.load('data/Kdis.npy')
# print(k_dis)
# plt.plot(np.arange(k_dis.shape[0]), k_dis)
# plt.show()

# differences = np.diff(k_dis)
# print(differences)

# eps = 0.0649  
# min_samples = 4  # 最小样本数
# dbscan = DBSCAN(eps=eps, min_samples=min_samples)
# dbscan.fit(X_std)

# labels = dbscan.labels_
# plt.scatter(X_std[:, 0], X_std[:, 1], c=labels, s=0.5)
# plt.xlabel('Phi')
# plt.ylabel('Psi')
# plt.title('DBSCAN Clustering')
# plt.show()