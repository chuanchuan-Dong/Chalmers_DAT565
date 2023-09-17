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
data_array = [[data.iloc[i]['phi'], data.iloc[i]['psi']] for i in range(len(data))]
data_array = np.array(data_array)
# data['phi'] = (data['phi'] - data['phi'].mean()) / data['phi'].std()
# data['psi'] = (data['psi'] - data['psi'].mean()) / data['psi'].std()

def task1():
    residue_name = data['residue name'].unique()
    for name in residue_name:
        phi, psi = data[data['residue name'] == name]['phi'].values, data[data['residue name'] == name]['psi'].values
        plt.scatter(phi, psi, label=name, s = 0.9)
    
    plt.xlabel('phi')
    plt.ylabel('psi')
    plt.title('the scatter plot of the given data')
    plt.legend()
    # plt.show()
    plt.clf()
    
    plt.hist2d(data['phi'], data['psi'], bins=50, cmap = 'Blues', density=True)
    plt.xlabel('phi')
    plt.ylabel('psi')
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.title('Histogram 2d between psi and phi')
    plt.show()

def task2(max_cluster = 8):
    score_list = []
    start_k = 2
    for i in range(start_k, max_cluster+1):        
        k = KMeans(n_clusters=i).fit(data_array)
        score = silhouette_score(data_array, k.predict(data_array))
        score_list.append(score)
        labels = np.array(k.labels_)
        
        plt.subplot(2, 4, i - start_k + 1)
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
    plt.plot([i for i in range(start_k, max_cluster + 1)], score_list)
    plt.show()

def task3():
    # # draw k-distance 
    task3_data = data
    task3_data['phi'] = (data['phi'] - data['phi'].mean()) / data['phi'].std()
    task3_data['psi'] = (data['psi'] - data['psi'].mean()) / data['psi'].std()
    task3_data_array = [[task3_data.iloc[i]['phi'], task3_data.iloc[i]['psi']] for i in range(len(data))]
    
    task3_data_array = np.array(task3_data_array)
    
    
    # for times, sample in enumerate(range(200, 501, 100)):
    #     dbscan = DBSCAN(eps=0.3,min_samples=sample).fit(task3_data_array)
    #     # dbscan.fit(data_array)
    #     labels = dbscan.labels_
    #     unique_label, count = np.unique(labels, return_counts=True)
    #     print(unique_label, count)
    #     plt.subplot(2,2,times+1)
    #     for i in range(-1, max(labels) + 1):
    #         index = np.where(labels==i)[0]
    #         # print(len(index))
    #         phi, psi = data.iloc[index]['phi'].values, data.iloc[index]['psi'].values
    #         if i == -1:
    #             plt.scatter(phi, psi, label=f'Outlier, Count:{count[i+1]} ', s=7, color='red', marker='*')
    #         else:

    #             plt.scatter(phi, psi, label=f'Categories:{i}, Count:{count[i+1]} ', s=0.7)
    #     plt.legend(loc='upper right')
    #     plt.xlabel('Phi')
    #     plt.ylabel('Psi')
    #     plt.title(f'sample={sample}')
    # plt.show()
    db = DBSCAN(eps=0.3, min_samples=200)
    db.fit(task3_data_array)
    # data_new = pd.concat([task3_data,pd.Series({'label': db.labels_})], axis=1)
    task3_data['label'] = db.labels_
    data33 =  (task3_data[task3_data['label']==-1]).groupby('residue name')
    plt.bar(data33.size().index, data33.size().values)
    plt.xlabel("resiue type")
    plt.ylabel("Outlier number")
    plt.title("Outliers number VS redidue type")
    plt.show()
"""
Plot the bar chat between residue and outlier in the case of minsample=200, eps=0.3
""" 

    

def task4():
    pro_data = data[data['residue name'] == 'PRO'].copy()
    pro_data['phi'] = (pro_data['phi'] - pro_data['phi'].mean()) / pro_data['phi'].std()
    pro_data['psi'] = (pro_data['psi'] - pro_data['psi'].mean()) / pro_data['psi'].std()
    data_array = [[pro_data.iloc[i]['phi'], pro_data.iloc[i]['psi']] for i in range(len(pro_data))]
    data_array = np.array(data_array)
    # plt.scatter(data_array[:, 0], data_array[:, 1])
    # plt.show()
    dbscan = DBSCAN(min_samples=50).fit(data_array)
    labels = dbscan.labels_
    for i in range(-1, max(labels) + 1):
        index = np.where(labels==i)[0]
        # print(len(index))
        phi, psi = pro_data.iloc[index]['phi'].values, pro_data.iloc[index]['psi'].values
        plt.scatter(phi, psi, label=i, s=1.5)
    plt.legend(labels=[j for j in range(-1, max(labels)+1)], loc='lower right')
    plt.xlabel('Phi')
    plt.ylabel('Psi')
    plt.title('DBSCAN Clustering')
    plt.show()
    


if __name__ == '__main__':
    # task1()
    # # task2()
    task3()
    task4()