import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
"""
Read data, Process data
"""
data = pd.read_csv('data/protein-angle-dataset.csv')
name = data['chain'].unique()
for i in name:
    plt.scatter(data[data['chain']==i]['phi'],
                data[data['chain']==i]["psi"], 
                s=1)
# fig,(ax1,ax2) = plt.subplots(1,2)
# ax1.scatter(data['phi'], data["psi"], s=0.5, alpha=0.5 )
# ax1.set_xlim(-180, 180)  
# ax1.set_ylim(-180, 180)  
# ax2.hist2d(data['phi'], data["psi"])
# ax2.set_xlim(-180, 180)  
# ax2.set_ylim(-180, 180)  
plt.show()

X = data[['phi','psi']].values
scaler = StandardScaler()
X_std = scaler.fit_transform(X)



# draw k-distance 
k = 3
k_dis = []
file_path = 'data/Kdis.npy'  
if os.path.exists(file_path):
    print('exist k_dis')
else:
    for i in range(X_std.shape[0]):
        dis =( ((X_std[i] - X_std)**2).sum(axis=1)**0.5 )
        dis.sort()
        k_dis.append(dis[k])
    k_dis.sort(reverse=True)
    np.save('data/Kdis.npy',k_dis)
    print('save to data/Kdis.npy')

k_dis = np.load('data/Kdis.npy')
print(k_dis)
plt.plot(np.arange(k_dis.shape[0]), k_dis)
plt.show()

differences = np.diff(k_dis)
print(differences)

eps = 0.0649  
min_samples = 4  # 最小样本数
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan.fit(X_std)

labels = dbscan.labels_
plt.scatter(X_std[:, 0], X_std[:, 1], c=labels, s=0.5)
plt.xlabel('Phi')
plt.ylabel('Psi')
plt.title('DBSCAN Clustering')
plt.show()