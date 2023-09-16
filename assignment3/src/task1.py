import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
"""
Read data, Process data
"""
data = pd.read_csv('data/protein-angle-dataset.csv')
residue_name = data['residue name'].unique()

print(data['chain'].unique())

data['phi'] = (data['phi'] - data['phi'].mean()) / data['phi'].std()
data['psi'] = (data['psi'] - data['psi'].mean()) / data['psi'].std()

# for name in residue_name:
#     phi, psi = data[data['residue name'] == name]['phi'].values, data[data['residue name'] == name]['psi'].values
#     plt.scatter(phi, psi, label=name)
    
# plt.legend()
# plt.show()


data_array = [[data.iloc[i]['phi'], data.iloc[i]['psi']] for i in range(len(data))]


data_array = np.array(data_array)
clusters = 4
k = KMeans(n_clusters=clusters).fit(data_array)
print(k.labels_)
print(len(k.labels_))
labels = np.array(k.labels_)

for i in range(clusters):
    index = np.where(labels==i)[0]
    phi, psi = data.iloc[index]['phi'].values, data.iloc[index]['psi'].values
    plt.scatter(phi, psi, label=i)
    # print(index)
    # exit(0)

# plt.scatter(data['phi'], data['psi'], label = k.labels_)
plt.show()
exit(0)
print(residue_name)

print(data.index)

print(data.head())

# plt.scatter(data['phi'].values, data['psi'].values)

# plt.show()

# for i in range(0, len(data), 100):
#     # print(data.loc[0]['phi'])
#     plt.scatter(data.iloc[i]['phi'], data.iloc[i]['psi'])

# plt.savefig('./pic.png')
# plt.show()

exit(0)

for i in data.index:
    plt.scatter(data['phi'][i],
                data["psi"][i], 
                label=data['residue name'][i],
                s=5)
print(data['residue name'][0])
plt.legend()       
plt.show()