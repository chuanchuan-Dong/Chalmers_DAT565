import pandas as pd
import matplotlib.pyplot as plt
"""
Read data, Process data
"""
data = pd.read_csv('data/protein-angle-dataset.csv')
name = data['residue name'].unique()
for i in name:
    plt.scatter(data[data['residue name']==i]['phi'],
            data[data['residue name']==i]["psi"], 
                label=i,
                s=5)
print(data['residue name'][0])
plt.legend()       
plt.show()