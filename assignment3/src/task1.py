import pandas as pd
import matplotlib.pyplot as plt
"""
Read data, Process data
"""
data = pd.read_csv('data/protein-angle-dataset.csv')
print(data.index)

for i in data.index:
    plt.scatter(data['phi'][i],
                data["psi"][i], 
                label=data['residue name'][i],
                s=5)
print(data['residue name'][0])
plt.legend()       
plt.show()