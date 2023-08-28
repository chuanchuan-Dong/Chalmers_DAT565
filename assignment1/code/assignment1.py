import pandas as pd
import matplotlib.pyplot as plt

root = 'python/DAT565/data/'
data = pd.read_csv(root + 'gdp-per-capita-penn-world-table.csv')

# print(data.shape)
# print(data.head())

# check the country data
columns = data.columns
# entity = set(data[columns[0]])
# print(entity)
# print(len(entity))

# print(data[columns[0] == 'Albania'])

albania_data = data[data[columns[0]] == 'Albania']
plot_data = albania_data
print(albania_data)