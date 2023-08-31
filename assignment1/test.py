import pandas as pd 

root = './data/national-gdp-penn-world-table.csv'

data = pd.read_csv(root)
print(data.columns[-1])