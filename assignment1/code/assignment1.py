import pandas as pd
import matplotlib.pyplot as plt

root = './assignment1/data/'

def preprocess_data():
    gdp_data = pd.read_csv(root + 'gdp-per-capita-penn-world-table.csv')
    life_data = pd.read_csv(root + 'life-expectancy-at-birth-total-years.csv')
    life_columns, gdb_columns = life_data.columns, gdp_data.columns
    year = life_columns[2]
    life_country, gdb_country = set(life_data[life_columns[0]]), set(gdp_data[gdb_columns[0]])
    
    # find the same countries
    country = list(life_country & gdb_country)
    # print(country)
    
    new_life_data = life_data[(life_data[year] >= 1970) & (life_data[year] <= 2019)]
    new_life_data = new_life_data[new_life_data[life_columns[0]].isin(country)]
    new_gdp_data = gdp_data[gdp_data[gdb_columns[0]].isin(country)]
    
    # d = dict()
    # diff = 0
    # for item in country:
    #     value = [len(new_life_data[new_life_data[life_columns[0]] == item]), len(new_gdp_data[new_gdp_data[gdb_columns[0]] == item])]
    #     d[item] = value
    #     diff += value[1] - value[0]
    # # print(new_life_data.head)
    # print(d)
    # print(diff)    
    
    print(len(new_life_data))
    print(len(new_gdp_data))

    return new_gdp_data, new_life_data

def task1():
    gdp_data, life_data = preprocess_data()
    mean_life, std_life = life_data[life_data.columns[3]].mean(), life_data[life_data.columns[3]].std()
    result = life_data[[life_data.columns[0], life_data.columns[3]]].groupby('Entity').mean()
    result = result.sort_values(by=life_data.columns[3], ascending=False)
    result = result[result[result.columns[0]] >= mean_life + std_life]
    print(mean_life)    
    print(std_life)
    print(result)
    print(type(result))

# task1()

gdp_data, life_data = preprocess_data()

print(gdp_data.isna().any())


# def check_life_mean():
#     columns = life_data.columns
#     life_item = columns[3]
#     life_mean = life_data[life_item].mean()
#     life_var = life_mean
#     print(life_mean)
#     print(columns)

# check_life_mean()

# print(life_data.head())
exit(0)

# print(gdb_data.shape)
# print(gdb_data.head())

# check the country gdb_data
columns = gdp_data.columns
# entity = set(gdb_data[columns[0]])
# print(entity)
# print(len(entity))

# print(gdb_data[columns[0] == 'Albania'])

albania_gdb_data = gdp_data[gdp_data[columns[0]] == 'Albania']
plot_gdb_data = albania_gdb_data
print(albania_gdb_data)