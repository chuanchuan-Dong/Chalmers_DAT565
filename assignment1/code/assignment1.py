import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
Defines here
'''
LifeExpShort = 'Life expectancy at birth, total (years)'
GdpShort = 'GDP per capita (output, multiple price benchmarks)'


'''
Read the Data
Process two Dataset and Combine two dataset
Set the same years arrange in diffent countries, if miss the datas, fill it with mean
'''
GdpData = pd.read_csv('data/gdp-per-capita-penn-world-table.csv')
LifeExpData = pd.read_csv('data/life-expectancy-at-birth-total-years.csv')

LifeExpCountry, GdpCountry = LifeExpData['Entity'].unique(), GdpData['Entity'].unique()
all_CountryList = np.intersect1d(LifeExpCountry, GdpCountry)         
all_years = np.arange(1970,2020)

#create the new dataframe, contains all possible combinations.
all_combination = pd.MultiIndex.from_product([all_CountryList, all_years], names=['Entity', 'Year'])
CombinedData = pd.DataFrame(index=all_combination).reset_index()

#merge data accroding to the CombinedData colum index.
CombinedData = pd.merge(CombinedData, LifeExpData, on=['Entity','Year'], how='left' )
CombinedData = pd.merge(CombinedData, GdpData, on=['Entity','Year','Code'], how='left' )

#Fill the miss data by the mean of each country
CountryMeans = CombinedData.groupby('Entity')[[LifeExpShort,GdpShort]].mean()


CombinedData[LifeExpShort].fillna(CountryMeans[LifeExpShort], inplace=True)
CombinedData[GdpShort].fillna(CountryMeans[GdpShort], inplace=True)


def task1(DataSet):
    LifeExp_Mean, LifeExp_std = DataSet[LifeExpShort].mean(), DataSet[LifeExpShort].std()
    Country_LifeExp_Mean = DataSet.groupby("Entity")[LifeExpShort].mean()
    Upper_LifeExp = Country_LifeExp_Mean[Country_LifeExp_Mean >= LifeExp_Mean+LifeExp_std]
    Upper_LifeExp = Upper_LifeExp.sort_values(ascending=False) # sort the value
    print(Upper_LifeExp)
    plt.figure()
    plt.bar(Upper_LifeExp.index, Upper_LifeExp.values)
    plt.axhline(y=LifeExp_Mean, color='orange', label='Gloabl Mean')
    plt.axhline(y=LifeExp_std, color='green', label='Gloabl std' )
    plt.axhline(y=LifeExp_std+LifeExp_Mean, color='red', label='one gloabl std above mean' )
    plt.xlabel('Country')
    plt.ylabel('Average Life Expectancy')
    plt.title('Average Life Expectancy by Country higher than one standard deviation above the mean ')
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()


def task2(DataSet):
  '''
   high life expectancy but have low GDP
   Defien high as mean+std, Define low as mean-std
  '''
  LifeExp_Mean, LifeExp_std = DataSet[LifeExpShort].mean(), DataSet[LifeExpShort].std()
  Gdp_Mean, Gdp_std = DataSet[GdpShort].mean(), DataSet[GdpShort].std()
  High_LifeExp_threshold = LifeExp_Mean + 0.35*LifeExp_std
  Low_Gdp_threshold = Gdp_Mean - 0.35*Gdp_std
  Country_LifeExp_Mean = DataSet.groupby("Entity")[[LifeExpShort, GdpShort]].mean()
  ans = Country_LifeExp_Mean[(Country_LifeExp_Mean[LifeExpShort] >= High_LifeExp_threshold) 
                            & (Country_LifeExp_Mean[GdpShort] <= Low_Gdp_threshold)]
  print(ans)
  

if __name__ == '__main__':
# Uncomment when executing task
  task1(CombinedData)
  # task2(CombinedData)
