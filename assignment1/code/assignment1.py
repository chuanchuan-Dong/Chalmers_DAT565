import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
Defines here
'''
LifeExpShort = 'Life expectancy at birth, total (years)'
GdpShort = 'GDP per capita (output, multiple price benchmarks)'
NationalGdpShort = 'GDP (output, multiple price benchmarks)'

'''
Read the Data
Process two Dataset and Combine two dataset
Set the same years arrange in diffent countries, if miss the datas, fill it with mean
'''
GdpData = pd.read_csv('data/gdp-per-capita-penn-world-table.csv')
LifeExpData = pd.read_csv('data/life-expectancy-at-birth-total-years.csv')
NationalGdpData = pd.read_csv('data/national-gdp-penn-world-table.csv')

LifeExpCountry, GdpCountry, NationalGdpCountry = LifeExpData['Entity'].unique(), GdpData['Entity'].unique(), NationalGdpData['Entity'].unique()
all_CountryList = np.intersect1d(LifeExpCountry, GdpCountry)
all_CountryList = np.intersect1d(all_CountryList, NationalGdpCountry)

CountryAndCode = GdpData[['Entity', 'Code']].drop_duplicates().values
CountryDic = dict()
for item in CountryAndCode:
  CountryDic[item[0]] = item[1]

all_years = np.arange(1970,2020)

#create the new dataframe, contains all possible combinations.
all_combination = pd.MultiIndex.from_product([all_CountryList, all_years], names=['Entity', 'Year'])
CombinedData = pd.DataFrame(index=all_combination).reset_index()
CombinedData['Code'] = [pd.NA for i in range(len(CombinedData))]
CombinedData['Code']=CombinedData['Code'].fillna(CombinedData['Entity'].apply(lambda x: CountryDic.get(x)))

#merge data accroding to the CombinedData colum index.
CombinedData = pd.merge(CombinedData, LifeExpData, on=['Entity','Year', 'Code'], how='left' )
CombinedData = pd.merge(CombinedData, GdpData,on = ['Entity', 'Year', 'Code'], how='left' )
CombinedData = pd.merge(CombinedData, NationalGdpData, how='left' )


#Fill the miss data by the mean of each country
CountryMeans = CombinedData.groupby('Entity')[[LifeExpShort,GdpShort, NationalGdpShort]].mean().reset_index().values
IndexDict = dict()
for item in CountryMeans:
  IndexDict[item[0]] = item[1:]
CombinedData[LifeExpShort] = CombinedData[LifeExpShort].fillna(CombinedData['Entity'].apply(lambda x: IndexDict.get(x)[0]))
CombinedData[GdpShort] = CombinedData[GdpShort].fillna(CombinedData['Entity'].apply(lambda x: IndexDict.get(x)[1]))
CombinedData[NationalGdpShort] = CombinedData[NationalGdpShort].fillna(CombinedData['Entity'].apply(lambda x: IndexDict.get(x)[2]))
CombinedData.to_csv('./data/test.csv')

'''
Following task
'''
def GdpVSExp():
    for country in all_CountryList:
        plt.scatter(CombinedData[CombinedData['Entity']==country][GdpShort],
                    CombinedData[CombinedData['Entity']==country][LifeExpShort], 
                    s=5,
                    label=country)
    plt.title('Scatter Plot of GDP per Capita vs. Life Expectancy')
    plt.xlabel('GDP per Capita')
    plt.ylabel('Life Expectancy')
    plt.legend(title='Country')
    plt.grid(True)
    plt.show()   


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
  Gdp_Mean, Gdp_std = DataSet[NationalGdpShort].mean(), DataSet[NationalGdpShort].std()
  High_LifeExp_threshold = LifeExp_Mean + LifeExp_std
  Low_Gdp_threshold = Gdp_Mean - 0.1*Gdp_std
  DataSet_Mean = DataSet.groupby("Entity")[[LifeExpShort,NationalGdpShort]].mean()
  print(DataSet_Mean[LifeExpShort].max())
  ans = DataSet_Mean[(DataSet_Mean[LifeExpShort] >= High_LifeExp_threshold) 
                            & (DataSet_Mean[NationalGdpShort] <= Low_Gdp_threshold)]
  print(ans)
  #Visualization
  plt.figure()
  for country in DataSet_Mean.index:
      if country in ans.index:  
        plt.scatter(
                    DataSet_Mean.loc[country][NationalGdpShort]/1e10, 
                    DataSet_Mean.loc[country][LifeExpShort], 
                    s=15,
                    label=country)
        plt.annotate(country, (DataSet_Mean.loc[country][NationalGdpShort]/1e10, DataSet_Mean.loc[country][LifeExpShort]))
      else:
               plt.scatter(
                  DataSet_Mean.loc[country][NationalGdpShort]/1e10, 
                  DataSet_Mean.loc[country][LifeExpShort], 
                  s=5,
                  label=country)
               
  line1 = plt.axhline(High_LifeExp_threshold, color='blue', label='High_LifeExp_threshold')
  line2 = plt.axvline(Low_Gdp_threshold/1e10, color='red', label='Low_Gdp_threshold')
  plt.legend(handles=[line1, line2])
  plt.xlabel(NationalGdpShort+'e^10')
  plt.ylabel(LifeExpShort) 

  plt.show()
     
  
def task3(Dataset:pd.DataFrame, threshold = 0.1):
  """
  strong economy has strong life expectancy
  """
  StrongEconomy = Dataset.sort_values(by=NationalGdpShort, ascending=False)
  print(StrongEconomy)



if __name__ == '__main__':
# Uncomment when executing task
  # GdpVSExp()

  # task1(CombinedData)
  task2(CombinedData)
  # task3(CombinedData)
