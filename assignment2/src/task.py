import pandas as pd
import numpy as np
import matplotlib.pyplot as plt    
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
'''
Read data & Process data
'''
Hemnet_data = pd.read_csv("data/hemnet.csv")
plt.figure()
plt.scatter(Hemnet_data['Living_area'], Hemnet_data['Selling_price'], label=' Outlier data')
plt.xlabel('Living Area')
plt.ylabel('Selling Price')
plt.title('Scatter Plot of Living Area vs Selling Price')


#drop null 
Hemnet_data.dropna(subset=['Living_area', 'Selling_price'], inplace=True)
#Drop outlier data, Z-score
SellPerArea_data = Hemnet_data['Selling_price'] / Hemnet_data['Living_area']
SellPerArea_mean = SellPerArea_data.mean()
SellPerArea_std = SellPerArea_data.std()
SellPerArea_z_score = (SellPerArea_data - SellPerArea_mean) / SellPerArea_std
z_score_threhold = 1.5
FilterHemnet_data = Hemnet_data[np.abs(SellPerArea_z_score) < z_score_threhold]
plt.scatter(FilterHemnet_data['Living_area'], FilterHemnet_data['Selling_price'],label='After data cleaning')
plt.legend()
#plt.show()

#split data to traing set and test set
x = FilterHemnet_data['Living_area'].values.reshape(-1,1)
y = FilterHemnet_data['Selling_price'].values
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
print(x_train)