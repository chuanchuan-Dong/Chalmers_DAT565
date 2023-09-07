import pandas as pd
import numpy as np
import matplotlib.pyplot as plt    
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

'''
Read data & Process data
'''
Hemnet_data = pd.read_csv("data/hemnet.csv")
plt.figure()
plt.subplot(1,2,1)
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



x = FilterHemnet_data['Living_area'].values.reshape(-1,1)
y = FilterHemnet_data['Selling_price'].values

'''
Start model
'''
model = LinearRegression()
model.fit(x, y)

print("Intercept of regression line:",model.intercept_)
print("slope of regression line:",model.coef_[0])
plt.plot([x.min(),x.max()], [model.predict([[x.min()]]), model.predict([[x.max()]])], label='Regression Line', color='green')

x_predict = [[100],[150],[200]]
PreditPoint = model.predict(x_predict)
plt.scatter([100,150,200], PreditPoint, label='Predict point', color='red')
for i, pred_num in enumerate(PreditPoint):
    plt.annotate(f'({x_predict[i][0]},{pred_num:.2f})', (x[i], PreditPoint[i]), textcoords='offset points', xytext=(0,10))
plt.legend()


#draw residual point
y_pred = model.predict(x)
residuals = y - y_pred

plt.subplot(1,2,2)
plt.scatter(x, residuals, label='Residuals')
plt.axhline(0, color='red', linestyle='--', label='Zero Residuals Line')
plt.xlabel('Living area')
plt.ylabel('Residuals')
plt.title('Residual Plot')            


plt.legend()
plt.show()