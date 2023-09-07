from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt


IrisDict = load_iris()

print(IrisDict.keys())

TargetDict = dict()
for i, target in enumerate(IrisDict['target_names']):
    TargetDict[i] = target
    
print(TargetDict)

IrisData, IrisTarget = pd.DataFrame(IrisDict['data'], columns=IrisDict['feature_names']), pd.DataFrame(IrisDict['target'], columns=['target'])
IrisData = pd.concat([IrisData, IrisTarget], axis=1)

print(IrisData.head)

TrainData, TestData = train_test_split(IrisData, train_size=0.7)

print(TrainData.shape)
print(TestData.shape)

model = LogisticRegression()

model.fit(TrainData[TrainData.columns[:-2]].values, TrainData[TrainData.columns[-1]])
y_true = TestData['target']
y_pred = model.predict(TestData[TestData.columns[:-2]])

cm = confusion_matrix(y_true, y_pred)
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=IrisDict['target_names'])
display.plot()
plt.show()

print(cm)

# print(TrainData.shape)
# print(TestData.shape)


# print(IrisData.shape)