from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt


IrisDict = load_iris()

print(IrisDict.keys())

FeatureNames = IrisDict['feature_names']

TargetDict = dict()
for i, target in enumerate(IrisDict['target_names']):
    TargetDict[i] = target
    
print(TargetDict)

IrisData, IrisTarget = pd.DataFrame(IrisDict['data'], columns=FeatureNames), pd.DataFrame(IrisDict['target'], columns=['target'])
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
# display.plot()
# plt.show()

print(cm)


def drawPlot(dataset:pd.DataFrame):
    figure = plt.figure()
    for i in range(len(FeatureNames)):
        for j in range(len(FeatureNames)):
            if i == j:
                plt.subplot(4,4,5 * i + 1)
                plt.text(0.25,0.5,dataset.columns[i])
            else:
                plt.subplot(4,4,4*i+j+1)
                attribute_x, attribute_y = dataset.columns[i], dataset.columns[j]
                l = []
                for key in TargetDict.keys():
                    data_x, data_y = dataset[(dataset['target'] == key)][attribute_x], dataset[(dataset['target'] ==key)][attribute_y]
                    fig = plt.scatter(data_x, data_y, s=1, label=key)
                    plt.xlabel([attribute_x])
                    plt.ylabel([attribute_y])
                    l.append(fig)
                # plt.legend(handles=l, loc='upper left', frameon=False)
            plt.xticks([])
            plt.yticks([])
    
    font_legend = {'size': 13}
    figure.legend(handles = l, labels = [TargetDict[i] for i in TargetDict.keys()], loc='lower right', prop=font_legend)
    plt.suptitle('The Iris Data Distribution of All feature')            
    plt.show()

drawPlot(IrisData)
# print(TrainData.shape)
# print(TestData.shape)


# print(IrisData.shape)