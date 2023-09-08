from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.colors import ListedColormap
from sklearn import datasets, neighbors
from sklearn.inspection import DecisionBoundaryDisplay

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import some data to play with
iris = datasets.load_iris()
print(iris)
exit(0)
# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target

# Create color maps
cmap_light = ListedColormap(["white", "cyan", "cornflowerblue"])
cmap_bold = ["blue", "c", "darkblue"]


for weights in ["uniform", "distance"]:
    for i, n_neighbors in enumerate(range(1,90,10)):
        # we create an instance of Neighbours Classifier and fit the data.
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
        clf.fit(X, y)

        ax = plt.subplot(3,3,i+1)
        DecisionBoundaryDisplay.from_estimator(
            clf,
            X,
            cmap=cmap_light,
            ax=ax,
            response_method="predict",
            plot_method="pcolormesh",
            xlabel=iris.feature_names[0],
            ylabel=iris.feature_names[1],
            shading="auto",
        )

        # Plot also the training points
        sns.scatterplot(
            x=X[:, 0],
            y=X[:, 1],
            hue=iris.target_names[y],
            palette=cmap_bold,
            alpha=1.0,
            edgecolor="black",
        )
        plt.title(
            "3-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights)
        )
    plt.show()
exit(0)
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