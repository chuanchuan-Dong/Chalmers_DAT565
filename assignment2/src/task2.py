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

<<<<<<< HEAD
def PreprocessIris():
    IrisDict = load_iris()

    FeatureNames = IrisDict['feature_names']

    TargetDict = dict()
    for i, target in enumerate(IrisDict['target_names']):
        TargetDict[i] = target

    IrisData, IrisTarget = pd.DataFrame(IrisDict['data'], columns=FeatureNames), pd.DataFrame(IrisDict['target'], columns=['target'])
    IrisData = pd.concat([IrisData, IrisTarget], axis=1)
    return IrisDict, IrisData, FeatureNames, TargetDict

def KNN(dataset:pd.DataFrame, feature_names, target_names):
    x, y = dataset[dataset.columns[:-2]].values, dataset[dataset.columns[-1]]

    # Create color maps
    cmap_light = ListedColormap(["white", "cyan", "cornflowerblue"])
    cmap_bold = ["blue", "c", "darkblue"]


    for weights in ["uniform", "distance"]:
        for i, n_neighbors in enumerate(range(1,90,10)):
            # we create an instance of Neighbours Classifier and fit the data.
            clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
            clf.fit(x, y)
=======
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
>>>>>>> c866b13eeb196e0125bd1853b041aafb8f3073c9

            ax = plt.subplot(3,3,i+1)
            DecisionBoundaryDisplay.from_estimator(
                clf,
                x,
                cmap=cmap_light,
                ax=ax,
                response_method="predict",
                plot_method="pcolormesh",
                xlabel=feature_names[0],
                ylabel=feature_names[1],
                shading="auto",
            )

            # Plot also the training points
            sns.scatterplot(
                x=x[:, 0],
                y=x[:, 1],
                hue=target_names[y],
                palette=cmap_bold,
                alpha=1.0,
                edgecolor="black",
            )
            plt.title(
                "3-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights)
            )
        plt.show()


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




if __name__ ==  '__main__':
    IrisDict,IrisData, FeatureNames, TargetDict = PreprocessIris()
    TrainData, TestData = train_test_split(IrisData, train_size=0.7)
    
    # print(TrainData.shape)
    # print(TestData.shape)

    model = LogisticRegression()

    model.fit(TrainData[TrainData.columns[:-2]].values, TrainData[TrainData.columns[-1]])
    y_true = TestData['target']
    y_pred = model.predict(TestData[TestData.columns[:-2]])

    cm = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=IrisDict['target_names'])
    
    print(TargetDict.values())
    
    KNN(IrisData, FeatureNames, list(TargetDict.values()))
    
    print(cm)
    drawPlot(IrisData)