from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.colors import ListedColormap
from sklearn import neighbors
from sklearn.inspection import DecisionBoundaryDisplay

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    x, y = dataset[dataset.columns[:-3]].values, dataset[dataset.columns[-1]]

    cmap_light = ListedColormap(["salmon", 'palegreen', 'moccasin'])
    cmap_bold = ["red", "green", "brown"]
    
    knn_dict = dict()
    knn_dict['uniform'], knn_dict['distance'] = dict(), dict()

    for weights in ["uniform", "distance"]:
        best_accuracy = 0
        for i, n_neighbors in enumerate(range(1,90,10)):
            clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
            clf.fit(x, y)
            temp_accuracy = (clf.predict(x) == y).sum() / len(y) * 100
            if temp_accuracy > best_accuracy:
                knn_dict[weights]['k'] = n_neighbors
                knn_dict[weights]['pred_y'] = clf.predict(x)
                knn_dict[weights]['y'] = y
                knn_dict[weights]['accuracy'] = temp_accuracy
                best_accuracy = temp_accuracy
            # print(clf.predict(x) == y)
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
            
            sns.scatterplot(
                x=x[:, 0],
                y=x[:, 1],
                hue = [TargetDict[i] for i in y],
                palette=cmap_bold,
                alpha=1.0,
                edgecolor="black",
            )
            plt.title(
                "k = %i" % (n_neighbors)
            )
        # print(best_accuracy)
        if weights == 'uniform':
            plt.suptitle("Scatter plot of KNN on uniform")
        else:
            plt.suptitle("Scatter plot of KNN on distance")
        plt.subplots_adjust(hspace=0.5)    
        plt.show()
    for key in knn_dict:
        cm = confusion_matrix(knn_dict[key]['y'], knn_dict[key]['pred_y'])
        display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names).plot(cmap='Blues')
        best_k = knn_dict[key]['k']
        display.ax_.set_title(f'{key}-KNN\'s confusion matrix (k={best_k})')
        plt.show()
        print(knn_dict[key]['k'])


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

def LR(dataset):
    TrainData, TestData = train_test_split(dataset, train_size=0.6)
    

    model = LogisticRegression(max_iter=1000)

    model.fit(TrainData[TrainData.columns[:-1]].values, TrainData[TrainData.columns[-1]].values)
    y_true = TestData['target']
    y_pred = model.predict(TestData[TestData.columns[:-1]].values)

    cm = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=IrisDict['target_names']).plot(cmap='Blues')
    display.ax_.set_title('Logistic Regression\'s confusion matrix')
    plt.show()
    print(cm)
    

if __name__ ==  '__main__':
    global TargetDict
    global IrisData
    IrisDict,IrisData, FeatureNames, TargetDict = PreprocessIris()
    
    # print(TargetDict.values())
    
    KNN(IrisData, FeatureNames, list(TargetDict.values()))
    LR(IrisData)
    drawPlot(IrisData)