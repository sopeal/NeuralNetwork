import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn import neighbors
import urllib.request

def main():

    # Seeds Dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"
    raw_data = urllib.request.urlopen(url)
    seeds_dataset = np.loadtxt(raw_data)
    seeds_data = seeds_dataset[:, 0:7]
    seeds_target = seeds_dataset[:, 7]

    # Iris Dataset
    iris = datasets.load_iris()

    # x = iris.data[0:150:2]
    # y = iris.target[0:150:2]
    # x_train = iris.data[1:149:2]
    # y_train = iris.target[1:149:2]

    x = seeds_data[0:210:2]
    y = seeds_target[0:210:2]

    x_train = seeds_data[1:210:2]
    y_train = seeds_target[1:210:2]

    x_train = x
    y_train = y

    n_neighours = 100

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    for weight in ['uniform', 'distance']:
        clf = neighbors.KNeighborsClassifier(n_neighours, weights=weight)
        clf.fit(x, y)

        x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1

        Z = clf.predict(x_train)
        accuracy = clf.score(x_train, y_train) * 100

        plt.figure()
        #plt.scatter(x_train[:, 0], x_train[:, 1], c=Z, cmap=cmap_light)
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title("3-Class classification (k = %i, weights = '%s')" % (n_neighours, weight))
        plt.show()
        print("Global accuracy: " + "{0:.2f}".format(accuracy) + " %")


if __name__ == "__main__":
    main()
