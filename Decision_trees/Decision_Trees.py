from __future__ import division, print_function, unicode_literals

import numpy as np
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
import sklearn
import sklearn.datasets
import sklearn.tree

np.random.seed(42)
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
PROJECT_ROOT_DIR = os.getcwd()


def image_path(fig_id):
    """
    get fig_id. If not exist, create.
    :param fig_id: 
    :return: 
    """
    assert isinstance(fig_id, str), "Must be a string."
    if not os.path.exists(os.path.join(PROJECT_ROOT_DIR, "images")):
        os.makedirs(os.path.join(PROJECT_ROOT_DIR, "images"))
    return os.path.join(PROJECT_ROOT_DIR, "images", fig_id)


def save_fig(fig_id, tight_layout=True):
    """
    Save the previous plotted image as .png file.
    :param fig_id: name of the image
    :param tight_layout: Automatically adjust subplot parameters to give specified padding.
    :return: 
    """
    assert isinstance(fig_id, str), "Must be a string."
    print("Saving figure: ", "<", fig_id, ">")
    if tight_layout:
        plt.tight_layout()
        plt.savefig(image_path(fig_id) + ".png", format="png", dpi=300)


# # Training and Visualizing
iris = sklearn.datasets.load_iris()
X = iris.data[:, 2:]  # = [:, (2, 3)]
y = iris["target"]

tree_clf = sklearn.tree.DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X, y)

sklearn.tree.export_graphviz(
    tree_clf,
    out_file=image_path("iris_tree.dot"),
    feature_names=iris.feature_names[2:],
    class_names=iris.target_names,
    rounded=True,
    filled=True
)


def plot_decision_boundary(clf, X, y, axes=[0, 7.5, 0, 3], iris=True, legend=False, plot_training=True):
    """
    
    :param clf: 
    :param X: 
    :param y: 
    :param axes: 
    :param iris: 
    :param Legend: 
    :param plot_training: 
    :return: 
    """
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = matplotlib.colors.ListedColormap(["#fafab0", "#9898ff", "#a0faa0"])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap, linewidth=10)
    if not iris:
        custom_cmap2 = matplotlib.colors.ListedColormap(["#7d7d58", "#4c4c7f", "#507d50"])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    if plot_training:
        plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "yo", label="Iris-Setosa")
        plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs", label="Iris-Versicolor")
        plt.plot(X[:, 0][y == 2], X[:, 1][y == 2], "g^", label="Iris-Virginica")
        plt.axis(axes)
    if iris:
        plt.xlabel("Petal length", fontsize=14)
        plt.ylabel("Petal width", fontsize=14)
    else:
        plt.xlabel(r"$x_1$", fontsize=18)
        plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
    if legend:
        plt.legend(loc="lower right", fontsize=14)


plt.figure(figsize=(8, 4))
plot_decision_boundary(tree_clf, X, y)
plt.plot([2.45, 2.45], [0, 3], "k-", linewidth=2)
plt.plot([2.45, 7.5], [1.75, 1.75], "k--", linewidth=2)
plt.plot([4.95, 4.95], [0, 1.75], "k:", linewidth=2)
plt.plot([4.85, 4.85], [1.75, 3], "k:", linewidth=2)
plt.text(1.40, 1.0, "Depth=0", fontsize=15)
plt.text(3.2, 1.80, "Depth=1", fontsize=13)
plt.text(4.05, 0.5, "(Depth=2)", fontsize=11)

save_fig("decision_tree_decision_boundaries_plot")
