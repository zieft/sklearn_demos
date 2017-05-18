from __future__ import division, print_function, unicode_literals

import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import sklearn.svm
import sklearn.datasets
import sklearn.preprocessing
import sklearn.pipeline

np.random.seed(42)
PROJECT_ROOT_DIR = os.getcwd()

plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12


def save_fig(fig_id, tight_layout=True):
    """
    Save the previous plotted image as .png file.
    :param fig_id: name of the image
    :param tight_layout: Automatically adjust subplot parameters to give specified padding.
    :return: 
    """
    assert isinstance(fig_id, str), "Must be a string."
    if not os.path.exists(os.path.join(PROJECT_ROOT_DIR, "images")):
        os.makedirs(os.path.join(PROJECT_ROOT_DIR, "images"))
    path = os.path.join(PROJECT_ROOT_DIR, "images", fig_id + ".png")
    print("Saving figure: ", "<", fig_id, ">")
    if tight_layout:
        plt.tight_layout()
        plt.savefig(path, format="png", dpi=300)


# # Large margin classification
iris = sklearn.datasets.load_iris()
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = iris["target"]

setosa_or_versicolor = (y == 0) | (y == 1)  # | 按位或运算
# 在布尔值的运算中，"按位或"就是"或"的意思
# 生成只含有setosa和versicolor两种Iris的训练集
X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]

# SVM Classivfier model
svm_clf = sklearn.svm.SVC(kernel="linear", C=float("inf"))
svm_clf.fit(X, y)

# # Bad models
x0 = np.linspace(0, 5.5, 200)
# 下面是三条手动设定的直线
pred_1 = 5 * x0 - 20
pred_2 = x0 - 1.8
pred_3 = 0.1 * x0 + 0.5


def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    """
    plot the decision boundary as well as the upper- and lower-gutter and the supoort vectors.
    At the decision boundary, w0*x0 + w1*x1 + b = 0 
    => decision_boundary (x1) = -w0/w1 * x0 - b/w1
    :param svm_clf: an instance of sklearn.svm.SVC
    :param xmin: min value of x0
    :param xmax: max value of x0
    :return: 
    """
    w = svm_clf.coef_[0]  # [0]将.coef_最外层的大括号去掉
    b = svm_clf.intercept_[0]  # 获得直线的截距

    # At the decision boundary, w0*x0 + w1*x1 + b = 0 这是一条直线
    # => x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0] / w[1] * x0 - b / w[1]

    # 生成决策边界上下两条虚线
    margin = 1 / w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    svs = svm_clf.support_vectors_  # support_vectors_: 真正影响决策边界的数据子集
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')  # s点的大小， facecolors点的颜色
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)


plt.figure(figsize=(12, 2.7))

# 绘制手动生成的"坏"决策边界
plt.subplot(121)
plt.plot(x0, pred_1, "g--", linewidth=2)
plt.plot(x0, pred_2, "m-", linewidth=2)
plt.plot(x0, pred_3, "r-", linewidth=2)
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "b-", label="Iris-Versicolor")
plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "y-", label="Iris-Setosa")
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="upper left", fontsize=14)
plt.axis([0, 5.5, 0, 2])

# 绘制由支持向量生成的决策边界
plt.subplot(122)
plot_svc_decision_boundary(svm_clf, 0, 5.5)
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "b-")
plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "y-")
plt.xlabel("Petal length", fontsize=14)
plt.axis([0, 5.5, 0, 2])

save_fig("large_margin_classification_plot")

# # Sensitivity to feature scales
Xs = np.array([[1, 50], [5, 20], [3, 80], [5, 60]]).astype(np.float64)
ys = np.array([0, 0, 1, 1])
svm_clf = sklearn.svm.SVC(kernel="linear", C=100)
svm_clf.fit(Xs, ys)

plt.figure(figsize=(12, 3.2))
plt.subplot(121)
plt.plot(Xs[:, 0][ys == 1], Xs[:, 1][ys == 1], "bo")
plt.plot(Xs[:, 0][ys == 0], Xs[:, 1][ys == 0], "ms")
plot_svc_decision_boundary(svm_clf, 0, 6)
plt.xlabel("$x_0$", fontsize=20)
plt.ylabel("$x_1$ ", fontsize=20, rotation=0)
plt.title("Unscaled", fontsize=16)
plt.axis([0, 6, 0, 90])

scaler = sklearn.preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(Xs)
svm_clf.fit(X_scaled, ys)

plt.subplot(122)
plt.plot(X_scaled[:, 0][ys == 1], X_scaled[:, 1][ys == 1], "bo")
plt.plot(X_scaled[:, 0][ys == 0], X_scaled[:, 1][ys == 0], "ms")
plot_svc_decision_boundary(svm_clf, -2, 2)
plt.xlabel("$x_0$", fontsize=20)
plt.title("Scaled", fontsize=16)
plt.axis([-2, 2, -2, 2])

save_fig("sensitivity_to_feature_scales_plot")

# # Sensitivity to outliers
X_outliers = np.array([[3.4, 1.3], [3.2, 0.8]])
y_outliers = np.array([0, 0])
Xo1 = np.concatenate([X, X_outliers[:1]], axis=0)
yo1 = np.concatenate([y, y_outliers[:1]], axis=0)
Xo2 = np.concatenate([X, X_outliers[1:]], axis=0)
yo2 = np.concatenate([y, y_outliers[1:]], axis=0)

svm_clf2 = sklearn.svm.SVC(kernel="linear", C=10 ** 9)  # float("inf"))
# 这里的参数C设定得非常大，因此对outlier非常敏感。
svm_clf2.fit(Xo2, yo2)

plt.figure(figsize=(12, 2.7))

plt.subplot(121)
plt.plot(Xo1[:, 0][yo1 == 1], Xo1[:, 1][yo1 == 1], "bs")
plt.plot(Xo1[:, 0][yo1 == 0], Xo1[:, 1][yo1 == 0], "yo")
plt.text(0.3, 1.0, "Impossible!", fontsize=24, color="red")
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.annotate("Outlier",
             xy=(X_outliers[0][0], X_outliers[0][1]),
             xytext=(2.5, 1.7),
             ha="center",
             arrowprops=dict(facecolor='black', shrink=0.1),
             fontsize=16,
             )
plt.axis([0, 5.5, 0, 2])

plt.subplot(122)
plt.plot(Xo2[:, 0][yo2 == 1], Xo2[:, 1][yo2 == 1], "bs")
plt.plot(Xo2[:, 0][yo2 == 0], Xo2[:, 1][yo2 == 0], "yo")
plot_svc_decision_boundary(svm_clf2, 0, 5.5)
plt.xlabel("Petal length", fontsize=14)
plt.annotate("Outlier",
             xy=(X_outliers[1][0], X_outliers[1][1]),
             xytext=(3.2, 0.08),
             ha="center",
             arrowprops=dict(facecolor='black', shrink=0.1),
             fontsize=16,
             )
plt.axis([0, 5.5, 0, 2])

save_fig("sensitivity_to_outliers_plot")

# # Large margin vs margin violations
# 生成一个只区分是不是Iris-Virginica的训练集
iris = sklearn.datasets.load_iris()
X = iris["data"][:, (2, 3)]
y = (iris["target"] == 2).astype(np.float64)  # np.float64将布尔值转换成数值

scaler = sklearn.preprocessing.StandardScaler()
svm_clf1 = sklearn.svm.LinearSVC(C=100, loss="hinge")
svm_clf2 = sklearn.svm.LinearSVC(C=1, loss="hinge")

scaled_svm_clf1 = sklearn.pipeline.Pipeline(
    [
        ("scaler", scaler),
        ("linear_svc", svm_clf1),
    ]
)

scaled_svm_clf2 = sklearn.pipeline.Pipeline(
    [
        ("scaler", scaler),
        ("linear_svc", svm_clf2),
    ]
)

scaled_svm_clf1.fit(X, y)
scaled_svm_clf2.fit(X, y)

print(scaled_svm_clf1.predict([[5.5, 1.7]]))
print(scaled_svm_clf2.predict([[5.5, 1.7]]))

# Convert to unscaled parameters
# 将正则化的数据非正则化的方法
b1 = svm_clf1.decision_function([-scaler.mean_ / scaler.scale_])
b2 = svm_clf2.decision_function([-scaler.mean_ / scaler.scale_])
w1 = svm_clf1.coef_[0] / scaler.scale_
w2 = svm_clf2.coef_[0] / scaler.scale_
svm_clf1.intercept_ = np.array([b1])
svm_clf2.intercept_ = np.array([b2])
svm_clf1.coef_ = np.array([w1])
svm_clf2.coef_ = np.array([w2])

# Find support vectors (LinearSVC does not do this automatically)
# 寻找支持向量的方法
t = y * 2 - 1
support_vectors_idx1 = (t * (X.dot(w1) + b1) < 1).ravel()
support_vectors_idx2 = (t * (X.dot(w2) + b2) < 1).ravel()
svm_clf1.support_vectors_ = X[support_vectors_idx1]
svm_clf2.support_vectors_ = X[support_vectors_idx2]

plt.figure(figsize=(12, 3.2))
plt.subplot(121)
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^", label="Iris-Virginica")
plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bs", label="Iris-Versicolo or Setosa")
plot_svc_decision_boundary(svm_clf1, 4, 6)
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="upper left", fontsize=14)
plt.title("$C = {}$".format(svm_clf1.C), fontsize=16)
plt.axis([4, 6, 0.8, 2.8])

plt.subplot(122)
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^")
plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bs")
plot_svc_decision_boundary(svm_clf2, 4, 6)
plt.xlabel("Petal length", fontsize=14)
plt.title("$C = {}$".format(svm_clf2.C), fontsize=16)
plt.axis([4, 6, 0.8, 2.8])

save_fig("regularization_plot")

# # None-linear classification
# plot two images to show the effect by adding features
X1D = np.linspace(-4, 4, 9).reshape(-1, 1)
X2D = np.c_[X1D, X1D ** 2]
y = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0])

plt.figure(figsize=(11, 4))

plt.subplot(121)
plt.grid(True, which="both")
plt.axhline(y=0, color="k")
plt.plot(X1D[:, 0][y == 0], np.zeros(4), "bs")
plt.plot(X1D[:, 0][y == 1], np.zeros(5), "g^")
plt.gca().get_yaxis().set_ticks([])
plt.xlabel("$x_1$", fontsize=20)
plt.axis([-4.5, 4.5, -0.2, 0.2])

plt.subplot(122)
plt.grid(True, which="both")
plt.axhline(y=0, color="k")
plt.axvline(x=0, color="k")
plt.plot(X2D[:, 0][y == 0], X2D[:, 1][y == 0], "bs")
plt.plot(X2D[:, 0][y == 1], X2D[:, 1][y == 1], "g^")
plt.xlabel("$x_1$", fontsize=20)
plt.ylabel("$x_2$", fontsize=20)
plt.gca().get_yaxis().set_ticks([0, 4, 8, 12, 16])
plt.plot([-4.5, 4.5], [6.5, 6.5], "r--", linewidth=3)
plt.axis([-4.5, 4.5, -1, 17])

plt.subplots_adjust(right=1)

save_fig("higher_dimensions_plot", tight_layout=False)

# Test Nonlinear Classification using SVC and moons dataset
X, y = sklearn.datasets.make_moons(n_samples=100, noise=0.15, random_state=42)


def plot_dataset(X, y, axes):
    """
    
    :param X: features
    :param y: targets
    :param axes: range of the plot
    :return: 
    """
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bs")
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^")
    plt.axis(axes)
    plt.grid(True, which="both")
    plt.xlabel("$x_1$", fontsize=20)
    plt.ylabel("$x_2$", fontsize=20, rotation=0)


plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
save_fig("Moons datasets")

# Adding features by creating a Pipeline containing a PolynomialFeatures transformer
# followed by a StandardScaler and a LinearSVC.
polynomial_svm_clf = sklearn.pipeline.Pipeline(
    [
        ("poly_features", sklearn.preprocessing.PolynomialFeatures(degree=3)),
        ("scaler", sklearn.preprocessing.StandardScaler()),
        ("svm_clf", sklearn.svm.LinearSVC(C=10, loss="hinge"))
    ]
)
polynomial_svm_clf.fit(X, y)


def plot_predictions(clf, axes):
    """
    
    :param clf: 
    :param axes: 
    :return: 
    """
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)


plot_predictions(polynomial_svm_clf, [-1.5, 2.5, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])

save_fig("moons_polynomial_svc_plot")

# Now, instead of using PolynomialFeatures plus LinearSVC
# We use Polynomial Kernel in SVC directly
poly_kernel_svm_clf = sklearn.pipeline.Pipeline(
    [
        ("scaler", sklearn.preprocessing.StandardScaler()),
        ("svm_clf", sklearn.svm.SVC(kernel="poly", degree=3, coef0=1, C=5)),
    ]
)
poly100_kernel_svm_clf = sklearn.pipeline.Pipeline(
    [
        ("scaler", sklearn.preprocessing.StandardScaler()),
        ("svm_clf", sklearn.svm.SVC(kernel="poly", degree=10, coef0=100, C=5))
    ]
)
poly_kernel_svm_clf.fit(X, y)
poly100_kernel_svm_clf.fit(X, y)

plt.figure(figsize=(11, 4))

axes = [-1.5, 2.5, -1, 1.5]
plt.subplot(121)
plot_predictions(poly_kernel_svm_clf, axes)
plot_dataset(X, y, axes)
plt.title("$d=3, r=1, C=5$", fontsize=18)

plt.subplot(122)
plot_predictions(poly100_kernel_svm_clf, axes)
plot_dataset(X, y, axes)
plt.title("$d=10, r=100, C=5$", fontsize=18)

save_fig("moons_kernelized_polynomial_svc_plot")
