from __future__ import division, print_function, unicode_literals

import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import sklearn.svm
import sklearn.datasets
import sklearn.preprocessing
import sklearn.pipeline
import sklearn.linear_model
import sklearn.preprocessing
import time
import sklearn.base

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
# decision_function: Distance of the samples X to the separating hyperplane.
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
X2D = np.c_[X1D, X1D ** 2]  # 添加特征
y = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0])

plt.figure(figsize=(11, 4))

plt.subplot(121)
plt.grid(True, which="both")
plt.axhline(y=0, color="k", linewidth=1)  # 加粗横轴
plt.plot(X1D[:, 0][y == 0], np.zeros(4), "bs")
plt.plot(X1D[:, 0][y == 1], np.zeros(5), "g^")
plt.gca().get_yaxis().set_ticks([])  # 去掉y轴
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
plt.gca().get_yaxis().set_ticks([0, 4, 8, 12, 16])  # 设定y轴标尺
plt.plot([-4.5, 4.5], [6.5, 6.5], "r--", linewidth=3)
plt.axis([-4.5, 4.5, -1, 17])

plt.subplots_adjust(right=1)

save_fig("higher_dimensions_plot")

# Test Nonlinear Classification using SVC and moons dataset
X, y = sklearn.datasets.make_moons(n_samples=100, noise=0.15, random_state=42)


def plot_dataset(X, y, axes):
    """
    
    :param X: features
    :param y: targets
    :param axes: range of the plot
    :return: 
    """
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bs")  # 绘制所有y=0对应的点
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^")
    plt.axis(axes)
    plt.grid(True, which="both")
    plt.xlabel("$x_1$", fontsize=20)
    plt.ylabel("$x_2$", fontsize=20, rotation=0)


plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
save_fig("Moons_datasets")

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
    x0s = np.linspace(axes[0], axes[1], 100)  # 按照输入的取值范围生成x0定义域
    x1s = np.linspace(axes[2], axes[3], 100)  # x1的定义域
    x0, x1 = np.meshgrid(x0s, x1s)  # x0:第0轴重复100遍，x1:第1轴重复100遍
    # np.meshgrid() Return coordinate matrices from coordinate vectors.
    X_ = np.c_[x0.ravel(), x1.ravel()]  # 生成定义范围内的所有的（10000个）点
    y_pred = clf.predict(X_).reshape(x0.shape)
    y_decision = clf.decision_function(X_).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)  # 绘制等高线图
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


# Adding Similarity Features
def gaussian_rbf(x, landmark, gamma):
    """
    Gaussian Radial Basis Function (RBF) Equation 5-1
    :param x: feature to be transformed (must be a vector, using .reshape(-1,1))
    :param landmark: landmark selected at the location of each and every instance in the dataset.
    :param gamma: Increasing gamma makes the bell-shape curve narrower
    :return: new feature with respect with the landmark
    """
    return np.exp(-gamma * np.linalg.norm(x - landmark, axis=1) ** 2)


gamma = 0.3

x1s = np.linspace(-4.5, 4.5, 200).reshape(-1, 1)
x2s = gaussian_rbf(x1s, -2, gamma)
x3s = gaussian_rbf(x1s, 1, gamma)

# identical
# x1s1 = np.linspace(-4.5, 4.5, 200).reshape(1, -1)
# x2s1 = gaussian_rbf(x1s, -2, gamma)
# x3s1 = gaussian_rbf(x1s, 1, gamma)

XK = np.c_[gaussian_rbf(X1D, -2, gamma), gaussian_rbf(X1D, 1, gamma)]
yk = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0])

plt.figure(figsize=(11, 4))

plt.subplot(121)
plt.grid(True, which="both")
plt.axhline(y=0, color="k")
plt.scatter(x=[-2, 1], y=[0, 0], s=150, alpha=0.5, c="red")
plt.plot(X1D[:, 0][yk == 0], np.zeros(4), "bs")
plt.plot(X1D[:, 0][yk == 1], np.zeros(5), "g^")
plt.plot(x1s, x2s, "g--")
plt.plot(x1s, x3s, "b:")
plt.gca().get_yaxis().set_ticks([0, 0.25, 0.5, 0.75, 1])
plt.xlabel("$x_1$", fontsize=20)
plt.ylabel("Similarity", fontsize=14)
plt.annotate(  # 在图像中加箭头
    "$\mathbf{x}$",
    xy=(X1D[3, 0], 0),  # Length 2 sequence specifying the *(x,y)* point to annotate
    xytext=(-0.5, 0.2),
    ha="center",
    arrowprops=dict(facecolor="black", shrink=0.1),
    fontsize=18
)
plt.text(-2, 0.9, "$x_2$", ha="center", fontsize=20)
plt.text(1, 0.9, "$x_3$", ha="center", fontsize=20)
plt.axis([-4.5, 4.5, -0.1, 1.1])

plt.subplot(122)
plt.grid(True, which="both")
plt.axhline(y=0, color="k")
plt.axvline(x=0, color="k")
plt.plot(XK[:, 0][yk == 0], XK[:, 1][yk == 0], "bs")
plt.plot(XK[:, 0][yk == 1], XK[:, 1][yk == 1], "g^")
plt.xlabel("$x_2$", fontsize=20)
plt.ylabel("$x_3$", fontsize=20, rotation=0)
plt.annotate(
    r"$\phi\left(\mathbf{x}\right)$",
    xy=(XK[3, 0], XK[3, 1]),
    xytext=(0.65, 0.50),
    ha="center",
    arrowprops=dict(facecolor="black", shrink=0.1),
    fontsize=18
)
plt.plot([-0.1, 1.1], [0.57, -0.1], "r--", linewidth=3)
plt.axis([-0.1, 1.1, -0.1, 1.1])

# plt.subplots_adjust(right=1)

save_fig("kernel_method_plot")

x1_example = X1D[3, 0]
for landmark in (-2, 1):
    k = gaussian_rbf(np.array([[x1_example]]), np.array([[landmark]]), gamma)
    print("Phi({}, {}) = {}".format(x1_example, landmark, k))

# Gaussian RBF kernel using the SVC class:
rbf_kernel_svm_clf = sklearn.pipeline.Pipeline(
    [
        ("scaler", sklearn.preprocessing.StandardScaler()),
        ("svm_clf", sklearn.svm.SVC(kernel="rbf", gamma=5, C=0.001))
    ]
)
rbf_kernel_svm_clf.fit(X, y)

gamma1, gamma2 = 0.1, 5
C1, C2 = 0.001, 1000
hyperparams = (gamma1, C1), (gamma1, C2), (gamma2, C1), (gamma2, C2)  # tuple of tuples

svm_clfs = []  # 生成四个参数不同的支持向量机分类器，并训练数据
for gamma, C in hyperparams:
    rbf_kernel_svm_clf = sklearn.pipeline.Pipeline(
        [
            ("scaler", sklearn.preprocessing.StandardScaler()),
            ("svm_clf", sklearn.svm.SVC(kernel="rbf", gamma=gamma, C=C))
        ]
    )
    rbf_kernel_svm_clf.fit(X, y)
    svm_clfs.append(rbf_kernel_svm_clf)

plt.figure(figsize=(11, 7))

for i, svm_clf in enumerate(svm_clfs):
    plt.subplot(221 + i)
    plot_predictions(svm_clf, [-1.5, 2.5, -1, 1.5])
    plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
    gamma, C = hyperparams[i]
    plt.title(r"$\gamma = {}, C = {}$".format(gamma, C), fontsize=16)

save_fig("moons_rbf_svc_plot")

# # Regression
# Linear Regression
m = 50
X = 2 * np.random.rand(m, 1)
y = (4 + 3 * X + np.random.randn(m, 1)).ravel()

svm_reg1 = sklearn.svm.LinearSVR(epsilon=1.5)
svm_reg2 = sklearn.svm.LinearSVR(epsilon=0.5)
svm_reg1.fit(X, y)
svm_reg2.fit(X, y)


def find_support_vectors(svm_reg, X, y):
    """
    
    :param svm_reg: An instance of sklearn.svm.SVR(LinearSVR)
    :param X: Training set
    :param y: Targets set
    :return: 
    """
    y_pred = svm_reg.predict(X)
    off_margin = (np.abs(y - y_pred) >= svm_reg.epsilon)
    return np.argwhere(off_margin)  # argwhere这个函数不理解


svm_reg1.support_ = find_support_vectors(svm_reg1, X, y)
svm_reg2.support_ = find_support_vectors(svm_reg2, X, y)

eps_x1 = 1
eps_y_pred = svm_reg1.predict([[eps_x1]])


def plot_svm_regression(svm_reg, X, y, axes):
    """
    
    :param svm_reg: 
    :param X: 
    :param y: 
    :param axes: 
    :return: 
    """
    x1s = np.linspace(axes[0], axes[1], 100).reshape(100, 1)
    y_pred = svm_reg.predict(x1s)
    plt.plot(x1s, y_pred, "k-", linewidth=2, label=r"$\hat{y}$")
    plt.plot(x1s, y_pred + svm_reg.epsilon, "k--")
    plt.plot(x1s, y_pred - svm_reg.epsilon, "k--")
    plt.scatter(X[svm_reg.support_], y[svm_reg.support_], s=180, facecolors="#FFAAAA")
    plt.plot(X, y, "bo")
    plt.xlabel("$x_1$", fontsize=18)
    plt.legend(loc="upper left", fontsize=18)
    plt.axis(axes)


plt.figure(figsize=(9, 4))
plt.subplot(121)
axis_svm_regression = [0, 2, 3, 11]
plot_svm_regression(svm_reg1, X, y, axis_svm_regression)
plt.title("$\epsilon = {}$".format(svm_reg1.epsilon), fontsize=18)
plt.ylabel("$y$", fontsize=18, rotation=0)
plt.annotate(
    "",
    xy=(eps_x1, eps_y_pred),
    xycoords="data",
    xytext=(eps_x1, eps_y_pred - svm_reg1.epsilon),
    textcoords="data",
    arrowprops={"arrowstyle": "<->", "linewidth": 1.5}
)
plt.text(0.91, 5.6, "$\epsilon$", fontsize=20)
plt.subplot(122)
plot_svm_regression(svm_reg2, X, y, axis_svm_regression)
plt.title("$\epsilon = {}$".format(svm_reg2.epsilon), fontsize=18)
save_fig("svm_regression_plot")

# 2nd-degree polynomial kernel SVR
m = 100
X = 2 * np.random.rand(m, 1) - 1
y = (0.2 + 0.1 * X + 0.5 * X ** 2 + np.random.randn(m, 1) / 10).ravel()

svm_poly_reg1 = sklearn.svm.SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
svm_poly_reg2 = sklearn.svm.SVR(kernel="poly", degree=2, C=0.01, epsilon=0.1)
svm_poly_reg1.fit(X, y)
svm_poly_reg2.fit(X, y)

plt.figure(figsize=(9, 4))
plt.subplot(121)
axis_svm_regression_poly = [-1, 1, 0, 1]
plot_svm_regression(svm_poly_reg1, X, y, axis_svm_regression_poly)
plt.title("$degree={}, C={}, \epsilon={}$".format(svm_poly_reg1.degree, svm_poly_reg1.C, svm_poly_reg1.epsilon),
          fontsize=18)
plt.ylabel("$y$", fontsize=18, rotation=0)

plt.subplot(122)
plot_svm_regression(svm_poly_reg2, X, y, axis_svm_regression_poly)
plt.title("$degree={}, C={}, \epsilon={}$".format(svm_poly_reg2.degree, svm_poly_reg2.C, svm_poly_reg2.epsilon),
          fontsize=18)
save_fig("svm_with_polynomial_kernel_plot")

# # Under the hook
iris = sklearn.datasets.load_iris()
X = iris["data"][:, (2, 3)]  # Only focus on the petal length, and width
y = (iris["target"] == 2).astype(np.float64)  # Iris-virginica


def plot_3D_decision_function(ax, w, b, x1_lim=[4, 6], x2_lim=[0.8, 2.8]):
    """
    
    :param ax: 
    :param w: 
    :param b: 
    :param x1_lim: 
    :param x2_lim: 
    :return: 
    """
    x1_in_bounds = (X[:, 0] > x1_lim[0]) & (X[:, 0] < x1_lim[1])
    X_crop = X[x1_in_bounds]
    y_crop = y[x1_in_bounds]
    x1s = np.linspace(x1_lim[0], x1_lim[1], 20)
    x2s = np.linspace(x2_lim[0], x2_lim[1], 20)
    x1, x2 = np.meshgrid(x1s, x2s)
    xs = np.c_[x1.ravel(), x2.ravel()]
    df = (xs.dot(w) + b).reshape(x1.shape)
    m = 1 / np.linalg.norm(w)
    boundary_x2s = -x1s * (w[0] / w[1]) - b / w[1]
    margin_x2s_1 = -x1s * (w[0] / w[1]) - (b - 1) / w[1]
    margin_x2s_2 = -x1s * (w[0] / w[1]) - (b + 1) / w[1]
    ax.plot_surface(x1s, x2, 0, color="b", alpha=0.2, cstride=100, rstride=100)
    ax.plot(x1s, boundary_x2s, 0, "k-", linewidth=2, label=r"$h=0$")
    ax.plot(x1s, margin_x2s_1, 0, "k--", linewidth=2, label=r"$h=\pm 1$")
    ax.plot(x1s, margin_x2s_2, 0, "k--", linewidth=2)
    ax.plot(X_crop[:, 0][y_crop == 1], X_crop[:, 1][y_crop == 1], 0, "g^")
    ax.plot_wireframe(x1, x2, df, alpha=0.3, color="k")
    ax.plot(X_crop[:, 0][y_crop == 0], X_crop[:, 1][y_crop == 0], 0, "bs")
    ax.axis(x1_lim + x2_lim)
    ax.text(4.5, 2.5, 3.8, "Decision function $h$", fontsize=15)
    ax.set_xlabel(r"Petal length", fontsize=15)
    ax.set_ylabel(r"Petal width", fontsize=15)
    ax.set_zlabel(r"$h = \mathbf{w}^t \cdot \mathbf{x} + b$", fontsize=18)
    ax.legend(loc="upper left", fontsize=16)


fig = plt.figure(figsize=(11, 6))
ax1 = fig.add_subplot(111, projection="3d")
plot_3D_decision_function(ax1, w=svm_clf2.coef_[0], b=svm_clf2.intercept_[0])

save_fig("iris_3D_plot")


# # Small wight vector results in a large margin
def plot_2D_decision_function(w, b, ylabel=True, x1_lim=[-3, 3]):
    """
    
    :param w: 
    :param b: 
    :param ylabel: 
    :param x1_lim: 
    :return: 
    """
    x1 = np.linspace(x1_lim[0], x1_lim[1], 200)
    y = w * x1 + b
    m = 1 / w

    plt.plot(x1, y)
    plt.plot(x1_lim, [1, 1], "k:")
    plt.plot(x1_lim, [-1, -1], "k:")
    plt.axhline(y=0, color="k")
    plt.axvline(x=0, color="k")
    plt.plot([m, m], [0, 1], "k--")
    plt.plot([-m, -m], [0, -1], "k--")
    plt.plot([-m, m], [0, 0], "k-o", linewidth=3)
    plt.axis(x1_lim + [-2, 2])
    if ylabel:
        plt.ylabel(r"$w_1 x_1$  ", rotation=0, fontsize=16)
    plt.title(r"$w_1 = {}$".format(w), fontsize=16)


plt.figure(figsize=(12, 3.2))
plt.subplot(121)
plot_2D_decision_function(1, 0)
plt.subplot(122)
plot_2D_decision_function(0.5, 0, ylabel=False)
save_fig("small_w_large_margin_plot")

X = iris["data"][:, (2, 3)]
y = (iris["target"] == 2).astype(np.float64)

svm_clf = sklearn.svm.SVC(kernel="linear", C=1)
svm_clf.fit(X, y)
svm_clf.predict([[5.3, 1.3]])

# # Hinge loss
t = np.linspace(-2, 4, 200)
h = np.where(1 - t < 0, 0, 1 - t)

plt.figure(figsize=(5, 2.8))
plt.plot(t, h, "b-", linewidth=2, label="$max(0, 1 - t)$")
plt.grid(True, which="both")
plt.axhline(y=0, color="k")
plt.axvline(x=0, color="k")
plt.yticks(np.arange(-1, 2.5, 1))
plt.xlabel("$t$", fontsize=16)
plt.axis([-2, 4, -1, 2.5])
plt.legend(loc="upper right", fontsize=16)
save_fig("hinge_plot")

# # Extra material
# # Training time
X, y = sklearn.datasets.make_moons(n_samples=1000, noise=0.4)
plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bs")
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^")

tol = 0.1
tols = []
times = []

for i in range(10):
    svm_clf = sklearn.svm.SVC(kernel="poly", gamma=3, C=10, tol=tol, verbose=1)
    t1 = time.time()
    svm_clf.fit(X, y)
    t2 = time.time()
    times.append(t2 - t1)
    tols.append(tol)
    print(i, tol, t2 - t1)
    tol /= 10
plt.semilogx(tols, times)

# # Identical linear classifiers
X, y = sklearn.datasets.make_moons(n_samples=100, noise=0.15, random_state=42)

C = 5
alpha = 1 / (C * len(X))

sgd_clf = sklearn.linear_model.SGDClassifier(
    loss="hinge",
    learning_rate="constant",
    eta0=0.001,
    alpha=alpha,
    n_iter=100000,
    random_state=42
)
svm_clf = sklearn.svm.SVC(kernel="linear", C=C)
lin_clf = sklearn.svm.LinearSVC(loss="hinge", C=C)

X_scaled = sklearn.preprocessing.StandardScaler().fit_transform(X)
sgd_clf.fit(X_scaled, y)
svm_clf.fit(X_scaled, y)
lin_clf.fit(X_scaled, y)

print("SGDClassifier(alpha={}):     ".format(sgd_clf.alpha), sgd_clf.intercept_, sgd_clf.coef_)
print("SVC:                   :     ", svm_clf.intercept_, svm_clf.coef_)
print("LinearSVC              :     ", lin_clf.intercept_, lin_clf.coef_)

# # Linear SVM classifier implementation using Batch Gradient Descent
X = iris["data"][:, (2, 3)]
y = (iris["target"] == 2).astype(np.float64).reshape(-1, 1)


class MyLinearSVC(sklearn.base.BaseEstimator):
    def __init__(self, C=1, eta0=1, eta_d=10000, n_epochs=1000, random_state=None):
        self.C = C
        self.eta0 = eta0
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.eta_d = eta_d

    def eta(self, epoch):
        return self.eta0 / (epoch + self.eta_d)

    def fit(self, X, y):
        """
        fit with random initialization
        :param X: 
        :param y: 
        :return: 
        """
        if self.random_state:
            np.random.seed(self.random_state)
        w = np.random.randn(X.shape[1], 1)  # n feature weights
        b = 0

        m = len(X)
        t = y * 2 - 1  # -1 if t==0, +1 if t==1
        X_t = X * t
        self.Js = []

        # Training
        for epoch in range(self.n_epochs):
            support_vectors_idx = (X_t.dot(w) + t * b < 1).ravel()
            X_t_sv = X_t[support_vectors_idx]
            t_sv = t[support_vectors_idx]

            J = 1 / 2 * np.sum(w * w) + self.C * (np.sum(1 - X_t_sv.dot(w)) - b * np.sum(t_sv))
            self.Js.append(J)

            w_gradient_vector = w - self.C * np.sum(X_t_sv, axis=0).reshape(-1, 1)
            b_derivative = -C * np.sum(t_sv)

            w = w - self.eta(epoch) * w_gradient_vector
            b = b - self.eta(epoch) * b_derivative

        self.intercept_ = np.array([b])
        self.coef_ = np.array([w])
        support_vectors_idx = (X_t.dot(w) + b < 1).ravel()
        self.support_vectors_ = X[support_vectors_idx]
        return self

    def decision_function(self, X):
        return X.dot(self.coef_[0]) + self.intercept_[0]

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(np.float64)


C = 2
svm_clf = MyLinearSVC(C=C, eta0=10, eta_d=1000, n_epochs=60000, random_state=2)
svm_clf.fit(X, y)
svm_clf.predict(np.array([[5, 2], [4, 1]]))

plt.plot(range(svm_clf.n_epochs), svm_clf.Js)
plt.axis([0, svm_clf.n_epochs, 0, 100])

print(svm_clf.intercept_, svm_clf.coef_)

svm_clf2 = sklearn.svm.SVC(kernel="linear", C=C)
svm_clf2.fit(X, y.ravel())
print(svm_clf2.intercept_, svm_clf2.coef_)

yr = y.ravel()
plt.figure(figsize=(12, 3.2))
plt.subplot(121)
plt.plot(X[:, 0][yr == 1], X[:, 1][yr == 1], "g^", label="Iris-Virginica")
plt.plot(X[:, 0][yr == 0], X[:, 1][yr == 0], "bs", label="Not Iris-Virginica")
plot_svc_decision_boundary(svm_clf, 4, 6)
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.title("MyLinearSVC", fontsize=14)
plt.axis([4, 6, 0.8, 2.8])

plt.subplot(122)
plt.plot(X[:, 0][yr == 1], X[:, 1][yr == 1], "g^")
plt.plot(X[:, 0][yr == 0], X[:, 1][yr == 0], "bs")
plot_svc_decision_boundary(svm_clf2, 4, 6)
plt.xlabel("Petal length", fontsize=14)
plt.title("sklearn.svm.SVC", fontsize=14)
plt.axis([4, 6, 0.8, 2.8])
save_fig("mySVC_vs_skSVC")

# Using a SGDClassifier to find support vectors

sgd_clf = sklearn.linear_model.SGDClassifier(loss="hinge", alpha=0.017, n_iter=50, random_state=42)
sgd_clf.fit(X, y.ravel())

m = len(X)
t = y * 2 - 1
X_b = np.c_[np.ones((m, 1)), X]  # add bias input x0=1
X_b_t = X_b * t
sgd_theta = np.r_[sgd_clf.intercept_[0], sgd_clf.coef_[0]]
print(sgd_theta)
support_vectors_idx = (X_b_t.dot(sgd_theta) < 1).ravel()
sgd_clf.support_vectors_ = X[support_vectors_idx]
sgd_clf.C = C

plt.figure(figsize=(5.5, 3.2))
plt.plot(X[:, 0][yr == 1], X[:, 1][yr == 1], "g^")
plt.plot(X[:, 0][yr == 0], X[:, 1][yr == 0], "bs")
plot_svc_decision_boundary(sgd_clf, 4, 6)
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.title("SGDClassifier", fontsize=14)
plt.axis([4, 6, 0.8, 2.8])
save_fig("SGDClassifier_with_support_vectors")
