import numpy as np
import numpy.random as rnd
import numpy.linalg
import os
import matplotlib
import matplotlib.pyplot as plt
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.pipeline

rnd.seed(42)
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12

PROJECT_ROOT_DIR = os.getcwd()


def save_fig(fig_id, tight_layout=True):
    """
    Save plot as .png file.
    :param fig_id: must be a String
    :param tight_layout: Automatically adjust subplot parameters to give specified padding.
    :return:
    """
    assert isinstance(fig_id, str), 'Must be a string.'
    if not os.path.exists(os.path.join(PROJECT_ROOT_DIR, 'images')):
        os.makedirs(os.path.join(PROJECT_ROOT_DIR, "images"))
    path = os.path.join(PROJECT_ROOT_DIR, "images", fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


# generate some linear-looking data to test normal equation.
X = 2 * np.random.rand(100, 1)  # Generate 100 rows 1 column, column vector like.
y = 4 + 3 * X + np.random.randn(100, 1)  # randn() generate standard normal distribution

plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 2, 0, 15])
save_fig("generated_data_plot")

# Now compute the Normal Equation.
X_b = np.c_[np.ones((100, 1)), X]  # add x0=1 to each instance.
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)  # Normal equations

print(theta_best)
"""
we got [[ 3.86501051]
        [ 3.13916179]]
close enough to y = 4 + 3x, the noise make it impossible to get exact 4 and 3.
"""

# Now use the best theta to predict.
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0=1 to each instance  np.c_按行拼贴，即将两个数组的列并在一起
y_predict = X_new_b.dot(theta_best)
print(y_predict)

plt.plot(X_new, y_predict, "r-", linewidth=2, label="Predictions")
plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.axis([0, 2, 0, 15])
save_fig("linear_model_predictions")

# The equivalent code using sklearn:
lin_reg = sklearn.linear_model.LinearRegression()
lin_reg.fit(X, y)
print(lin_reg.intercept_)
print(lin_reg.coef_)
lin_reg.predict(X_new)

# End


# Linear regression using batch gradient descent
theta_path_bgd = []


def plot_gradient_descent(theta, eta, theta_path=None, steps=10, n_iterations=1000):
    """
    
    :param theta: 
    :param eta: 
    :param theta_path: 
    :return: 
    """
    m = len(X_b)
    plt.plot(X, y, "b.")
    for iteration in range(n_iterations):
        if iteration < steps:
            y_predict = X_new_b.dot(theta)
            style = "b-" if iteration > 0 else "r--"  # 第一步画红虚线，后面的画蓝线
            plt.plot(X_new, y_predict, style)
        gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)  # Equation 4-6 s.137
        theta = theta - eta * gradients  # Equation 4-7
        if theta_path is not None:
            theta_path.append(theta)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 2, 0, 15])
    plt.title(r"$\eta={}$".format(eta), fontsize=16)


theta = rnd.randn(2, 1)

plt.figure(figsize=(10, 4))
plt.subplot(131)
plot_gradient_descent(theta, eta=0.02)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.subplot(132)
plot_gradient_descent(theta, eta=0.1, theta_path=theta_path_bgd)
plt.subplot(133)
plot_gradient_descent(theta, eta=0.5)

save_fig("gradient_descent_plot")

# Stochastic Gradient Descent
theta_path_sgd = []

n_iterations = 50

theta = rnd.randn(2, 1)


def learning_schedule(t):
    t0, t1 = 5, 50
    return t0 / (t + t1)


m = len(X_b)

for epoch in range(n_iterations):
    for i in range(m):
        if epoch == 0 and i < 20:
            y_predict = X_new_b.dot(theta)
            style = "b-" if i > 0 else "r--"
            plt.plot(X_new, y_predict, style)
        random_index = rnd.randint(m)  # 随机生成0到m间的一个整数。
        xi = X_b[random_index:random_index + 1]  # random_index +1的作用是什么?
        yi = y[random_index: random_index + 1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
        theta_path_sgd.append(theta)

plt.plot(X, y, 'b.')
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", fontsize=18)
plt.axis([0, 2, 0, 15])
save_fig("sgd_plot")

# Usiing sklearn to do the same thing
sgd_reg = sklearn.linear_model.SGDRegressor()
sgd_reg.fit(X, y.ravel())
print(sgd_reg.intercept_, sgd_reg.coef_)

# Mini-batch Gradient Descent
theta_path_mgd = []

n_iterations = 50
minibatch_size = 20

theta = rnd.randn(2, 1)


def learning_schedule2(t):
    t0, t1, = 10, 1000
    return t0 / (t + t1)


t = 0
for epoch in range(n_iterations):
    shuffled_indices = rnd.permutation(m)
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    for i in range(0, m, minibatch_size):
        t += 1
        xi = X_b_shuffled[i:i + minibatch_size]
        yi = y_shuffled[i:i + minibatch_size]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule2(t)
        theta = theta - eta * gradients
        theta_path_mgd.append(theta)

theta_path_bgd = np.array(theta_path_bgd)
theta_path_sgd = np.array(theta_path_sgd)
theta_path_mgd = np.array(theta_path_mgd)

plt.figure(figsize=(7, 4))
plt.plot(theta_path_sgd[:, 0], theta_path_sgd[:, 1], "r-s", linewidth=1, label="Stochastic")
plt.plot(theta_path_mgd[:, 0], theta_path_mgd[:, 1], "g-+", linewidth=2, label="Mini-batch")
plt.plot(theta_path_bgd[:, 0], theta_path_bgd[:, 1], "b-o", linewidth=3, label="Batch")
plt.legend(loc="upper left", fontsize=16)
plt.xlabel(r"$\theta_0$", fontsize=20)
plt.ylabel(r"$\theta_1$   ", fontsize=20, rotation=0)
plt.axis([2.5, 4.5, 2.3, 3.9])
save_fig("gradient_descent_paths_plot")

# Polynomial Regression
# generate some nonlinear data based on a simple quadratic equation
rnd.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

plt.plot(X, y, 'b.')
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([-3, 3, 0, 10])
save_fig("quadratic_data_plot")

# Extend the data set
poly_features = sklearn.preprocessing.PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
print(X[0])
print(X_poly[0])  # X_poly now contains the original feature of X plus the square of this feature.

# Using Lin_reg to train the extended dataset.
lin_reg = sklearn.linear_model.LinearRegression()
lin_reg.fit(X_poly, y)
print(lin_reg.intercept_, lin_reg.coef_)

X_new = np.linspace(-3, 3, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)
y_new = lin_reg.predict(X_new_poly)
plt.plot(X, y, 'b.')
plt.plot(X_new, y_new, 'r-', linewidth=2, label="predictions")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.axis([-3, 3.0, 10])
save_fig("quadratic_predictions_plot")


# Learning Curves
# Define a function that plots the learning curves of a model:

def plot_learning_curves(model, X, y):
    """
    把数据集分成训练集和验证集，分别求它们的mean squared error并作图。
    :param model: a training model instance like LinearRegressor etc.
    :param X: 
    :param y:
    :arg m: every integers smaller than the length of training set to iterate.
    :return: 
    """
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(sklearn.metrics.mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(sklearn.metrics.mean_squared_error(y_val_predict, y_val))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="Training set")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Validation set")
    plt.legend(loc="upper right", fontsize=14)
    plt.xlabel("Training set size", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)


plot_learning_curves(lin_reg, X, y)
plt.axis([0, 80, 0, 3])
save_fig("underfitting_learning_curves_plot")

