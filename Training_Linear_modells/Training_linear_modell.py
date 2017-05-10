import numpy as np
import numpy.random as rnd
import numpy.linalg
import os
import matplotlib
import matplotlib.pyplot as plt
import sklearn.linear_model

rnd.seed(42)
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
#
PROJECT_ROOT_DIR = os.getcwd()


def save_fig(fig_id, tight_layout=True):
    """

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
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 2, 0, 15])
save_fig("generated_data_plot")

# Now compute the Normal Equation.
X_b = np.c_[np.ones((100, 1)), X]  # add x0=1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

print(theta_best)
"""
we got [[ 3.86501051]
        [ 3.13916179]]
close enough to y = 4 + 3x, the noise make it impossible to get exact 4 and 3.
"""

# Now use the best theta to predict.
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0=1 to each instance
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
            style = "b-" if iteration > 0 else "r--"
            plt.plot(X_new, y_predict, style)
        gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients
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
