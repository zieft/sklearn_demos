import numpy as np
import numpy.random as rnd
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm
import six.moves
import sklearn.datasets
import scipy.io

plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12

PROJECT_ROOT_DIR = os.getcwd()


def save_fig(fig_id, tight_layout=True):
    assert isinstance(fig_id, str), "Must be a string."
    if not os.path.exists(os.path.join(PROJECT_ROOT_DIR, "images")):
        os.makedirs(os.path.join(PROJECT_ROOT_DIR, "images"))
    path = os.path.join(PROJECT_ROOT_DIR, "images", fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


try:
    mnist = sklearn.datasets.fetch_mldata('MNIST original')
except six.moves.urllib.error.HTTPError as ex:
    print("Could not download MNIST data from mldata.org, trying alternative....")

    mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
    mnist_path = "./mnist-original.mat"
    response = six.move.urllib.request.urlopen(mnist_alternative_url)
    with open(mnist_path, 'wb') as f:
        content = response.read()
        f.write(content)
    mnist_raw = scipy.io.loadmat(mnist_path)
    mnist = {
        "data": mnist_raw["data"].T,
        "target": mnist_raw["label"][0],
        "COL_NAMES": ["label", "data"],
        "DESCR": "mldata.org dataset: mnist-original",
    }
    print("Success!")

X, y = mnist["data"], mnist["target"]
print(X.shape)


def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")


some_digit_index = 36000
some_digit = X[some_digit_index]
plot_digit(some_digit)
save_fig("example")


def plot_digits(instances, images_per_row=10, **options):
    """
    more digits plot
    :param instances: 
    :param images_per_row: 
    :param options: 
    :return: see file "more_digits_plot.png"
    :key: np.concatenate() Join a sequence of arrays along an existing axis.
    """
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size, size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row:(row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap=matplotlib.cm.binary, **options)
    plt.axis("off")


plt.figure(figsize=(9, 9))
example_images = np.r_[X[:12000:600], X[13000:30600:600], X[30600:60000:590]]  # ??
plot_digits(example_images, images_per_row=10)
save_fig("more_digits_plot")

# Since MNIST dataset has been already separated into training set and test set
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = rnd.permutation(60000)  # this will guarantee that all cross-validation folds will be similar
X_train, y_trian = X_train[shuffle_index], y_train[shuffle_index]
"""
permutation(x):
Randomly permute a sequence, or return a permuted range.
"""


