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
some_digit = X[some_digit_inde:x]
plot_digit(some_digit)
save_fig("example")

