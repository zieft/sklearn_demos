# -*- Coding: utf-8 -*-

# Setting up
from __future__ import division, print_function, unicode_literals

import numpy as np
import numpy.random as rnd
import os
import matplotlib
import matplotlib.pyplot as plt
import tarfile
from six.moves import urllib
import pandas as pd
import hashlib

rnd.seed(42)  # to make this notebook's output stable across runs

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

PROJECT_ROOT_DIR = os.getcwd()
CHAPTER_ID = "end_to_end_project"


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


# Get the data

DATASETS_URL = "https://github.com/ageron/handson-ml/raw/master/datasets"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DATASETS_URL + "/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """
    Make a new dir named housing and download housing.tgz from housing_url,
    Extract the tgz file into a csv file.
    :param housing_url: 
    :param housing_path: 
    :return: housing directory with housing.csv and housing.tgz
    """
    if not os.path.exists(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)  # Help on function urlretrieve in module urllib
    with tarfile.open(tgz_path) as f:
        f.extractall(path=housing_path)


fetch_housing_data()


def load_housing_data(housing_path=HOUSING_PATH):
    """
    data loader for csv file.
    :param housing_path: 
    :return: a pd DataFrame, use .info() to see the details
    """
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)


housing = load_housing_data()

"""
Use lines below to see the overview of this dataset.
"""
# housing.head()
# housing.info()
# housing['ocean_proximity'].value_counts()
# print(housing.describe())  # Generate various summary statistics, excluding NaN values.

housing.hist(bins=50, figsize=(11, 8))
save_fig('attribute_histogram_plots')


# plt.show()


def split_train_test(data, test_ratio):
    """
    Split dataset into two parts. One for testing and the other for training.
    :param data: the pandas DataFrame
    :param test_ratio: The ratio that the size of test set over the whole dataset.
    :return: Purely integer-location based indexing for selection by position.
    """
    shuffled_indices = rnd.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


train_set, test_set = split_train_test(housing, 0.2)


# print(len(train_set), len(test_set))

def test_set_check(identifier, test_ratio, hash):
    """
    ????????????????????????????????????????????????????????????????
    :param identifier: 
    :param test_ratio: 
    :param hash: 
    :return: 
    """
    return bytearray(hash(np.int64(identifier)).digest())[-1] < 256 * test_ratio


def split_train_test_by_id(data, test_ratio, hash=hashlib.md5):
    """
    ????????????????????????????????????????????????????????????????
    :param data: 
    :param test_ratio: 
    :param hash: 
    :return: 
    """
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~n_test_set], data.loc[in_test_set]
