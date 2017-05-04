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
import sklearn.model_selection
import matplotlib.image as mpimg

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
# print(housing.describe())  # Generate various summary statistics,
# excluding NaN values.

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
    shuffled_indices = rnd.permutation(len(data))  # always generates the
    # same shuffled indices.
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


train_set_1, test_set_1 = split_train_test(housing, 0.2)
print(len(train_set_1), len(test_set_1))


def test_set_check(identifier, test_ratio, hash):
    """
    ????????????????????????????????????????????????????????????????
    compute a hash of each instance’s identifier, keep only the last byte 
    of the hash, and put the instance in the test set if this value is lower
    or equal to 51 (~20% of 256).
    :param identifier: 
    :param test_ratio: 
    :param hash: 
    :return: 
    """
    return bytearray(hash(np.int64(identifier)).digest())[-1] < 256 * test_ratio


def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    """
    ????????????????????????????????????????????????????????????????
    This ensures that the test set will remain consistent across multiple runs, 
    even if you refresh the dataset. The new test set will contain 20% of the 
    new instances, but it will not contain any instance that was previously in 
    the training set.
    :param data: 
    :param test_ratio: 
    :param hash: 'index'
    :return: 
    """
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]  # 按位取反运算


# the housing dataset does not have an identifier column.
# The simplest solution is to use the row index as the ID
housing_with_id = housing.reset_index()  # adds an 'index' column
train_set_2, test_set_2 = split_train_test_by_id(housing_with_id, 0.2, 'index')
test_set_2.head()

# train_test_split() from sklearn do exactly the same thing.
train_set, test_set = sklearn.model_selection.train_test_split \
    (housing, test_size=0.2, random_state=42)  # 42??

test_set.head()

# an experts who said that the median income is a very important attribute
# to predict median housing prices. Let's see
housing["median_income"].hist()

housing["income_cat"] = np.ceil(housing['median_income'] / 1.5)  # /1.5 to limit
# the number of income categories, ceil to round up to have discrete categories
housing["income_cat"].where(housing['income_cat'] < 5, 5.0, inplace=True)
housing["income_cat"].value_counts()

# Stratified ShuffleSplit cross-validator
# Provides train/test indices to split data in train/test sets.
# stratified sampling based on the income category
split = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]  # loc: Purely label-location based indexer for selection by label.
    strat_test_set = housing.loc[test_index]


def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)


train_set, test_set = sklearn.model_selection.train_test_split(housing, test_size=0.2, random_state=42)

compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
}).sort_index()
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
print(compare_props)

# remove the income_cat attribute so the data is back to its original state
for set in (strat_train_set, strat_test_set):
    set.drop("income_cat", axis=1, inplace=True)

# Discover and visualize the data to gain insights
# bad visualization
housing = strat_train_set.copy()
housing.plot(kind="scatter", x='longitude', y='latitude')
save_fig('bad_visualization_plot')

# better visualization with transparent effect
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)  # alpha = transparent
save_fig("better_visualization_plot")

# add house price, populations into plot and using different colors.
housing.plot(kind='scatter', x="longitude", y="latitude",
             s=housing['population'] / 100, label='population',
             c="median_house_value", cmap=plt.get_cmap('jet'),
             colorbar=True, alpha=0.4, figsize=(10, 7),
             )
"""
DataFrame plotting accessor and method (housing.plot())
The radius of each circle represents the district’s population (option s), 
and the color represents the price (option c). We will use a predefined color 
map (option cmap) called jet, which ranges from blue (low values) to red 
(high prices).
"""
plt.legend()  # Places a legend on the axes.
save_fig("housing_prices_scatterplot")

# add map as background
california_img = mpimg.imread(PROJECT_ROOT_DIR + '/images/california.png')
ax = housing.plot(kind='scatter', x="longitude", y="latitude",
                  s=housing['population'] / 100, label='population',
                  c="median_house_value", cmap=plt.get_cmap('jet'),
                  colorbar=True, alpha=0.4, figsize=(10, 7),
                  )
plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5)
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)

prices = housing["median_house_value"]
tick_values = np.linspace(prices.min(), prices.max(), 11)
cbar = plt.colorbar()
cbar.ax.set_yticklabels(["$%dk"%(round(v/1000)) for v in tick_values], fontsize=14)
cbar.set_label('Median House Value', fontsize=16)

plt.legend(fontsize=16)
save_fig("california_housing_prices_plot")

#