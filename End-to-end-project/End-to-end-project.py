# -*- Coding: utf-8 -*-

# Setting up
from __future__ import division, print_function, unicode_literals

import numpy as np
import numpy.random as rnd
import scipy as sp
import scipy.stats
import os
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tarfile
import pandas as pd
import pandas.tools.plotting
import hashlib
import sklearn
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.base
import sklearn.pipeline
import sklearn.metrics
import sklearn.tree
import sklearn.linear_model
import sklearn.ensemble
import sklearn.svm
from six.moves import urllib

rnd.seed(42)  # to make this script's output stable across runs

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

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
    Disadvantage: Each time run this function will generate complete different test-set.
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
# stratified sampling(按权重比例取样) based on the income category
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
cbar.ax.set_yticklabels(["$%dk" % (round(v / 1000)) for v in tick_values], fontsize=14)
cbar.set_label('Median House Value', fontsize=16)

plt.legend(fontsize=16)
save_fig("california_housing_prices_plot")

# Looking for Correlations
# compute the standard correlation coefficient (also called Pearson’s r)
# between every pair of attributes using the corr() method
corr_matrix = housing.corr()

# how much each attribute correlates with the median house value
corr_matrix["median_house_value"].sort_values(ascending=False)

housing.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.3)
plt.axis([0, 16, 0, 550000])
save_fig("income_vs_house_value_scatterplot")

# Pandas Scatter Matrix
# which plots every numerical attribute against every other numerical attribute.
attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"
              ]
pandas.tools.plotting.scatter_matrix(housing[attributes], figsize=(11, 8))
save_fig("scatter_matrix_plot")

# Experimenting with Attribute Combinations
housing['rooms_per_household'] = housing["total_rooms"] / housing['population']
housing['bedrooms_per_room'] = housing['total_bedrooms'] / housing['total_rooms']
housing['population_per_household'] = housing['population'] / housing['households']

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

housing.plot(kind="scatter", x="rooms_per_household", y="median_house_value",
             alpha=0.2
             )
plt.axis([0, 5, 0, 520000])
save_fig('rooms_per_household')

housing.describe()

# Prepare the data for Machine Learning algorithms
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

housing_copy = housing.copy().iloc[21:24]
print(housing_copy)

# methods to deal with the missing values.
# option 1

housing_copy.dropna(subset=["total_bedrooms"], how='any')

# option 2
housing_copy = housing.copy().iloc[21:24]
print(housing_copy.drop("total_bedrooms", axis=1))

# option 3
# Set the values to some value (zero, the mean, the median, etc)
housing_copy = housing.copy().iloc[21:24]
median = housing_copy["total_bedrooms"].median()
housing_copy["total_bedrooms"].fillna(median, inplace=True)
print(housing_copy)

# use sklearn to take care of missing values
imputer = sklearn.preprocessing.Imputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)
print(housing_tr.iloc[21:24])

# statistics_ : array of shape (n_features,)
# The imputation fill value for each feature if axis == 0.
print(imputer.statistics_)
# Compare to below
print(housing_num.median().values)

housing_tr.head()

# Handling Text and Categorical Attributes
# Convert text Label into numbers with sklearn.preprocessing.LabelEncoder
# 0 represent <1H OCEAN, 1 represents INLAND etc
encoder = sklearn.preprocessing.LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)
print(housing_cat_encoded)
# compare with below
print(housing_cat)
print(encoder.classes_)

# OneHotEncoding s. 92
encoder = sklearn.preprocessing.OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
print(housing_cat_1hot)  # this is a sparse matrix
print(housing_cat_1hot.toarray())  # convert that matrix into a 2D-array

# LabelEncoding and OneHotEncoding can be applied at the same time by using LabelBinarizer
encoder = sklearn.preprocessing.LabelBinarizer()
housing_cat_1hot_2 = encoder.fit_transform(housing_cat)
print(housing_cat_1hot_2)  # But this is already in 2D-array.

# Custom Transformers s. 94
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

housing_extra_attribs = pd.DataFrame(housing_extra_attribs,
                                     columns=list(housing.columns) + ["rooms_per_household",
                                                                      "population_per_household"])
housing_extra_attribs.head()

# Transformation Pipelines
# A numerical Pipeline
num_pipeline = sklearn.pipeline.Pipeline([
    ("imputer", sklearn.preprocessing.Imputer(strategy="median")),
    ("attribs_adder", CombinedAttributesAdder()),
    ("std_scaler", sklearn.preprocessing.StandardScaler()),
])

num_pipeline.fit_transform(housing_num)


# A full Pipeline includes numerical and categorical values
# Before that we define a selector transformer: it simply transforms the
# data by selecting the desired attributes (numerical or categorical),
# dropping the rest, and converting the resulting DataFrame to a NumPy array.

class DataFrameSelector(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = sklearn.pipeline.Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', sklearn.preprocessing.Imputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', sklearn.preprocessing.StandardScaler()),
])

cat_pipeline = sklearn.pipeline.Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('label_binarizer', sklearn.preprocessing.LabelBinarizer()),
])

preparation_pipeline = sklearn.pipeline.FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])

housing_prepared = preparation_pipeline.fit_transform(housing)
print(housing_prepared)
print(housing_prepared.shape)

# Select and Train a Model
# Linear Regression model

lin_reg = sklearn.linear_model.LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# try the full pipeline on a few training instances
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = preparation_pipeline.transform(some_data)

print("Predictions:\t", lin_reg.predict(some_data_prepared))
print("Labels\t\t", list(some_labels))

# measure RMSE and MAE on the whole training set
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = sklearn.metrics.mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)
lin_mae = sklearn.metrics.mean_absolute_error(housing_labels, housing_predictions)
print(lin_mae)

# linear regression is not so good in this case, let's try another one,
# the DecisionTreeRegression
tree_reg = sklearn.tree.DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_rmse = np.sqrt(sklearn.metrics.mean_squared_error(housing_labels, housing_predictions))
print(tree_rmse)

# Better Evaluation Using Cross-Validation
# K-fold cross-validation
tree_scores = sklearn.model_selection.cross_val_score(tree_reg, housing_prepared,
                                                      housing_labels,
                                                      scoring="neg_mean_squared_error",
                                                      cv=10
                                                      )
tree_rmse_scores = np.sqrt(-tree_scores)


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


display_scores(tree_rmse_scores)

lin_scores = sklearn.model_selection.cross_val_score(lin_reg, housing_prepared, housing_labels,
                                                     scoring="neg_mean_squared_error", cv=10
                                                     )
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)
"""
the Decision Tree model is overfitting so badly that it performs worse than the
Linear Regression model.
"""

# RandomForestRegression
forest_reg = sklearn.ensemble.RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)

forest_mse = sklearn.metrics.mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
print(forest_rmse)

t1 = time.time()
forest_scores = sklearn.model_selection.cross_val_score(forest_reg, housing_prepared, housing_labels,
                                                        scoring="neg_mean_squared_error", cv=10
                                                        )
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)
t2 = time.time()
print("It takes {} sec to get this result.".format(t2 - t1))

scores = sklearn.model_selection.cross_val_score(lin_reg, housing_prepared, housing_labels,
                                                 scoring="neg_mean_squared_error", cv=10)
print(pd.Series(np.sqrt(-scores)).describe())

# Support vector machine
t1 = time.time()
svm_reg = sklearn.svm.SVR(kernel="linear")
svm_reg.fit(housing_prepared, housing_labels)
housing_predictions = svm_reg.predict(housing_prepared)
t2 = time.time()
print("It takes {} sec to solve.".format(t2 - t1))

svm_mse = sklearn.metrics.mean_squared_error(housing_labels, housing_predictions)
svm_rmse = np.sqrt(svm_mse)
print(svm_rmse)

# Fine-Tune The Model
# Grid Search
# searches for the best combination of hyperparameter values for the RandomForestRegressor:
param_grid = [
    {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
    {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
]

forest_reg = sklearn.ensemble.RandomForestRegressor()
grid_search = sklearn.model_selection.GridSearchCV(forest_reg, param_grid, cv=5, scoring="neg_mean_squared_error")
grid_search.fit(housing_prepared, housing_labels)

print(grid_search.best_params_)  # Get the best parameters
print(grid_search.best_estimator_)  # Get the best estimators

# get the evaluation scores
cvres = grid_search.cv_results_  # is a Dictionary
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

print(pd.DataFrame(grid_search.cv_results_))  # Convert result into DataFrame

# Randomized Search
t1 = time.time()
param_distribs = {
    "n_estimators": scipy.stats.randint(low=1, high=200),
    "max_features": scipy.stats.randint(low=1, high=8),
}

forest_reg = sklearn.ensemble.RandomForestRegressor()
rnd_search = sklearn.model_selection.RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                                        n_iter=10, cv=5, scoring="neg_mean_squared_error")
rnd_search.fit(housing_prepared, housing_labels)
t2 = time.time()
print("This fitting process takes {} sec to complete.".format(t2 - t1))

cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

# Analyze the Best Models and Their Errors
feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)

extra_attribs = ["rooms_per_household", "population_per_household", "bedrooms_per_room"]
cat_one_hot_attribs = list(encoder.classes_)
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
print(sorted(zip(feature_importances, attributes), reverse=True))


# Evaluate the System on the Test Set
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_transformed = preparation_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_transformed)

final_mse = sklearn.metrics.mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)
