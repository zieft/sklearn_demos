from __future__ import division, print_function, unicode_literals

import numpy as np
import numpy.random as rnd
import os
import matplotlib
import matplotlib.pyplot as plt

rnd.seed(42)
PROJECT_ROOT_DIR = os.getcwd() + "/Dementionality_Reduction"

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

