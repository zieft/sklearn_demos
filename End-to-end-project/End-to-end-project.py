# -*- Coding: utf-8 -*-

# Setting up
from __future__ import division, print_function, unicode_literals

import numpy as np
import numpy.random as rnd
import os
import matplotlib
import matplotlib.pyplot as plt

rnd.seed(42)  # to make this notebook's output stable across runs

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"


def save_fig(fig_id, tight_layout=True):
    """
    
    :param fig_id: 
    :param tight_layout: Automatically adjust subplot parameters to give specified padding.
    :return: 
    """
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + '.png')
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

# Get the data

DATASETS_URL = ""