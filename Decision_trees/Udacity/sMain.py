#!/usr/bin/python

""" lecture and example code for decision tree unit """

import sys
from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from classifyDT import classify

features_train, labels_train, features_test, labels_test = makeTerrainData()



### the classify() function in classifyDT is where the magic
### happens--fill in this function in the file 'classifyDT.py'!
clf1 = classify_2(features_train, labels_train)
clf2 = classify_50(features_train, labels_train)


from sklearn.metrics import accuracy_score

acc1 = accuracy_score(clf1.predict(features_test), labels_test)
acc2 = accuracy_score(clf2.predict(features_test), labels_test)
print acc1, acc2



accuracy_score(labels_test, clf1)
#### grader code, do not modify below this line

prettyPicture(clf1, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())