from sklearn import tree

def classify_2(features_train, labels_train):
    ### your code goes here--should return a trained decision tree classifer
    clf = tree.DecisionTreeClassifier()
    clf.fit(features_train, labels_train)


    return clf

def classify_50(features_train, labels_train):
    clf = tree.DecisionTreeClassifier(min_samples_split=50)
    clf.fit(features_train, labels_train)

    return clf
