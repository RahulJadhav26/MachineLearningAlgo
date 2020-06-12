import sklearn
from sklearn import datasets
from sklearn import svm
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

# print(cancer.feature_names)

# print(cancer.target_names)

X = cancer.data
Y = cancer.target

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.2)

classes = ["malignant" "benign"]

clf = svm.SVC(kernel="linear" ,C=3)

clf.fit(X_train,Y_train)

Y_pred = clf.predict(X_test)

acc =metrics.accuracy_score(Y_test,Y_pred)

print(acc)
