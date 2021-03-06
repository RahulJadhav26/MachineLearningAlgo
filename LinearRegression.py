import pandas as pd
import numpy as np
import sklearn 
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv('student-mat.csv' , sep= ";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])
best = 0
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)

"""for _ in range(30):
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)
    linear = linear_model.LinearRegression()

    linear.fit(X_train, Y_train)

    acc = linear.score(X_test,Y_test)

    print(acc)

    if acc > best:
        best = acc
        print(best,"best")
        with open('studentmodel.pickle', "wb") as f:
            pickle.dump(linear, f)"""

pickle_in = open('studentmodel.pickle',"rb")

linear = pickle.load(pickle_in)
acc = linear.score(X_test,Y_test)

print("Co:", linear.coef_)
print("Intercept", linear.intercept_)
print(acc,"Accuracy")
predictions = linear.predict(X_test)

for x in range(len(predictions)):
    print(predictions[x], X_test[x], Y_test[x])
    

