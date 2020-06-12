import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model,preprocessing

data = pd.read_csv("car.data")
print(data.head())

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door= le.fit_transform(list(data["door"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
clss = le.fit_transform(list(data["class"]))

predict = "class"

X = list(zip(buying,maint,door,lug_boot,safety,clss))
Y=list(clss)

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X,Y, test_size= 0.1)

model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, Y_train)
acc = model.score(X_test, Y_test)
print(acc)

predicted = model.predict(X_test)
names = ["unacc", "acc", "good" , "vgood"]

for x in range(len(X_test)):
    print("Predicted" , names[predicted[x]])
    print("Data:" ,X_test[x])
    print("Actual:" ,names[Y_test[x]])
    n= model.kneighbors([X_test[x]], 9, True) 
    print(n)
