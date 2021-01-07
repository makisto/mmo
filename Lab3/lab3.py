import numpy
import pandas
import sklearn
from sklearn import model_selection
from sklearn.linear_model import *
from sklearn import preprocessing

def is_numeric(a):
    try:
        float(a)
        return True
    except ValueError:
        return False

data = pandas.read_csv('winequalityN.csv', header=0)
lst = data.values
for l in lst:
    l[0] = 1 #if l[0]=='white' else 0
#print(lst[0])
x = lst[...,0:12]
y = lst[...,12]
L = len(y)
for i in range(0, 12):
    for j in range(L):
        if not is_numeric(x[j, i]) or numpy.isnan(x[j, i]):
            x[j, i] = 0#print('error!!! ', j, i, x[j,i])
    #x[...,i] = preprocessing.normalize([x[...,i]])
for i in range(L):
    if numpy.isnan(y[i]):
        y[i] = 0
#y=preprocessing.normalize([y])
#print(x)
#print(y)
#y=y[0]

for i in range(0, 10):
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3)

    model = LassoCV(cv=30)
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)

    test_len = len(x_test)

    correct=0

    for i in range(test_len):
        if(abs(y_test[i] - predicted[i]) < 1):
            correct += 1

    print((correct / test_len) * 100)
