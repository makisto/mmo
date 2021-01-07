import csv
import pandas
import numpy
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

data = {'red': [], 'white': []}
y = {'red': [], 'white': []}
with open ('winequalityN.csv') as f:
    r = csv.reader(f)
    features = next(r)
    for row in r:
        row = [x if x != '' else 0 for x in row]
        data[row[0]].append([float(x) for x in row[1:-1]])
        y[row[0]].append(int(row[-1]))

optimal = 0.0
optimal_alpha = 0.0
for i in range(0, 10):
    x_train_red, x_test_red, y_train_red, y_test_red = train_test_split(data['red'],
    y['red'], test_size = 0.3)

    lr = LassoCV(normalize=True).fit(x_train_red, y_train_red)

    alpha = lr.alpha_

    predicted_red = lr.predict(x_test_red)

    test_len_red = len(x_test_red)

    correct = 0

    for i in range(test_len_red):
        if(abs(y_test_red[i] - predicted_red[i]) < 1):
            correct += 1

    print("RED:" + str((correct / test_len_red) * 100))
    print("ALPHA:" + str(alpha) + "\n")

    if(optimal < (correct / test_len_red) * 100):
        optimal = (correct / test_len_red) * 100
        optimal_alpha = alpha

print("OPTIMAL RED:" + str(optimal))
print("OPTIMAL ALPHA:" + str(optimal_alpha) + "\n")

optimal = 0.0
optimal_alpha = 0.0
for i in range(0, 10):
    x_train_white, x_test_white, y_train_white, y_test_white = train_test_split(data['white'],
    y['white'], test_size = 0.3)

    lw = LassoCV(normalize=True).fit(x_train_white, y_train_white)

    alpha = lw.alpha_

    predicted_white = lr.predict(x_test_white)

    test_len_white = len(x_test_white)

    correct = 0

    for i in range(test_len_white):
        if(abs(y_test_white[i] - predicted_white[i]) < 1):
            correct += 1

    print("WHITE:" + str((correct / test_len_white) * 100))
    print("ALPHA:" + str(alpha) + "\n")

    if(optimal < (correct / test_len_white) * 100):
        optimal = (correct / test_len_white) * 100
        optimal_alpha = alpha

print("OPTIMAL WHITE:" + str(optimal))
print("OPTIMAL ALPHA:" + str(optimal_alpha) + "\n")

optimal = 0.0
optimal_alpha = 0.0
for i in range(0, 10):
    x_train_both, x_test_both, y_train_both, y_test_both = train_test_split(data['red'] + data['white'],
    y['red'] + y['white'], test_size = 0.3)

    lb = LassoCV(normalize=True).fit(x_train_both, y_train_both)

    alpha = lb.alpha_

    predicted_both = lr.predict(x_test_both)

    test_len_both = len(x_test_both)

    correct = 0

    for i in range(test_len_both):
        if(abs((y_test_both[i]) - predicted_both[i]) < 1):
            correct += 1

    print("BOTH:" + str((correct / test_len_both) * 100))
    print("ALPHA:" + str(alpha) + "\n")

    if(optimal < (correct / test_len_both) * 100):
        optimal = (correct / test_len_both) * 100
        optimal_alpha = alpha

print("OPTIMAL BOTH:" + str(optimal))
print("OPTIMAL ALPHA:" + str(optimal_alpha) + "\n")
