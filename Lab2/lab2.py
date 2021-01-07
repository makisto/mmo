import csv
import math
import random
from collections import defaultdict
from collections import Counter
from typing import *

import numpy as np
from sklearn import tree
from sklearn.impute import SimpleImputer

class DataItem:
    def __init__(self, row: list):
        self.attributes = [None] * 13
        for i in range(len(self.attributes)):
            self.attributes[i] = None if row[i] == "?" else row[i]
        self.final_class = row[len(row) - 1]

    def __str__(self):
        return str(self.attributes) + ", " + str(self.final_class)

class DecisionTree:
    def __init__(self, train_mas: List[DataItem], trees_depth: int):
        n_attributes = len(train_mas[0].attributes)

        self.attr_subset_indexes = random.sample(range(n_attributes), trees_depth)
        attribute_subset = [subset_by_indexes(item.attributes, self.attr_subset_indexes) for item in train_mas]

        classes = [item.final_class for item in train_mas]

        self.sk_dec_tree = tree.DecisionTreeClassifier()
        self.sk_dec_tree.fit(attribute_subset, classes)

    def predict(self, X) -> list:
        X = [subset_by_indexes(attrs, self.attr_subset_indexes) for attrs in X]
        return self.sk_dec_tree.predict(X)

class decision_forest:
    def __init__(self, decision_trees: List[DecisionTree]):
        self.decision_trees = decision_trees

    def predict(self, X) -> list:
        vote_table = zip(*[dec_tree.predict(X) for dec_tree in self.decision_trees])
        return [Counter(votes).most_common(1)[0][0] for votes in vote_table]

def read_csv_file(file: str) -> List[DataItem]:
    with (open(file)) as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        return [DataItem(row) for row in reader]

def fill_missing_attributes(data: List[DataItem]) -> List[DataItem]:
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    attributes = [item.attributes for item in data]
    imp = imp.fit_transform(attributes)
    return [DataItem([*imp[i], data[i].final_class]) for i in range(len(data))]

def initialize():
    full_data_mas = read_csv_file('heart_data.csv')
    full_data_mas = fill_missing_attributes(full_data_mas)
    return full_data_mas

def subset_by_indexes(l: list, indexes: List[int]) -> list:
    return [l[i] for i in indexes]

def split_data(data: List[DataItem], train_mas_ratio: float) -> Tuple[List[DataItem], List[DataItem]]:
    n = len(data)
    n_training = int(n * train_mas_ratio)

    grouped_by_class = defaultdict(list)
    for entry in data:
        grouped_by_class[entry.final_class].append(entry)
    classes = list(grouped_by_class.keys())

    train_mas = []
    test_mas = []

    class_i = 0
    for i in range(n):
        class_i = (class_i + 1) % len(classes)
        final_class = classes[class_i]
        group = grouped_by_class[final_class]
        if len(group) == 0:
            classes.remove(final_class)
            continue

        group.pop
        item = group.pop(random.randint(0, len(group) - 1))
        target_data = train_mas if i < n_training else test_mas
        target_data.append(item)

    return train_mas, test_mas

def main():
    optimal_iteration = 0
    optimal_trees = 0
    optimal_depth = 0
    optimal_average = 0
    iterations = 1
    while iterations <= 10:
        average = 0
        full_data_mas = initialize()
        (train_mas, test_mas) = split_data(full_data_mas, 0.7)
        trees_count = 10
        while trees_count <= 50:
            trees_depth = 2
            while trees_depth <= 8:
                correct_precisions = 0
                decision_trees = [DecisionTree(train_mas, trees_depth) for i in range(trees_count)]
                decision_trees_forest = decision_forest(decision_trees)
                predicted: list = decision_trees_forest.predict([item.attributes for item in test_mas])
                for i in range(len(predicted)):
                    if predicted[i] == test_mas[i].final_class:
                        correct_precisions = correct_precisions + 1
                print("Iteration:" + str(iterations) + " Trees:" + str(trees_count) + " Depth:" + str(trees_depth) +" Result:" + str(correct_precisions / len(test_mas)))
                if correct_precisions / len(test_mas) > optimal_average:
                    optimal_iteration = iterations
                    optimal_average = (correct_precisions / len(test_mas))
                    optimal_trees = trees_count
                    optimal_depth = trees_depth
                average += (correct_precisions / len(test_mas))
                trees_depth += 1
            trees_count += 10
        print("Average for Iteration" + str(iterations) + " " + str(average / 350))
        iterations += 1
    print("Optmal parametrs:" + "Iteration: " + str(optimal_iteration) + " Trees: " + str(optimal_trees) + " Depth: " + str(optimal_depth) + " Precisions: " + str(optimal_average))


if __name__ == '__main__':
    main()
    exit()
