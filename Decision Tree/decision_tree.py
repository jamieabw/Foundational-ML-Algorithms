import numpy as np

class Decision_Tree:
    def __init__(self):
        self.root = None

    def train(self, data, labels):
        self.root = self.build_tree(data, labels)

    def build_tree(self, data):
        best_score = float("inf")
        best_feature = None
        best_threshold = None
        features = len(data[0]) # number of features to classify
        for i in range(features):
            ...
        

    def calculate_impurity(self, y):
        true_counter, false_counter = (0,0)
        for d in y:
            if d == 0:
                false_counter += 1
                continue
            true_counter += 1
        return 1 - (((true_counter/(true_counter + false_counter)) ** 2) +  ((false_counter / (true_counter + false_counter)) ** 2))

    def get_best_split(self, data):
        ...

    def split_data(self, data, feature, threshold):
        x, y = data
        left_mask = x[:, feature] < threshold
        right_mask = x[: feature] >= threshold
        x_left = x[left_mask]
        y_left = y[left_mask]
        x_right = x[right_mask]
        y_right = y[right_mask]
        return x_left, y_left, x_right, y_right

    def get_split_score(self, y_left, y_right):
        n = len(y_left) + len(y_right)
        left_weight = len(y_left) / n
        right_weight = len(y_right) / n 
        return (left_weight * self.calculate_impurity(y_left)) + (right_weight * self.calculate_impurity(y_right))    


    def predict(self, data):
        ...


    def get_mid_points(self, data: list):
        mid_points = []
        for i in range(len(data) -1):
            mid_points.append((data[i] + data[i+1]) / 2)
        return mid_points



training_data = [([5,8], 1),
                  ([1,2], 0)]

tree = Decision_Tree()
