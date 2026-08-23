import numpy as np
from node import Node

class Decision_Tree:
    def __init__(self):
        self.root = None

    def train(self, data, labels, max_depth):
        self.max_depth = max_depth
        data = np.array(data)
        labels = np.array(labels)
        self.root = self.build_tree(data, labels)

    def build_tree(self, data, labels, depth=0):
        if len(np.unique(labels)) == 1: # first stopping condition
            return Node(prediction=labels[0])
        if depth >= self.max_depth:
            return Node(prediction=self.get_most_common_class(labels))
        # second stopping condition needs to be max depth when implemented
        feature, threshold = self.get_best_split(data, labels)
        if feature is None:
            return Node(prediction=self.get_most_common_class(labels))
        x_left, y_left, x_right, y_right = self.split_data(data, labels, feature, threshold)
        left = self.build_tree(x_left, y_left, depth=depth+1)
        right = self.build_tree(x_right, y_right, depth=depth + 1)
        return Node(feature=feature, threshold=threshold, left=left, right=right)
        

    def calculate_impurity(self, y):
        true_counter, false_counter = (0,0)
        for d in y:
            if d == 0:
                false_counter += 1
                continue
            true_counter += 1
        return 1 - (((true_counter/(true_counter + false_counter)) ** 2) +  ((false_counter / (true_counter + false_counter)) ** 2))

    def get_best_split(self, data, labels):
        best_feature = None
        best_threshold = None
        best_score = float("inf")
        features = len(data[0]) # use first piece of data as a reference for number of features
        for feature in range(features):
            temp = []
            for d in data:
                temp.append(d[feature]) #
            mid_points = self.get_mid_points(temp)
            for threshold in mid_points:
                x_left, y_left, x_right, y_right = self.split_data(data, labels, feature, threshold)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                
                score = self.get_split_score(y_left, y_right)
                if score < best_score:
                    best_score = score
                    best_threshold = threshold
                    best_feature = feature
        return best_feature, best_threshold

    def split_data(self, data, labels, feature, threshold):
        left_mask = data[:, feature] < threshold
        right_mask = data[:, feature] >= threshold
        x_left = data[left_mask]
        y_left = labels[left_mask]
        x_right = data[right_mask]
        y_right = labels[right_mask]
        return x_left, y_left, x_right, y_right

    def get_split_score(self, y_left, y_right):
        n = len(y_left) + len(y_right)
        left_weight = len(y_left) / n
        right_weight = len(y_right) / n 
        return (left_weight * self.calculate_impurity(y_left)) + (right_weight * self.calculate_impurity(y_right))    


    def predict(self, data):
        current_node = self.root
        while current_node.prediction is None:
            feature = current_node.feature
            threshold = current_node.threshold
            if data[feature] < threshold:
                current_node = current_node.left
            else:
                current_node = current_node.right 
        return current_node.prediction


    def get_mid_points(self, data: list):
        data = sorted(set(data))
        mid_points = []
        for i in range(len(data) -1):
            mid_points.append((data[i] + data[i+1]) / 2)
        return mid_points

    def get_most_common_class(self, labels):
        classes, counts = np.unique(labels, return_counts=True)
        return classes[np.argmax(counts)]

