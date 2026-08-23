import numpy as np
from node import Node

class Decision_Tree:
    def __init__(self):
        self.root = None

    def train(self, data: list[list], labels: list[int], max_depth: int):
        """Trains the decision tree by building from the root node, with a max depth.

        Args:
            data (list[list]): Data to train the data on
            labels (list[int]): Data labels to train the data on
            max_depth (int): The maximum depth of the decision tree to prevent overfitting
        """        
        self.max_depth = max_depth
        data = np.array(data)
        labels = np.array(labels)
        self.root = self.build_tree(data, labels)

    def build_tree(self, data: np.array, labels: np.array, depth=0) -> Node:
        """Recursively builds the tree, builds the left and right nodes and then the root node.

        Args:
            data (np.array): Subset of data to train the current node on
            labels (np.array): Subset of labels to train the current node on
            depth (int, optional): The depth of the current node. Defaults to 0.

        Returns:
            Node: The node with either the prediction or a question to compare the data to.
        """        
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
        

    def calculate_impurity(self, y: np.array) -> float:
        """Calculates the gini impurity score of the given nodes labels

        Args:
            y (np.array): The labels corresponding with the current node

        Returns:
            float: The gini impurity score of the given node
        """        
        true_counter, false_counter = (0,0)
        for d in y:
            if d == 0:
                false_counter += 1
                continue
            true_counter += 1
        return 1 - (((true_counter/(true_counter + false_counter)) ** 2) +  ((false_counter / (true_counter + false_counter)) ** 2))

    def get_best_split(self, data: np.array, labels: np.array) -> tuple:
        """Finds the feature and threshold combination which has the best split score to use as the node's question

        Args:
            data (np.array): Data to use the features and thresholds of
            labels (np.array): Labels to get the scores

        Returns:
            tuple: The best feature and threshold combination as a tuple
        """        
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

    def split_data(self, data: np.array, labels: np.array, feature: int, threshold: float) -> tuple:
        """Splits the given data and labels on a particular question determined by feature and threshold

        Args:
            data (np.array): Data to split based on question
            labels (np.array): Labels to split respectively with their data
            feature (int): Feature to compare to threshold
            threshold (float): The threshold value to compare the feature values to

        Returns:
            tuple: A tuple of the data and labels split
        """        
        left_mask = data[:, feature] < threshold
        right_mask = data[:, feature] >= threshold
        x_left = data[left_mask]
        y_left = labels[left_mask]
        x_right = data[right_mask]
        y_right = labels[right_mask]
        return x_left, y_left, x_right, y_right

    def get_split_score(self, y_left: np.array, y_right: np.array) -> float:
        """Finds the weighted average of the two children node's gini impurity scores

        Args:
            y_left (np.array): Labels belonging to left 
            y_right (np.array): Labels belong to right

        Returns:
            float: The split score associated with the node's gini impurity scores.
        """        
        n = len(y_left) + len(y_right)
        left_weight = len(y_left) / n
        right_weight = len(y_right) / n 
        return (left_weight * self.calculate_impurity(y_left)) + (right_weight * self.calculate_impurity(y_right))    


    def predict(self, data: np.array) -> int:
        """Uses the trained decision tree to classify the data based on its features and learnt thresholds.

        Args:
            data (np.array): Data to use to traverse the tree

        Returns:
            int: The classification predicted by the model.
        """        
        current_node = self.root
        while current_node.prediction is None:
            feature = current_node.feature
            threshold = current_node.threshold
            if data[feature] < threshold:
                current_node = current_node.left
            else:
                current_node = current_node.right 
        return current_node.prediction


    def get_mid_points(self, data: list[float]) -> list[float]:
        """Gets the midpoints between consecutive pieces of data in a sorted manner.

        Args:
            data (list[float]): Data to find the midpoints of

        Returns:
            list[float]: The midpoints of the provided data.
        """        
        data = sorted(set(data))
        mid_points = []
        for i in range(len(data) -1):
            mid_points.append((data[i] + data[i+1]) / 2)
        return mid_points

    def get_most_common_class(self, labels: np.array) -> int:
        """Gets the most common classification on a list of labels

        Args:
            labels (np.array): The list of labels to find the most common classification with

        Returns:
            int: The most common classification
        """        
        classes, counts = np.unique(labels, return_counts=True)
        return classes[np.argmax(counts)]

