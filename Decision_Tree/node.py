class Node:
    def __init__(self, feature=None, threshold=None, prediction=None, numerical=None, left=None, right=None):
        self.feature = feature
        self.threshold = threshold
        self.numerical = numerical

        self.left = left
        self.right = right
        self.prediction = prediction

    def compare(self, data):
        if self.numerical:
            return data < self.threshold
        else:
            return data == self.threshold

        
