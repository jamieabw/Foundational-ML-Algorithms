class Node:
    def __init__(self, feature, threshold):
        self.feature = feature
        self.threshold = threshold

        self.left = None
        self.right = None
        self.prediction = None
        
