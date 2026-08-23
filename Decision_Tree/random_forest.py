from decision_tree import Decision_Tree

class Random_Forest:
    def __init__(self, forest_size, max_depth=5):
        self.forest_size = max(forest_size, 1)
        self.max_depth = max_depth
        self.forest = [Decision_Tree() for i in range(self.max_depth)]


    def train(self, data, labels):
        forest_training_data = []
        forest_training_data.append(data)
        