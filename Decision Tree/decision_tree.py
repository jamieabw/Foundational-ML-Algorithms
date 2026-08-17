class Decision_Tree:
    def __init__(self):
        self.root = None

    def train(self, data):
        self.root = self.build_tree(data)

    def build_tree(self, data):
        ...

    def calculate_impurity(self, data):
        true_counter, false_counter = (0,0)
        for d in data:
            if d[-1] == 0:
                false_counter += 1
                continue
            true_counter += 1
        return 1 - ((true_counter/(true_counter + false_counter)) * (false_counter / (true_counter + false_counter)))

    def split_data(self, data, feature):
        # finds the best split.
        ...

    def predict(self, data):
        ...