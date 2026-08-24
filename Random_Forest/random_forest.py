from Decision_Tree.decision_tree import Decision_Tree
from random import randint

class Random_Forest:
    def __init__(self, forest_size: int, max_depth=5):
        self.forest_size = max(forest_size, 1)
        self.max_depth = max_depth
        self.forest = [Decision_Tree() for i in range(self.forest_size)]


    def train(self, data: list, labels: list):
        """Trains each tree within the decision tree on a random subset of data with a random subset of features

        Args:
            data (list): Original data to train the forest
            labels (list): Original labels for data
        """        
        forest_training_data = []
        forest_training_labels = []
        #forest_training_data.append(data)
        #forest_training_labels.append(labels) # the first tree gets all features, all data
        indexes = range(len(data))
        for n in range(self.forest_size):
            feature_start = randint(0, len(data[0])-1)
            feature_end = randint(feature_start, len(data[0])-1)
            temp_data = []
            temp_labels = []
            for i in range(len(indexes)):
                random_data = randint(0, len(indexes)-1)
                #print(data[random_data])
                temp_data.append(data[random_data][feature_start:feature_end+1])
                temp_labels.append(labels[random_data])
            forest_training_data.append(temp_data)
            forest_training_labels.append(temp_labels)
        for tree_num in range(len(self.forest)):
            self.forest[tree_num].train(forest_training_data[tree_num], forest_training_labels[tree_num], self.max_depth)


    def predict(self, data: list) -> int:
        """Passes the data through each tree in the forest to get a classification, then uses majority voting to determine the classification

        Args:
            data (list): Data to predict a classification for

        Returns:
            int: The predicted classification that was majority voted
        """        
        prediction_counter = {}
        for tree in self.forest:
            prediction = int(tree.predict(data))
            if prediction not in prediction_counter:
                prediction_counter[prediction] = 1
            else:
                prediction_counter[prediction] += 1
        biggest = 0
        biggest_key = None
        for key, value in prediction_counter.items():
            if value > biggest:
                biggest = value
                biggest_key = key
        print(prediction_counter.items())
        return biggest_key




                
        