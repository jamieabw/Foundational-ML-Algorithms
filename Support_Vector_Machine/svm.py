import numpy as np
class SVM:
    def __init__(self, learning_rate: float, epochs: int, C: float):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.C = C
        self.w1 = 0
        self.w2 = 0
        self.b = 0

    def train(self, data: np.array, labels: np.array):
        """Trains the SVM to find the best decision boundary which maximises the margins

        Args:
            data (np.array): The data to train the SVM on
            labels (np.array): the labels associated to the data
        """        
        for i in range(self.epochs):
            scores = (
                self.w1 * data[:, 0] + self.w2 * data[:, 1] + self.b
            )
            margins = scores * labels
            violating = margins < 1 # those within the margin
            self.dw1, self.dw2 = self.w1, self.w2
            self.dw1 -= self.C * np.sum(labels[violating] * data[violating, 0])
            self.dw2 -= self.C * np.sum(labels[violating] * data[violating, 1])
            self.db = -self.C * np.sum(labels[violating])

            self.w1 -= self.learning_rate * self.dw1
            self.w2 -= self.learning_rate * self.dw2
            self.b -= self.learning_rate * self.db

    def predict(self, data: list) -> int:
        """Sees which side of the decision boundary the data point lies on, then classifies it based on that

        Args:
            data (list): data point to predict classifcation for

        Returns:
            int: predicted classification
        """        
        score = (data[0]) * self.w1 + (data[1] * self.w2) + self.b
        if score >= 0:
            return 1
        return -1
