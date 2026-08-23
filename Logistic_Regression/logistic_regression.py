from math import e, log
class LogisticRegression:
    def __init__(self):
        self.gradient = 0
        self.bias = 0
        self.learning_rate = 0.01

    def sigmoid(self, x: float) -> float:
        """represents the sigmoid function

        Args:
            x (float): input to the sigmoid function

        Returns:
            float: the output of the sigmoid function with input x
        """        
        return 1 / (1 + (e ** (-x)))

    def predict(self, x : float) -> float:
        """uses the model to predict the classification of a piece of data

        Args:
            x (float): the piece of data which will be classified

        Returns:
            float: the classification of the data (round to the nearest whole number)
        """        
        return self.sigmoid((self.gradient * x) + self.bias)

    def binary_cross_entropy_loss(self, prediction: float, actual: int) -> float:
        """the binary cross entropy loss equation for classification

        Args:
            prediction (float): what the model predicted the classification as
            actual (int): what the actual classification is in the labeled data

        Returns:
            float: the loss value corresponding to the prediction of the model and the actual data supplied
        """        
        sum = 0
        for p, a in zip(prediction, actual):
            p = max(min(p, 1 - 1e-15), 1e-15)
            sum += -((a * log(p) + ((1-a) * (log(1 - p)))))
        return sum / len(prediction)

    def loss_d_gradient(self, prediction: float, actual: int, data: int) -> float:
        """the derivative of the loss with respect to the weight/gradient

        Args:
            prediction (float): what the model predicted
            actual (int): the actual classification
            data (int): the original data

        Returns:
            float: the mean derivative value using these inputs
        """        
        sum = 0
        for p, a, d in zip(prediction, actual, data):
            sum += (p - a) * d
        return sum / len(prediction)

    def loss_d_bias(self, prediction: float, actual: int) -> float:
        """the derivative of the loss with respect to the bias (intercept)

        Args:
            prediction (float): what the model predicted
            actual (int): the actual classification

        Returns:
            float: the mean derivative value using the inputs
        """        
        sum = 0
        for p, a in zip(prediction, actual):
            sum += (p - a)
        return sum / len(prediction)

    def train(self, data: list[tuple]):
        """trains the logistic regression by using gradient descent to find the optimal minimum values for the weight and bias to minimise the loss

        Args:
            data (list[tuple]):a data array composed of tuples of (data, classification)
        """        
        loss = 1
        while loss > 1e-4:
            inputs = [d[0] for d in data]
            actuals = [d[1] for d in data]
            predictions = [self.predict(d[0]) for d in data]
            loss = self.binary_cross_entropy_loss(predictions, actuals)
            self.gradient -= self.learning_rate * self.loss_d_gradient(predictions, actuals, inputs)
            self.bias -= self.learning_rate * self.loss_d_bias(predictions, actuals)
            print(loss)
