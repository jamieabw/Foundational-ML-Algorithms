from math import e, log
class LogisticRegression:
    def __init__(self):
        self.gradient = 0
        self.bias = 0
        self.learning_rate = 0.01

    def sigmoid(self, x: float):
        return 1 / (1 + (e ** (-x)))

    def predict(self, x : float):
        return self.sigmoid((self.gradient * x) + self.bias)

    def binary_cross_entropy_loss(self, prediction, actual):
        sum = 0
        for p, a in zip(prediction, actual):
            p = max(min(p, 1 - 1e-15), 1e-15)
            sum += -((a * log(p) + ((1-a) * (log(1 - p)))))
        return sum / len(prediction)

    def loss_d_gradient(self, prediction, actual, data):
        sum = 0
        for p, a, d in zip(prediction, actual, data):
            sum += (p - a) * d
        return sum / len(prediction)

    def loss_d_bias(self, prediction, actual):
        sum = 0
        for p, a in zip(prediction, actual):
            sum += (p - a)
        return sum / len(prediction)

    def train(self, data):
        loss = 1
        while loss > 1e-4:
            inputs = [d[0] for d in data]
            actuals = [d[1] for d in data]
            predictions = [self.predict(d[0]) for d in data]
            loss = self.binary_cross_entropy_loss(predictions, actuals)
            self.gradient -= self.learning_rate * self.loss_d_gradient(predictions, actuals, inputs)
            self.bias -= self.learning_rate * self.loss_d_bias(predictions, actuals)
            print(loss)


t = LogisticRegression()
t.train([(1,1), (19,1), (-1, 0), (-15, 0), (0,1), (-4, 0), (4, 1), (6, 1), (8, 1)])
print(t.predict(-2))
