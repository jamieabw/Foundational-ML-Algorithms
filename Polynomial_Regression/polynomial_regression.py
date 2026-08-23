class Polynomial_Regression:
    def __init__(self, degree: int):
        self.__degree = degree
        self.__gradients: list[float] = [1 for i in range(degree)]
        self.__intercept: float = 0
        self.__learning_rate: float = 0.00001

    def get_gradients(self) -> list[float]:
        """
        gets the current values being used for the gradients
        
        :param self: Description
        :return: the current values being used for the gradients
        :rtype: list[float]
        """
        return self.__gradients
    
    def get_intercept(self) -> float:
        """
        gets the current value being used for the intercept
        
        :param self: 
        :return: the current value being used for the intercept
        :rtype: float
        """
        return self.__intercept
    
    def get_learning_rate(self) -> float:
        """
        gets the current learning rate being used
        
        :param self: 
        :return: the current learning rate being used
        :rtype: float
        """
        return self.__learning_rate
    
    def set_learning_rate(self, learning_rate: float):
        """
        set a new learning rate
        
        :param self:
        :param learning_rate: the new learning rate to use
        :type learning_rate: float
        """
        self.__learning_rate = learning_rate

    def loss(self, predicted_points: list[tuple], points: list[tuple]) -> float:
        """
        gets the loss value given the known data points and the predicted data points
        
        :param self: 
        :param predicted_points: the list of points predicted by the algorithm
        :type predicted_points: list[tuple]
        :param points: the list of known data points used to fit the line
        :type points: list[tuple]
        :return: the loss value, how far the predictions are from the actual values
        :rtype: float
        """
        average: float = 0.0
        for i in range(len(predicted_points)):
            average += ((points[i][1] - predicted_points[i][1]) ** 2)
        return average / len(predicted_points)
    
    def loss_d_gradients(self, predicted_points: list[tuple], points: list[tuple]) -> list[float]:
        """
        finds the derivative of the loss with respect to the gradients, used to adjust the gradient values
        
        :param self:
        :param predicted_points: the list of points predicted by the alogrithm
        :type predicted_points: list[tuple]
        :param points: the list of known data points which are used to fit the line
        :type points: list[tuple]
        :return: the average of the derivatives of the loss with respect to the gradient
        :rtype: float
        """
        averages: list[float] = [0.0 for i in range(self.__degree)]
        for i in range(len(predicted_points)):
            for j in range(len(averages)):
                averages[j] += (-2 * (points[i][0] ** (j+1)) * (points[i][1] - predicted_points[i][1]))
        return [average / len(predicted_points) for average in averages]

    def loss_d_intercept(self, predicted_points: list[tuple], points: list[tuple]) -> float:
        """
        finds the derivative of the loss with respect to the intercept, used to adjust the intercept value 
        
        :param self: 
        :param predicted_points: the list of points predicted by the alogrithm
        :type predicted_points: list[tuple]
        :param points: the list of known data points which are used to fit the line
        :type points: list[tuple]
        :return: the average of the derivatives with respect to the intercept
        :rtype: float
        """
        average: float = 0.0
        for i in range(len(predicted_points)):
            average += (-2  * (points[i][1] - predicted_points[i][1]))
        return average / len(predicted_points)
    
    def predict(self, x: float) -> float:
        """
        given an x coordinate, returns the model's prediction for the y coordinate
        
        :param self: 
        :param x: x coordinate of the point you want to predict
        :type x: float
        :return: the predicted y coordinate associated with the given x coordinate
        :rtype: float
        """
        result: float = 0.0
        for i in range(len(self.__gradients)):
            result += ((x ** (i+1)) * self.__gradients[i])
        return result + self.__intercept
    
    def train(self, points: list[tuple], verbose=False):
        """
        Create a line of best fit through repeatedly adjusting the parameters using the loss until loss is below 1e-10
        
        :param self: 
        :param points: The given data points to fit the line to
        :type points: list[tuple]
        """
        loss = 1
        while loss > 1e-5:
            predicted_points: list[tuple] = []
            for point in points:
                x = point[0]
                predicted_points.append((x, self.predict(x)))
            if verbose:
                print(self.loss(predicted_points, points))
                print(predicted_points)
            gradient_adjustments: list[float] = self.loss_d_gradients(predicted_points, points)
            for i in range(len(self.__gradients)):
                self.__gradients[i] = self.__gradients[i] - (self.__learning_rate * gradient_adjustments[i])
            self.__intercept = self.__intercept - (self.__learning_rate * self.loss_d_intercept(predicted_points, points))
            loss = self.loss(predicted_points, points)
