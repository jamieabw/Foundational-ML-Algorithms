class Linear_Regression:
    def __init__(self):
        self.__gradient: float = 0.0
        self.__intercept: float = 0.0
        self.__learning_rate: float = 0.001

    def get_gradient(self) -> float:
        """
        gets the current value being used for the gradient
        
        :param self: Description
        :return: the current value being used for the gradient
        :rtype: float
        """
        return self.__gradient
    
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

    def loss_d_gradient(self, predicted_points: list[tuple], points: list[tuple]) -> float:
        """
        finds the derivative of the loss with respect to the gradient, used to adjust the gradient value
        
        :param self:
        :param predicted_points: the list of points predicted by the alogrithm
        :type predicted_points: list[tuple]
        :param points: the list of known data points which are used to fit the line
        :type points: list[tuple]
        :return: the average of the derivatives of the loss with respect to the gradient
        :rtype: float
        """
        average: float = 0.0
        for i in range(len(predicted_points)):
            average += (-2 * points[i][0] * (points[i][1] - predicted_points[i][1]))
        return average / len(predicted_points)

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
        return (self.__gradient * x) + self.__intercept
    
    def train(self, points: list[tuple], verbose=False):
        """
        Create a line of best fit through repeatedly adjusting the parameters using the loss until loss is below 1e-10
        
        :param self: 
        :param points: The given data points to fit the line to
        :type points: list[tuple]
        """
        loss = 1
        while loss > 1e-10:
            predicted_points: list[tuple] = []
            for point in points:
                x = point[0]
                predicted_points.append((x, self.predict(x)))
            if verbose:
                print(self.loss(predicted_points, points))
                print(predicted_points)
            self.__gradient = self.__gradient - (self.__learning_rate * self.loss_d_gradient(predicted_points, points))
            self.__intercept = self.__intercept - (self.__learning_rate * self.loss_d_intercept(predicted_points, points))
            loss = self.loss(predicted_points, points)
