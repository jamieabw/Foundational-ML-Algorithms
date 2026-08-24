from K_Nearest_Neighbours.point import Point
from math import sqrt
class K_Nearest_Neighbours:
    def __init__(self, k: int):
        self.k = k
        self.__points = []

    def train(self, data: list):
        """Adds every point and its classification to the tracked points

        Args:
            data (list): List of points to form clusters
        """        
        for point in data:
            self.__points.append(point)

    def predict(self, point: Point) -> str:
        """Predicts the classification of a point by finding the k nearest neighbouring points and seeing their classification, predicts classification by majority vote 
        of k neighbours

        Args:
            point (Point): The point to predict the classification for

        Returns:
            str: The classification of the given point
        """        
        k_nearest = sorted([(p, self.get_distance(point, p)) for p in self.__points], key=lambda a : a[1])[0:self.k]
        classification_counter = {}
        for neighbour in k_nearest:
            classification = neighbour[0].get_classification()
            if classification not in classification_counter:
                classification_counter[classification] = 1
            else:
                classification_counter[classification] += 1
        biggest = 0
        biggest_class = None
        for key, value in classification_counter.items():
            if value > biggest:
                biggest = value
                biggest_class = key
        return biggest_class


    def get_distance(self, point: Point, other: Point) -> float:
        """Finds the euclidean distance between 2 points

        Args:
            point (Point): First point
            other (Point): Second point

        Returns:
            float: The euclidean distance between the first and second points
        """        
        return sqrt(((point.x - other.x) ** 2) + ((point.y - other.y) ** 2))

