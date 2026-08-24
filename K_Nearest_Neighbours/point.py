class Point:
    def __init__(self, x: float, y: float, classification=None):
        self.x = x
        self.y = y
        self.__classification = classification

    def get_classification(self):
        return self.__classification

    def set_classification(self, classification: str):
        self.__classification = classification