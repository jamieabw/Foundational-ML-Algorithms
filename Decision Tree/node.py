class Node:
    def __init__(self, node_condition: function):
        self.__node_condition = node_condition

    def evaluate(self, answer):
        return self.__node_condition(answer) # left for true, right for false