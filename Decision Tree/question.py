class Question:
    def __init__(self, column, value):
        self.column = column # which column of the data areyou looking at (feature)
        self.value = value # value of the feature

    def is_numeric(self):
        try:
            int(self.value)
            return True
        except:
            return False


    def match(self, example):
        value = example[self.column]
        if self.is_numeric():
            return value >= self.value
        return value == self.value

    def __str__(self):
        condition = "=="
        if self.is_numeric():
            condition = "<="
        return f"is {self.column} {condition} {self.value}"
