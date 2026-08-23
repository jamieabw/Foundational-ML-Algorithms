from logistic_regression import LogisticRegression

logistic_regression = LogisticRegression()
logistic_regression.train([(1,1), (19,1), (-1, 0), (-15, 0), (0,1), (-4, 0), (4, 1), (6, 1), (8, 1)]) # positive numbers yield 1, negative numbers yield 0
print(logistic_regression.predict(-2)) # should yield close to 0