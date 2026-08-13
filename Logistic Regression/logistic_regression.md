# Logistic Regression

Logistic regression uses similar terms to linear regression, notably the mx+c part, however, it then passes this overall value into a sigmoid function:
1 / (1 + e^-x) which maps all real numbers to a value between 0 and 1. Therefore, logistic regression is ideal for classification tasks.

Furthermore, another difference between logistic regression and linear regression is the loss function in use. Whilst linear regression uses the mean square loss function
which takes the difference between the predicted and actual value, squares it, then gets the mean average over the set of training data, logistic regression instead uses binary cross entropy loss, which is more effective for classification tasks, due to it having more significant gradients for incorrect classifications.