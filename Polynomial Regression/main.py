from polynomial_regression import Polynomial_Regression
def main():
    polynomial_regression = Polynomial_Regression(3) # will try to fit onto 2x^3 - 4x^2 - 5x + 1
    points = [(2,-9), (5,126), (7, 456), (-2, -21), (-7, -846)]
    polynomial_regression.train(points, False)
    print(polynomial_regression.predict(10)) # should be ~1551, comes out as 1551.0022163436104
    print(polynomial_regression.get_gradients())
    print(polynomial_regression.get_intercept())

if __name__ == "__main__":
    main()