from linear_regression import Linear_Regression

def main():
    linear_regression = Linear_Regression()
    points = [(0,1), (1,4), (3,10), (10, 31)] # y = 3x+1
    linear_regression.train(points)
    print(f"equation fitted as: y = {linear_regression.get_gradient()}x + {linear_regression.get_intercept()}") # fits as y = 3.0000017414392692x + 0.9999865886935222
    print(f"model predicts (2, {linear_regression.predict(2)})") 

if __name__ == "__main__":
    main()