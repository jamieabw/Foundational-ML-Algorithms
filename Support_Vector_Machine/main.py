from Support_Vector_Machine.svm import SVM
import numpy as np
def main():
    svm = SVM(0.01, 100, 1.0)
    data = np.array([
    [1, 1],
    [1, 2],
    [2, 1],
    [2, 2],

    [6, 6],
    [6, 7],
    [7, 6],
    [7, 7]
                    ])

    labels = np.array([
        -1,
        -1,
        -1,
        -1,

        1,
        1,
        1,
        1
                    ])
    svm.train(data, labels)
    print(svm.predict([-1.5, 3])) # correctly predicts -1
    

if __name__ == "__main__":
    main()