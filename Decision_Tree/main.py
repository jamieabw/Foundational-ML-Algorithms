from Decision_Tree.decision_tree import Decision_Tree
def main():
    decision_tree = Decision_Tree()
    training_data = [
    [1, 5],
    [2, 6],
    [2, 8],
    [3, 5],
    [4, 7],
    [5, 6],
    [6, 8],
    [7, 7],
    [8, 9],
    [9, 8],
    [10, 7],
    [11, 9],
    [3,9],
    [4,10],
    [2,10],
                ] # represents hours studied, hours slept
    training_labels = [
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1
                    ] # represents fail/pass test
    decision_tree.train(training_data, training_labels, 5)
    test_case = [5,8] # should pass
    print(decision_tree.predict(test_case)) # outputs 1 correctly


if __name__ == "__main__":
    main()