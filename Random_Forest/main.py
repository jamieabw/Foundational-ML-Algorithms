from Random_Forest.random_forest import Random_Forest
from random import randint

def main():
    print("Hello, World!")
    data, labels = gen_data()
    random_forest = Random_Forest(10, max_depth=7)
    random_forest.train(data, labels)
    print(random_forest.predict([10, 10, 10, 4]))


def gen_data():
    data = []
    labels = []
    for i in range(100):
        hours_studied = randint(0,12)
        hours_slept = randint(0, 12)
        days_missed = randint(0,12)
        previous_passes = randint(0,12)
        if (hours_studied + hours_slept + previous_passes) - days_missed >= 15:
            labels.append(1)
        else:
            labels.append(0)
        data.append([hours_studied, hours_slept, previous_passes, days_missed])
    return data, labels

if __name__ == "__main__":
    main()