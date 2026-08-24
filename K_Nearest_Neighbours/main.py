from K_Nearest_Neighbours.point import Point
from K_Nearest_Neighbours.k_nearest_neighbours import K_Nearest_Neighbours
def main():
    data = [Point(0,0, "Cat"), Point(1,1,"Cat"), Point(-1,-1,"Cat"), Point(0,1, "Cat"), Point(10,10,"Dog"), Point(12,12,"Dog"), Point(7,7,"Dog")]
    test_point = Point(6,6)
    k_nearest_neighbours = K_Nearest_Neighbours(2)
    k_nearest_neighbours.train(data)
    print(k_nearest_neighbours.predict(test_point)) # correctly predicts Dog


if __name__ == "__main__":
    main()