import utils as ut
from KNN_model import KNN



def main():

    examples_list = ut.EXAMPLES_LIST
    y = ut.Y
    knn_model = KNN(5)
    knn_model.train()
    knn_model.test()

    pass







if __name__ == '__main__':
    main()