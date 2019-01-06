import utils as ut
from KNN_model import KNN
from Bayes_model import Bayes



def main():

    examples_list = ut.EXAMPLES_LIST
    y = ut.Y
    # knn_model = KNN(5)
    # knn_model.train()
    # knn_model.test()

    bayes_model = Bayes()
    bayes_model.train()
    bayes_model.test()

    pass







if __name__ == '__main__':
    main()