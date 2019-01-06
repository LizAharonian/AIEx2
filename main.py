import utils as ut
from KNN_model import KNN
from Bayes_model import Bayes
from Decision_tree_model import Decision_tree_model_class



def main():

    examples_list = ut.EXAMPLES_LIST
    y = ut.Y
    # knn_model = KNN(5)
    # knn_model.train()
    # knn_model.test()

    # bayes_model = Bayes()
    # bayes_model.train()
    # bayes_model.test()
    decision_tree = Decision_tree_model_class()
    decision_tree.train()
    decision_tree.test()
    pass







if __name__ == '__main__':
    main()