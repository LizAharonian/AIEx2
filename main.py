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
    decision_tree_model = Decision_tree_model_class()
    decision_tree_model.train()
    decision_tree_model.test()
    tree = decision_tree_model.decision_tree.print_tree(decision_tree_model.decision_tree.root)
    with open("liz.txt", 'w') as liz:
        liz.write(tree)
    pass







if __name__ == '__main__':
    main()