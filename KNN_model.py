import utils as ut


"""
KNN class.
implementation of KNN algo.
"""
class KNN(object):
    def __init__(self,k):
        """
        constructor.
        :param k: number of nearest neighbors.
        """
        self.k = k
        pass

    def predict(self,predict_ex):
        """
        predict function.
        :param predict_ex: example to be tagged.
        :return: the predicted tag.
        """
        if len(ut.EXAMPLES_LIST) < self.k:
            return None
        hamming_distances = []
        for ex_dict, y in ut.EXAMPLES_LIST:
            hamming_distances.append((self.get_hamming_distance(ex_dict,predict_ex),y))
        hamming_distances.sort(key=lambda tup: tup[0])
        yes_counter = 0
        no_counter = 0
        for i in range(self.k):
            y = hamming_distances[i][1]
            if y==ut.YES:
                yes_counter += 1
            else:
                no_counter += 1
        return ut.YES if yes_counter > no_counter else ut.NO

    def get_hamming_distance(self,ex1_dict, ex2_dict):
        """
        get_hamming_distance function.
        :param ex1_dict: example 1 for comparison.
        :param ex2_dict: example 2 for comparison.
        :return: hamming distance.
        """
        hamming_dis = 0
        for feature in ut.FEATURES_LIST:
            if (ex1_dict[feature] != ex2_dict[feature]):
                hamming_dis += 1
        return hamming_dis

    def train(self):
        """
        train function.
        runs the model on the train data.
        :return: acc
        """
        y_list = []
        y_hat_list = []
        for ex_dict in ut.EXAMPLES_LIST:
            y_list.append(ex_dict[1])
            y_hat_list.append(self.predict(ex_dict[0]))

        acc = ut.compute_accuracy(y_hat_list, y_list)
        return acc

    def test(self):
        """
        test func.
        runs the model on the test data.
        :return: list of predictions and acc
        """
        y_list = []
        y_hat_list = []
        for ex_dict in ut.TEST_LIST:
            y_list.append(ex_dict[1])
            y_hat_list.append(self.predict(ex_dict[0]))
        acc = ut.compute_accuracy(y_hat_list, y_list)
        return y_hat_list, acc



