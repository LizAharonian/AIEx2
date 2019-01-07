import utils as ut

"""
Bayes class.
operates classification acording to bayes algo.
"""

class Bayes(object):
    """
    constructor.
    """
    def __init__(self):
        pass

    def predict(self,predict_ex):
        """
        predict function.
        returns tag of requested example
        :param predict_ex: requested example to operate the prediction on.
        :return: the predicted tag.
        """
        probs = []
        probs.append(self.get_probability_per_class(predict_ex, ut.YES))
        probs.append(self.get_probability_per_class(predict_ex, ut.NO))
        return self.argmax(probs)



    def get_probability_per_class(self, predict_ex, tag):
        """
        get_probability_per_class function.
        :param predict_ex: requested example to operate the prediction on.
        :param tag: the tag we want it's probability.
        :return:
        """
        multiply = 1
        for feature in ut.FEATURES_LIST:
            multiply *= self.get_probability_for_feature_per_class(feature, tag, predict_ex[feature])
        return multiply

    def get_probability_for_feature_per_class(self,feature, tag, feature_val):
        """
        get_probability_for_feature_per_class function.
        :param feature:the specified feature.
        :param tag:the tag we want it's prob.
        :param feature_val:feature val of feature to get probability of.
        :return:
        """
        examples_from_this_tag = ut.EXAMPLES_DICT[tag]
        count = 0
        for example_dict in examples_from_this_tag:
            if example_dict[feature] == feature_val:
                count +=1
        numerator = count + 1
        denominator = len(examples_from_this_tag) + len(ut.FEATURES_VALS_DICT[feature])
        return float(numerator)/float(denominator)


    def argmax(self,probs):
        """
        argmax function.
        fund the tag of maximum prob.
        :param probs: probs list of tags.
        :return: the tag it's prob is the biggest.
        """
        return ut.YES if probs[0] >= probs[1] else ut.NO

    def get_probability_for_class(self,tag):
        """
        get_probability_for_class function.
        :param tag: the tag we want it's prob.
        :return: prob of class
        """
        total_examples_num = len(ut.EXAMPLES_LIST)
        number_of_examples_of_this_tag = len(ut.EXAMPLES_DICT[tag])
        return float(number_of_examples_of_this_tag)/float(total_examples_num)



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
        return y_hat_list,acc

