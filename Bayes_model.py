import utils as ut



class Bayes(object):
    def __init__(self):
        pass

    def predict(self,predict_ex):
        probs = []
        probs.append(self.get_probability_per_class(predict_ex, ut.YES))
        probs.append(self.get_probability_per_class(predict_ex, ut.NO))
        return self.argmax(probs)



    def get_probability_per_class(self, predict_ex, tag):
        multiply = 1
        for feature in ut.FEATURES_LIST:
            multiply *= self.get_probability_for_feature_per_class(feature, tag, predict_ex[feature])
        return multiply

    def get_probability_for_feature_per_class(self,feature, tag, feature_val):
        examples_from_this_tag = ut.EXAMPLES_DICT[tag]
        count = 0
        for example_dict in examples_from_this_tag:
            if example_dict[feature] == feature_val:
                count +=1
        numerator = count + 1
        denominator = len(examples_from_this_tag) + len(ut.FEATURES_VALS_DICT[feature])
        return float(numerator)/float(denominator)


    def argmax(self,probs):
        return ut.YES if probs[0] >= probs[1] else ut.NO

    def get_probability_for_class(self,tag):
        total_examples_num = len(ut.EXAMPLES_LIST)
        number_of_examples_of_this_tag = len(ut.EXAMPLES_DICT[tag])
        return float(number_of_examples_of_this_tag)/float(total_examples_num)



    def train(self):
        y_list = []
        y_hat_list = []
        for ex_dict in ut.EXAMPLES_LIST:
            y_list.append(ex_dict[1])
            y_hat_list.append(self.predict(ex_dict[0]))

        acc = ut.compute_accuracy(y_hat_list, y_list)
        print ('accuracy on train is: ' + str(acc))

    def test(self):
        y_list = []
        y_hat_list = []
        for ex_dict in ut.TEST_LIST:
            y_list.append(ex_dict[1])
            y_hat_list.append(self.predict(ex_dict[0]))
        acc = ut.compute_accuracy(y_hat_list, y_list)
        print ('accuracy on test is: ' + str(acc))
        return y_hat_list,acc

