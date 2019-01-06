import utils as ut



class KNN(object):
    def __init__(self):
        pass

    def predict(self,predict_ex):




        pass
    def get_probability_for_positive_tag(self):

        pass
    def get_probability_for_negative_tag(self):

        pass

    def argmax(self,probs):
        return ut.YES if probs[0] >= probs[1] else ut.NO

    def get_probability_for_class(self,tag):
        total_examples_num = len(ut.EXAMPLES_LIST)



        pass




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
