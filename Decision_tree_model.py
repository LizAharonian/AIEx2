import utils as ut
from Decision_tree_node import Decision_tree_node_class
from Decision_tree import Decision_tree_class
import math



class Decision_tree_model_class(object):
    def __init__(self):
        root = self.DTL(ut.EXAMPLES_LIST, ut.FEATURES_LIST,self.mode(ut.EXAMPLES_LIST),0)
        self.decision_tree = Decision_tree_class(root)

    def predict(self,predict_ex):
        return self.decision_tree.discover_tree(predict_ex)


    def DTL(self, examples_tuple, attributes, default, depth):
        if len(examples_tuple) == 0:
            return Decision_tree_node_class(None, depth, True, default, None)
        elif self.is_all_examples_tagged_the_same(examples_tuple):
            return Decision_tree_node_class(None, depth, True, examples_tuple[0][1], None)
        elif len(attributes) == 0:
            return Decision_tree_node_class(None, depth, True, examples_tuple[0][1], None)
        else:
            best_att = set.choose_att(attributes,examples_tuple)
            children_dict = {}
            root = Decision_tree_node_class(best_att,depth,False,None,children_dict)
            for feature_val in ut.FEATURES_VALS_DICT[best_att]:
                sub_examples = self.get_examples_of_the_specified_feature_val(examples_tuple, best_att, feature_val)
                children_dict[feature_val] = self.DTL(sub_examples, attributes.remove(best_att),self.mode(examples_tuple), depth + 1)
            return root
    def get_examples_of_the_specified_feature_val(self, examples_tuple, feature, feature_val):
        sub_examples = [example_tuple[0] for example_tuple in examples_tuple if example_tuple[0][feature] == feature_val]
        return  sub_examples

    def choose_att(self,left_att_list, examples_tuple):
        entropy = self.calculate_entropy()
        gain_list = []
        for feature in left_att_list:
            gain = self.compute_gate(feature, examples_tuple, entropy)
            gain_list.append((feature, gain))
        gain_list.sort(key=lambda tup: tup[0])
        return gain_list.pop()[1]

    def calculate_entropy(self, examples_tuple):
        positive_prob = self.get_probability_of_tag(ut.YES, examples_tuple)
        negative_prob = self.get_probability_of_tag(ut.NO, examples_tuple)
        return -positive_prob*math.log(positive_prob, 2.0) -negative_prob*math.log(negative_prob, 2.0)

    def get_probability_of_tag(self, tag, examples_tuple):
        counter = 0
        for ex in examples_tuple:
            if ex[1] == tag:
                counter += 1
        return float(counter)/float(len(examples_tuple))


    def compute_gate(self,feature, examples_tuple, entropy):
        sigma = 0
        for feature_val in ut.FEATURES_VALS_DICT[feature]:
            sub_ex = self.get_examples_of_the_specified_feature_val(examples_tuple, feature, feature_val)
            precent = float(len(sub_ex))/float(len(ut.EXAMPLES_LIST))
            sub_entropy = self.calculate_entropy(examples_tuple)
            sigma += precent * sub_entropy
        return entropy - sigma


    def is_all_examples_tagged_the_same(self, examples_tuple):
        tags = set()
        for example in examples_tuple:
            tag = example[1]
            tags.add(tag)
        return len(tags) == 1


    def mode(self,examples_tuple):
        positive_counter = 0
        negative_counter = 0
        for example_tuple in examples_tuple:
            tag = example_tuple[0]
            if tag == ut.YES:
                positive_counter += 1
            else:
                negative_counter += 1
        return ut.YES if positive_counter >= negative_counter else ut.NO




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
