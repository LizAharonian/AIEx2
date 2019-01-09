import utils as ut
from Decision_tree_node import Decision_tree_node_class
from Decision_tree import Decision_tree_class
import math

"""
Decision_tree_model_class class.
implements dtl algo to create decision tree.
"""

class Decision_tree_model_class(object):
    def __init__(self):
        """
        constructor.
        """
        root = self.DTL(ut.EXAMPLES_LIST, ut.FEATURES_LIST,self.mode(ut.EXAMPLES_LIST),0)
        self.decision_tree = Decision_tree_class(root)

    def predict(self,predict_ex):
        """
        predict function.
        :param predict_ex: example we want to tag.
        :return: pred of example.
        """
        return self.decision_tree.discover_tree(predict_ex)


    def DTL(self, examples_tuple, attributes, default, depth):
        """
        dtl algo.
        creates decision tree.
        :param examples_tuple: list of tagged examples.
        :param attributes: set of features.
        :param default: default vals for stop conditions.
        :param depth: depth of node.
        :return: root of tree
        """
        if len(examples_tuple) == 0:
            return Decision_tree_node_class(None, depth, True, default, None)
        elif self.is_all_examples_tagged_the_same(examples_tuple):
            return Decision_tree_node_class(None, depth, True, examples_tuple[0][1], None)
        elif len(attributes) == 0:
            return Decision_tree_node_class(None, depth, True, self.mode(examples_tuple), None)
        else:
            best_att = self.choose_att(attributes,examples_tuple)
            if best_att == 'AI':
                pass
            child_att = attributes[:]
            child_att.remove(best_att)
            children_dict = {}
            root = Decision_tree_node_class(best_att,depth,False,None,children_dict)
            for feature_val in sorted(list(ut.FEATURES_VALS_DICT[best_att])):
                if feature_val=="91":
                    pass
                sub_examples = self.get_examples_of_the_specified_feature_val(examples_tuple, best_att, feature_val)
                children_dict[feature_val] = self.DTL(sub_examples,child_att ,self.mode(examples_tuple), depth + 1)
            return root
    def get_examples_of_the_specified_feature_val(self, examples_tuple, feature, feature_val):
        """
        get_examples_of_the_specified_feature_val function.
        :param examples_tuple: examples to be picked from.
        :param feature: feature we want the examples to have with specified val.
        :param feature_val: feature val we want to get examples with.
        :return: examples that their feature value is feature_val
        """
        sub_examples = [example_tuple for example_tuple in examples_tuple if example_tuple[0][feature] == feature_val]
        return  sub_examples

    def choose_att(self,left_att_list, examples_tuple):
        """
        choose_att function.
        :param left_att_list: set of features we want to pick one.
        :param examples_tuple: list of examples.
        :return: best att.
        """
        entropy = self.calculate_entropy(examples_tuple)
        max_feature = None
        max_gain = -1
        for feature in left_att_list:
            gain = self.compute_gate(feature, examples_tuple, entropy)
            if gain > max_gain:
                max_gain = gain
                max_feature = feature
        return max_feature

    def calculate_entropy(self, examples_tuple):
        """
        calculate_entropy function.
        :param examples_tuple: list of examples.
        :return: entropy calc
        """
        if not examples_tuple:
            return 0
        positive_prob = self.get_probability_of_tag(ut.YES, examples_tuple)
        negative_prob = self.get_probability_of_tag(ut.NO, examples_tuple)
        # mult_pos = 1
        # mult_neg = 1
        # if positive_prob > 0:
        #     mult_pos *= positive_prob*math.log(positive_prob, 2)
        # else:
        #     mult_pos = 0
        # if negative_prob > 0:
        #     mult_neg *= negative_prob*math.log(negative_prob, 2)
        # else:
        #     mult_neg = 0

        if positive_prob == 0.0 or negative_prob == 0.0:
            return 0

        return -positive_prob*math.log(positive_prob, 2) - negative_prob*math.log(negative_prob, 2)

    def get_probability_of_tag(self, tag, examples_tuple):
        """
        get_probability_of_tag function.
        :param tag: tag we want to know it's prob.
        :param examples_tuple: list of examples
        :return: prob of the specified tag.
        """
        counter = 0
        for ex in examples_tuple:
            if ex[1] == tag:
                counter += 1
        # if len(examples_tuple) == 0:
        #     return 0
        # else:
        return float(counter)/float(len(examples_tuple))


    def compute_gate(self,feature, examples_tuple, entropy):
        """
        compute_gate function.
        :param feature: feature the gain is calculated on.
        :param examples_tuple: list of examples.
        :param entropy: entropy calc.
        :return: gain of the feature.
        """
        sigma = 0
        for feature_val in sorted(list(ut.FEATURES_VALS_DICT[feature])):
            sub_ex = self.get_examples_of_the_specified_feature_val(examples_tuple, feature, feature_val)
            precent = float(len(sub_ex))/float(len(examples_tuple))
            sub_entropy = self.calculate_entropy(sub_ex)
            sigma += (precent * sub_entropy)
        return entropy - sigma


    def is_all_examples_tagged_the_same(self, examples_tuple):
        """
        is_all_examples_tagged_the_same function.
        :param examples_tuple: list of examples
        :return: true if all examples have the same tag, false otherwise.
        """
        tags = set()
        for example in examples_tuple:
            tag = example[1]
            tags.add(tag)
        return len(tags) == 1


    def mode(self,examples_tuple):
        """
        mode function.
        :param examples_tuple: list of examples.
        :return: the most common tag.
        """
        positive_counter = 0
        negative_counter = 0
        for example_tuple in examples_tuple:
            tag = example_tuple[1]
            if tag == ut.YES:
                positive_counter += 1
            else:
                negative_counter += 1

        return ut.YES if positive_counter >= negative_counter else ut.NO

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


