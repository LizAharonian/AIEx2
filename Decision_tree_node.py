
"""
Decision_tree_node_class class.
represent node of the decision tree.
"""

class Decision_tree_node_class(object):
    def __init__(self,attribute, depth, is_leaf, pred, children_dict):
        """
        constructor.
        :param attribute: feature of node
        :param depth: depth of the node in the tree
        :param is_leaf: boolean - indicates if the node is a leaf
        :param pred: prediction in case the node is a leaf
        :param children_dict: pointers to children according to feature val.
        """
        self.attribute = attribute
        self.depth = depth
        self.is_leaf = is_leaf
        self.pred = pred
        self.children_dict = children_dict
