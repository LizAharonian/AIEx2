


class Decision_tree_node_class(object):
    def __init__(self,attribute, depth, is_leaf, pred, children_dict):
        self.attribute = attribute
        self.depth = depth
        self.is_leaf = is_leaf
        self.pred = pred
        self.children_dict = children_dict
