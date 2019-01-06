
class Decision_tree_class(object):
    def __init__(self,root):
        self.root = root

    def discover_tree(self, example_dict):
        curr_node = self.root
        while not curr_node.is_leaf:
            for key, value in curr_node.children_dict.iteritems():
                if example_dict[curr_node.attribute] == key:
                    curr_node = value
                    break
        return curr_node.pred
