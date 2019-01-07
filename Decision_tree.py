"""
Decision_tree_class.
represent tree object created from dtl algo.
"""

class Decision_tree_class(object):
    def __init__(self,root):
        """
        constructor.
        :param root: root of the tree.
        """
        self.root = root

    def discover_tree(self, example_dict):
        """
        discover_tree function.
        :param example_dict: example we want to know its tag according to the tree.
        :return: pred
        """
        curr_node = self.root
        while not curr_node.is_leaf:
            for key in curr_node.children_dict:
                value = curr_node.children_dict[key]
                if example_dict[curr_node.attribute] == key:
                    curr_node = value
                    break
        return curr_node.pred

    def print_tree(self, node):
        """
        print_tree function.
        prints the tree.
        :param node:
        :return:string of the decision tree
        """
        string = ""
        for child in sorted(node.children_dict):
            string += node.depth * "\t"
            if node.depth > 0:
                string += "|"
            string += node.attribute + "=" + child
            if node.children_dict[child].is_leaf:
                string += ":" + node.children_dict[child].pred + "\n"
            else:
                string += "\n" + self.print_tree(node.children_dict[child])

        return string

