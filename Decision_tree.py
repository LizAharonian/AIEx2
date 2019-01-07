
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

    def print_tree(self, node):
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

