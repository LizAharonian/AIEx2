from KNN_model import KNN
from Bayes_model import Bayes
from Decision_tree_model import Decision_tree_model_class



def main():
    """
    main function.
    runs the program.
    """
    knn_model = KNN(5)
    knn_res, knn_acc = knn_model.test()

    bayes_model = Bayes()
    bayes_res, bayes_acc = bayes_model.test()

    decision_tree_model = Decision_tree_model_class()
    dtl_res, dtl_acc = decision_tree_model.test()
    tree = decision_tree_model.decision_tree.print_tree(decision_tree_model.decision_tree.root)
    # write the output tree
    with open("output_tree.txt", 'w') as tree_output:
        tree_output.write(tree)
    # write the output file
    with open("output.txt", 'w') as output_file:
        file_lines = get_file_lines(knn_res,bayes_res,dtl_res)
        lines = []
        lines.append("Num\tDT\tKNN\tnaiveBase")
        lines += file_lines
        lines.append("\t" + str(dtl_acc) + "\t" + str(knn_acc) + "\t" + str(bayes_acc))
        output_file.writelines("\n".join(lines))


def get_file_lines(knn_res, bayes_res, dtl_res):
    """
    get_file_lines function.
    :param knn_res: knn preds
    :param bayes_res: bayes pred
    :param dtl_res: dtl preds
    :return: list of each line in the file
    """
    lines = []
    counter = 1
    for k, b, dtl in zip(knn_res, bayes_res, dtl_res):
        lines.append(str(counter) + "\t" + dtl + "\t" + k + "\t" +b)
        counter += 1
    return lines



if __name__ == '__main__':
    main()