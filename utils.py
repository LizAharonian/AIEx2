FEATURES_LIST = []
Y = None
EXAMPLES_LIST = []
TEST_LIST = []

YES = 'yes'
NO = 'no'

#todo: it can also be true\false! and not just yes\no





def read_train_file(file_name):
    global FEATURES_LIST,Y,EXAMPLES_LIST
    is_first_line = True
    with open(file_name) as train_file:
        content = train_file.readlines()
        for line in content:
            if is_first_line:
                FEATURES_LIST = line.split('\t')
                Y = FEATURES_LIST.pop().strip('\n').strip('\r')
                is_first_line = False
            else:
                example_vals = line.split('\t')
                y_val = example_vals.pop().strip('\n').strip('\r')
                example = {feature : val for feature, val in zip(FEATURES_LIST,example_vals)}
                EXAMPLES_LIST.append((example,y_val))

def read_test_file(file_name):
    global TEST_LIST
    is_first_line = True
    with open(file_name) as test_file:
        content = test_file.readlines()
        for line in content:
            if is_first_line:
                is_first_line = False
            else:
                test_list = line.split('\t')
                y_val = test_list.pop().strip('\n').strip('\r')
                example = {feature: val for feature, val in zip(FEATURES_LIST, test_list)}
                TEST_LIST.append((example, y_val))


def compute_accuracy(y_hat_list, y_list):
    good = 0
    for y_hat, y in zip(y_hat_list, y_list):
        if y == y_hat:
            good += 1
    return round(float(good) / float(len(y_hat_list)) * 100,2)





read_train_file('train.txt')
read_test_file('test.txt')