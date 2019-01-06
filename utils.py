FEATURES_LIST = []
Y = None
EXAMPLES_LIST = []
TEST_LIST = []

EXAMPLES_DICT = {}

FEATURES_VALS_DICT = {}

YES = 'yes'
NO = 'no'

#todo: it can also be true\false! and not just yes\no - use set for distinct





def read_train_file(file_name):
    global FEATURES_LIST,Y,EXAMPLES_LIST, POSITIVE_EXAMPLES_NUM, NEGATIVE_EXAMPLES_NUM
    POSITIVE_EXAMPLES_NUM = 0
    NEGATIVE_EXAMPLES_NUM = 0
    FEATURES_LIST = []
    Y = None
    EXAMPLES_LIST = []
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
    TEST_LIST = []
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

def create_split_examples_dict():
    global EXAMPLES_DICT
    EXAMPLES_DICT = {}
    EXAMPLES_LIST[YES] = []
    EXAMPLES_LIST[NO] = []
    for ex_tuple in EXAMPLES_LIST:
        tag = ex_tuple[1]
        if tag == YES:
            EXAMPLES_LIST[YES].append(ex_tuple[0])
        else:
            EXAMPLES_LIST[NO].append(ex_tuple[0])


def create_tag_set():
    global YES, NO
    set_tags = set()
    for example_tuple in EXAMPLES_LIST:
        set_tags.add(example_tuple[1])
    for tag in set_tags:
        lower_tag = tag.lower()
        if lower_tag in ["t", "yes", "true", "1"]:
            YES = tag
        else:
            NO = tag

def create_features_sets():
    global FEATURES_VALS_DICT
    FEATURES_VALS_DICT = {}
    for feature in FEATURES_LIST:
        FEATURES_VALS_DICT[feature] = set()
    for example_tuple in EXAMPLES_LIST:
        example_dict = example_tuple[0]
        for feature in FEATURES_LIST:
            FEATURES_VALS_DICT[feature].add(example_dict[feature])







read_train_file('train.txt')
read_test_file('test.txt')
create_tag_set()
create_features_sets()
create_split_examples_dict()