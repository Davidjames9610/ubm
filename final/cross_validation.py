

from sklearn.model_selection import RepeatedKFold
random_state = 12883823
rkf = RepeatedKFold(n_splits=4, n_repeats=1, random_state=random_state)
from sklearn.model_selection import train_test_split

TRAIN_DATA = 'train_data'
TRAIN_LABEL = 'train_label'
VAL_DATA = 'val_data'
VAL_LABEL = 'val_label'
TEST_DATA = 'test_data'
TEST_LABEL = 'test_label'

def split_data_for_cross_validation(data, labels, n_splits, test_ratio=0.2):
    """
    Parameters
    data - to be split
    n_splits - number of splits for k-fold
    test_ratio = 0.2, ratio of data for final testing

    Returns
    data_train, label_train, data_val, label_val, X_test, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_ratio, random_state=42) # 20 % of data held out for final testing
    random_state = None # 12883823
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=1, random_state=random_state)

    data_train = []
    data_val = []
    label_train = []
    label_val = []

    for train_indices, validate_indices in rkf.split(X_train):
        data_train.append([X_train[i] for i in train_indices])
        label_train.append([y_train[i] for i in train_indices])

        data_val.append([X_train[i] for i in validate_indices])
        label_val.append([y_train[i] for i in validate_indices])

    return {
        TRAIN_DATA: data_train,
        TRAIN_LABEL: label_train,
        VAL_DATA: data_val,
        VAL_LABEL: label_val,
        TEST_DATA: X_test,
        TEST_LABEL: y_test
    }