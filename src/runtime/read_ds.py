import numpy as np

from common.ds_reader import load_ds_single_file

def populate_data(normalization_func, bbox_handler, xtrain, xtrain_1, ytrain, ytrain_1, xtest, xtest_1, ytest, ytest_1):
    train_size = xtrain_1.shape[0]-1
    test_size = xtest_1.shape[0]-1
    for i in range(0, train_size):
        xtrain_1[i] = normalization_func(xtrain_1[i])
        ytrain_1[i] = bbox_handler(ytrain_1[i])

    for i in range(0, test_size):
        xtest_1[i] = normalization_func(xtest_1[i])
        ytest_1[i] = bbox_handler(ytest_1[i])

    xtrain = np.append(xtrain, xtrain_1, axis=0)
    ytrain = np.append(ytrain, ytrain_1, axis=0)

    xtest = np.append(xtest, xtest_1, axis=0)
    ytest = np.append(ytest, ytest_1, axis=0)

    return xtrain, ytrain, xtest, ytest


def read_lidc_dataset(normalization_func=lambda x: x, box_reader=lambda bbox: (bbox[0], bbox[1], bbox[2], bbox[3])):

    x_train_1, y_train_1, x_test_1, y_test_1 = load_ds_single_file(512, 0, 1, 'consensus-pt-1')

    x_shape = (0, 512, 512)
    y_shape = (0, 4)

    x_train = np.empty(x_shape)
    y_train = np.empty(y_shape)
    x_test = np.empty(x_shape)
    y_test = np.empty(y_shape)

    x_train, y_train, x_test, y_test = populate_data(normalization_func, box_reader,  x_train, x_train_1, y_train, y_train_1, x_test, x_test_1,
                                                     y_test, y_test_1)

    del x_train_1, y_train_1
    del x_test_1, y_test_1

    x_train_2, y_train_2, x_test_2, y_test_2 = load_ds_single_file(512, 0, 1, 'consensus-pt-2')

    x_train, y_train, x_test, y_test = populate_data(normalization_func, box_reader, x_train, x_train_2, y_train, y_train_2, x_test, x_test_2,
                                                     y_test, y_test_2)

    del x_train_2, y_train_2
    del x_test_2, y_test_2

    x_train_3, y_train_3, x_test_3, y_test_3 = load_ds_single_file(512, 0, 1, 'consensus-pt-3')

    x_train, y_train, x_test, y_test = populate_data(normalization_func, box_reader, x_train, x_train_3, y_train, y_train_3, x_test, x_test_3,
                                                     y_test, y_test_3)

    del x_train_3, y_train_3
    del x_test_3, y_test_3

    x_train_4, y_train_4, x_test_4, y_test_4 = load_ds_single_file(512, 0, 1, 'consensus-pt-4')

    x_train, y_train, x_test, y_test = populate_data(normalization_func, box_reader, x_train, x_train_4, y_train, y_train_4, x_test, x_test_4,
                                                     y_test, y_test_4)

    del x_train_4, y_train_4
    del x_test_4, y_test_4

    return x_train, y_train, x_test, y_test
