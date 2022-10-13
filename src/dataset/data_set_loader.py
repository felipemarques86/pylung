import os
import pickle
from pathlib import Path
import logging
import numpy as np


def load_pre_processed_lidc_idri(image_size, base_path):
    if image_size != 128 and image_size != 256 and image_size != 512:
        raise Exception('image_size parameter must have the value: 128, 256 or 512. ' + str(image_size) +
                        ' is not a valid value')
    if base_path is None:
        raise Exception('base_path is mandatory')

    path = base_path + 'LIDC-IDRI-size-' + str(image_size) + os.sep
    path_high_res = base_path + 'LIDC-IDRI-size-512' + os.sep

    logging.debug('path=%s', path)
    logging.debug('path_high_res=%s', path_high_res)

    files_to_count = Path(path).glob('lidc_image_*')
    total_images = 0
    for file in files_to_count:
        total_images = total_images + 1

    logging.debug('Total images is %d', total_images)

    x_train = []
    x_test = []
    y_train = []
    y_test = []
    train_to = total_images * 0.8
    for i in range(0, total_images):
        image_filename = path + 'lidc_image_' + str(i) + '.pkl'
        with open(image_filename, 'rb') as filePointer:  # Overwrites any existing file.
            if len(x_train) < train_to:
                raw_img = pickle.load(filePointer)
                x_train.append(raw_img)
            else:
                image_filename = path_high_res + 'lidc_image_' + str(i) + '.pkl'
                with open(image_filename, 'rb') as filePointer2:  # Overwrites any existing file.
                    x_test.append(pickle.load(filePointer2))

        bbox_filename = path + 'lidc_scaled_box_' + str(i) + '.pkl'
        with open(bbox_filename, 'rb') as filePointer:  # Overwrites any existing file.
            if (len(y_train) < train_to):
                y_train.append(pickle.load(filePointer))
            else:
                bbox_filename = path_high_res + 'lidc_scaled_box_' + str(i) + '.pkl'
                with open(bbox_filename, 'rb') as filePointer2:  # Overwrites any existing file.
                    y_test.append(pickle.load(filePointer2))

    (xtest), (ytest) = (
        np.asarray(x_test),
        np.asarray(y_test),
    )

    # Convert the list to numpy array, split to train and test dataset
    ytrain = np.asarray(y_train)
    xtrain = np.asarray(x_train)

    return xtrain, ytrain, xtest, ytest
