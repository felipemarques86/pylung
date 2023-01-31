import configparser

import cv2
import numpy as np

from main.dataset_utilities.dataset_reader_classes import DatasetTransformer
from main.dataset_utilities.lidc_dataset_reader_classes import CustomLidcDatasetReader
from main.utilities.utilities_lib import two_one_hot_malignancy, LIDC_ANN_Y0, LIDC_ANN_Y1, LIDC_ANN_X0, LIDC_ANN_X1, \
    info

config = configparser.ConfigParser()

# check config.ini is in the root folder
config_file = config.read('config.ini')
if len(config_file) == 0:
    raise Exception("config.ini file not found")

BATCHSIZE = 50
CLASSES = 2
EPOCHS = 40
TRAIN_SIZE = 0.8
IMAGE_SIZE = 64

m0 = 0
m1 = 0

def resize(image, annotation):
    global m0, m1
    image = np.float32(image)
    im = cv2.resize(image[annotation[LIDC_ANN_Y0]:annotation[LIDC_ANN_Y1], annotation[LIDC_ANN_X0]:annotation[LIDC_ANN_X1]], (IMAGE_SIZE, IMAGE_SIZE))
    s = image[annotation[LIDC_ANN_Y0]:annotation[LIDC_ANN_Y1], annotation[LIDC_ANN_X0]:annotation[LIDC_ANN_X1]].shape
    print(s[0], s[1])
    m0 = max(m0, s[0])
    m1 = max(m1, s[1])
    return np.int16(im)


ds_root_folder = config['DATASET'][f'processed_lidc_idri_location']
dataset_reader = CustomLidcDatasetReader(location=ds_root_folder + '/main/')
dataset_reader.dataset_data_transformers.append(DatasetTransformer(function=two_one_hot_malignancy))
dataset_reader.dataset_image_transformers.append(DatasetTransformer(function=resize))
dataset_reader.load_custom()

x = dataset_reader.images
y = dataset_reader.annotations

(x_train), (y_train) = (
    x[: int(len(x) * TRAIN_SIZE)],
    y[: int(len(y) * TRAIN_SIZE)],
)
(x_valid), (y_valid) = (
    np.asarray(x[int(len(x) * TRAIN_SIZE):]),
    np.asarray(y[int(len(y) * TRAIN_SIZE):]),
)

dataset_reader.images = x_train
dataset_reader.annotations = y_train

dataset_reader.augment(0.1, 0.3)

(x_train), (y_train) = (
    np.asarray(dataset_reader.images),
    np.asarray(dataset_reader.annotations),
)

info(f'{m0}, {m1}')


