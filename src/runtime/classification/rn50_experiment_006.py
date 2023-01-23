import pickle
import random
import time

import cv2
import numpy as np
from keras import Sequential
from keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import resnet50

from common.ds_reader import get_ds_single_file_name
from common.experiment import print_history, print_results_classification, run_experiment_cce, shuffle_data

DROP_OUT_1 = 0.4
TRAIN_SIZE = 0.9
TEST_SIZE = 1 - TRAIN_SIZE
IMAGE_SIZE = 224
SCAN_COUNT_PERC = 1
PAD = 0
TRAIN = True
history = []
input_shape = (IMAGE_SIZE, IMAGE_SIZE, 1)  # input image shape
learning_rate = 0.000001
weight_decay =  0.00001
batch_size = 250
num_epochs = 1000


start_time = time.perf_counter()

MIN_BOUND = -1000.0
MAX_BOUND = 600.0

def populate_data(normalization_func, bbox_handler, images, annotations, images_1, annotations_1):
    for i in range(0, len(images_1)):
        images.append(normalization_func(images_1[i]))
        annotations.append(bbox_handler(annotations_1[i]))

    return images, annotations


def load_ds(image_size, pad, scan_count_perc, type):
    with open(get_ds_single_file_name(image_size, pad, scan_count_perc, type), 'rb') as filePointer:
        data = pickle.load(filePointer)
    return data


def read_lidc_dataset(normalization_func=lambda x: x, box_reader=lambda bbox: (int(bbox[0]), int(bbox[1]), int(bbox[2]),
                                                                               int(bbox[3]))):
    images = []
    annotations = []

    for i in range(0, 10):
        images_1 = load_ds(IMAGE_SIZE, 0, 1, 'images-pt-' + str(i))
        annotations_1 = load_ds(IMAGE_SIZE, 0, 1, 'annotations-pt-' + str(i))
        images, annotations = populate_data(normalization_func, box_reader, images, annotations, images_1, annotations_1)

    images, annotations = shuffle_data(images, annotations)

    return images, annotations


def normalize(im):
    return im

# normalize an LIDC-IDRI image to grayscale and resize it
def normalize2(im):

    im = np.float32(im)
    # im[im < MIN_BOUND] = -1000
    # im[im > MAX_BOUND] = 600
    # im = (im + 1000) / (600 + 1000)
    # im = im * 255

    #image = Image.fromarray(im)

    image = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    #image = keras.utils.img_to_array(image)



    return np.int8(image)
    #return im
    #gray_normalized = im.clip(0, 80) / 80 * 255
    #rgb = np.squeeze(np.stack([im] * 3, axis=2))
    #dim = np.zeros((224, 224, 1))
    #return np.stack((im, dim, dim), axis=2)
    #keras.utils.img_to_array(Image.fromarray(im, 'RGB'))

def transform_bbox_normal(data):
    clazz = 0
    if(data[4] > 3):
       clazz = 1
    ret = [0, 0]
    ret[clazz] = 1
    return ret
    # ret = [0, 0, 0, 0, 0]
    # ret[data[4]-1] = 1
    # return ret


images, annotations = read_lidc_dataset(normalize, transform_bbox_normal)

(x_train), (y_train) = (
    np.asarray(images[: int(len(images) * TRAIN_SIZE)]),
    np.asarray(annotations[: int(len(annotations) * TRAIN_SIZE)]),
)
(x_test), (y_test) = (
    np.asarray(images[int(len(images) * TRAIN_SIZE):]),
    np.asarray(annotations[int(len(annotations) * TRAIN_SIZE):]),
)

lidc_model = Sequential()
base_model = resnet50.ResNet50(include_top=False,
                   input_shape=(224, 224, 3),
                   pooling='avg', classes=2,
                   weights='imagenet')
base_model.trainable = False
lidc_model.add(base_model)
lidc_model.add(Flatten())
lidc_model.add(Dropout(DROP_OUT_1))
lidc_model.add(Dense(224, activation='relu'))
lidc_model.add(Dense(2, activation='softmax'))
lidc_model.trainable = True

file_name = "resnet50_model6_5classes.h5"

if TRAIN:
    history = run_experiment_cce(
        lidc_model, learning_rate, weight_decay, batch_size, num_epochs, x_train, y_train
    )
    lidc_model.save(file_name)
    print_history(history)

lidc_model.load_weights(file_name, by_name=False, skip_mismatch=False, options=None)

print_results_classification(lidc_model, x_test, y_test)

end_time = time.perf_counter()
print(end_time - start_time, "seconds")
