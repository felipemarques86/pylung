import pickle
import time

import cv2
import numpy as np
from keras import Sequential
from keras.layers import Flatten, Dense

from tensorflow.keras.applications import resnet50
from common.ds_reader import get_ds_single_file_name
from common.experiment import print_history, run_experiment_adamw, print_results_resized
from vit.keras_custom.keras_vit import create_vit_object_detector

TRAIN_SIZE = 0.8
TEST_SIZE = 1 - TRAIN_SIZE
IMAGE_SIZE = 224
SCAN_COUNT_PERC = 1


PAD = 0
patch_size = 32 # Size of the patches to be extracted from the input images
input_shape = (IMAGE_SIZE, IMAGE_SIZE, 1)  # input image shape
learning_rate = 0.000001
weight_decay = 0.001
batch_size = 30
num_epochs = 50
num_patches = (IMAGE_SIZE // patch_size) ** 2
projection_dim = 64
num_heads = 10
TRAIN = True
# Size of the transformer layers
transformer_units = [
    projection_dim * 2,
    projection_dim,
]
transformer_layers = 4
mlp_head_units = [2048, 1024, 512, 64, 32]  # Size of the dense layers
history = []
num_patches = (IMAGE_SIZE // patch_size) ** 2

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

    for i in range(1, 4):
        images_1 = load_ds(-1, 0, 1, 'img-consensus-pt-' + str(i))
        annotations_1 = load_ds(-1, 0, 1, 'ann6-consensus-pt-' + str(i))
        images, annotations = populate_data(normalization_func, box_reader, images, annotations, images_1, annotations_1)

    return images, annotations


# normalize an LIDC-IDRI image to grayscale
def normalize(im):
    im = np.float32(im)
    image = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    return np.int8(image)


def transform_bbox_normal(bb):
    print(bb)
    y0 = bb[0]
    y1 = bb[1]
    x0 = bb[2]
    x1 = bb[3]
    return y0, y1, x0, x1


def transform_bbox_to_square(bb):
    y0 = bb[0]
    y1 = bb[1]
    x0 = bb[2]
    x1 = bb[3]
    return x0, y0, max(x1 - x0, y1 - y0)


def transform_square_to_bbox(sq):
    x0 = sq[0]
    y0 = sq[1]
    s = sq[2]
    return y0, y0 + s, x0, x0 + s


def transform_bbox_to_circle(bb):
    y0 = bb[0]
    y1 = bb[1]
    x0 = bb[2]
    x1 = bb[3]
    return int(x0 + (x1 - x0) / 2), int(y0 + (y1 - y0) / 2), max(x1 - x0, y1 - y0) / 2


def transform_circle_to_bbox(circ):
    x0 = circ[0]
    y0 = circ[1]
    r = circ[2]
    y1 = y0 + r * 2
    x1 = x0 + r * 2
    return y0 - r, y1, x0 - r, x1

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
                   pooling='avg', classes=4,
                   weights='imagenet')
base_model.trainable = False
lidc_model.add(base_model)
lidc_model.add(Flatten())
lidc_model.add(Dense(224, activation='relu'))
lidc_model.add(Dense(4, activation='softmax'))
lidc_model.trainable = True

file_name = "resnet50_model.h5"

if TRAIN:
    history = run_experiment_adamw(
        lidc_model, learning_rate, weight_decay, batch_size, num_epochs, x_train, y_train
    )
    lidc_model.save(file_name)
    print_history(history)

lidc_model.load_weights(file_name, by_name=False, skip_mismatch=False, options=None)


lidc_model.load_weights(file_name, by_name=False, skip_mismatch=False, options=None) # ('vit_object_detector.h5')

print_results_resized(IMAGE_SIZE, lidc_model, x_test, y_test, transform_bbox_normal)

end_time = time.perf_counter()
print(end_time - start_time, "seconds")
