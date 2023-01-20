import pickle
import time

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import activations

from common.ds_reader import get_ds_single_file_name
from common.experiment import print_history, run_experiment_adamw, print_results_classification, run_experiment_cce, \
    shuffle_data
from vit.keras_custom.keras_vit import create_vit_object_detector

DROP_OUT_1 = .1
DROP_OUT_2 = .3
DROP_OUT_3 = .3
TRAIN_SIZE = 0.8
TEST_SIZE = 1 - TRAIN_SIZE
IMAGE_SIZE = 224
SCAN_COUNT_PERC = 1
PAD = 0
patch_size = 16 # Size of the patches to be extracted from the input images
input_shape = (IMAGE_SIZE, IMAGE_SIZE, 1)  # input image shape
learning_rate = 0.00001
weight_decay =  0.0001
batch_size = 100
num_epochs = 1000
num_patches = (IMAGE_SIZE // patch_size) ** 2
projection_dim = 64
num_heads = 8
TRAIN = False
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
        images_1 = load_ds(IMAGE_SIZE, 0, 1, 'img-consensus-pt-' + str(i))
        annotations_1 = load_ds(IMAGE_SIZE, 0, 1, 'ann6-consensus-pt-' + str(i))
        images, annotations = populate_data(normalization_func, box_reader, images, annotations, images_1, annotations_1)

    images, annotations = shuffle_data(images, annotations)

    return images, annotations


# normalize an LIDC-IDRI image to grayscale
def normalize(im):
    return im


def transform_to_one_hot(data):
    clazz = 0
    if(data[5] > 3):
        clazz = 1
    ret = [0, 0]
    ret[clazz] = 1
    return ret

images, annotations = read_lidc_dataset(normalize, transform_to_one_hot)

(x_train), (y_train) = (
    np.asarray(images[: int(len(images) * TRAIN_SIZE)]),
    np.asarray(annotations[: int(len(annotations) * TRAIN_SIZE)]),
)
(x_test), (y_test) = (
    np.asarray(images[int(len(images) * TRAIN_SIZE):]),
    np.asarray(annotations[int(len(annotations) * TRAIN_SIZE):]),
)

vit_object_detector = create_vit_object_detector(
    input_shape,
    patch_size,
    num_patches,
    projection_dim,
    num_heads,
    transformer_units,
    transformer_layers,
    mlp_head_units,
    2,
    'softmax',
    DROP_OUT_1,
    DROP_OUT_2,
    DROP_OUT_3,
)


#, threshold=0.0, max_value=1

file_name = "vit_model_" + str(IMAGE_SIZE) + "_" + str(learning_rate) + "-" + str(weight_decay) + "-" + str(num_heads) + "-" + str(projection_dim) + \
            "-" + str(transformer_layers) + "-" + str(num_epochs) + ".h5"

if TRAIN:
    history = run_experiment_cce(
        vit_object_detector, learning_rate, weight_decay, batch_size, num_epochs, x_train, y_train
    )
    vit_object_detector.save(file_name)
    print_history(history)

vit_object_detector.load_weights(file_name, by_name=False, skip_mismatch=False, options=None)

print_results_classification(vit_object_detector, x_test, y_test)

end_time = time.perf_counter()
print(end_time - start_time, "seconds")
