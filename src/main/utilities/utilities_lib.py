import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from colorama import Fore, Style

LIDC_ANN_Y0 = 0
LIDC_ANN_Y1 = 1
LIDC_ANN_X0 = 2
LIDC_ANN_X1 = 3
LIDC_ANN_ML = 4
LIDC_ANN_SP = 5


def warning(message: str):
    print(f"{Fore.YELLOW}{Style.BRIGHT}[WARNING] {Style.RESET_ALL}{Fore.YELLOW}{message}{Style.RESET_ALL}")


def error(message: str):
    print(f"{Fore.RED}{Style.BRIGHT}[ERROR] {Style.RESET_ALL}{Fore.RED}{message}{Style.RESET_ALL}")


def info(message: str):
    print(f"{Fore.BLUE}{Style.BRIGHT}[INFO] {Style.RESET_ALL}{Fore.BLUE}{message}{Style.RESET_ALL}")



def get_data_transformer(data_transformer_name):
    num_classes = None
    data_transformer = None
    loss = None
    if data_transformer_name == 'binary_malignancy':
        data_transformer = binary_malignancy
        num_classes = 1
        loss = 'binary_crossentropy'
    if data_transformer_name == 'binary_non_module':
        data_transformer = binary_non_module
        num_classes = 1
        loss = 'binary_crossentropy'
    elif data_transformer_name == 'six_one_hot_malignancy':
        data_transformer = six_one_hot_malignancy
        num_classes = 6
        loss = 'categorical_crossentropy'
    elif data_transformer_name == 'five_one_hot_malignancy':
        data_transformer = five_one_hot_malignancy
        num_classes = 5
        loss = 'categorical_crossentropy'
    elif data_transformer_name == 'two_one_hot_malignancy':
        data_transformer = two_one_hot_malignancy
        num_classes = 2
        loss = 'categorical_crossentropy'
    elif data_transformer_name == 'two_one_hot_non_module':
        data_transformer = two_one_hot_non_module
        num_classes = 2
        loss = 'categorical_crossentropy'
    elif data_transformer_name == 'bbox':
        data_transformer = bbox
        num_classes = 4
        loss = 'categorical_crossentropy'
    return num_classes, data_transformer, loss


def filter_out_class3_malignancy(data):
    return data[LIDC_ANN_ML] == 3

def filter_out_class0_malignancy(data):
    return data[LIDC_ANN_ML] == 0


def binary_non_module(data, _):
    if data[LIDC_ANN_ML] == 0:
        return [0]
    return [1]

def binary_malignancy(data, _):
    clazz = 0
    if data[LIDC_ANN_ML] > 3:
        clazz = 1
    ret = [0, 0]
    ret[clazz] = 1
    return [clazz]

def six_one_hot_malignancy(data, _):
    ret = [0, 0, 0, 0, 0, 0]
    ret[int(data[LIDC_ANN_ML]-1)] = 1
    return ret


def five_one_hot_malignancy(data, _):
    ret = [0, 0, 0, 0, 0]
    ret[int(data[LIDC_ANN_ML]-1)] = 1
    return ret

def two_one_hot_malignancy(data, _):
    ret = [0, 0]
    clazz = 0
    if data[LIDC_ANN_ML] > 3:
        clazz = 1
    ret[clazz] = 1
    return ret

def two_one_hot_non_module(data, _):
    ret = [0, 0]
    clazz = 0
    if data[LIDC_ANN_ML] > 0:
        clazz = 1
    ret[clazz] = 1
    return ret

def bbox(data, _):
    return data[LIDC_ANN_Y0], data[LIDC_ANN_Y1], data[LIDC_ANN_X0], data[LIDC_ANN_X1],


# def resize_color(image, annotation):
#     image = np.float32(image)
#     im = cv2.resize(image[annotation[LIDC_ANN_Y0]:annotation[LIDC_ANN_Y1], annotation[LIDC_ANN_X0]:annotation[LIDC_ANN_X1]], (IMAGE_SIZE, IMAGE_SIZE))
#     im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
#     return np.int16(im)
#
# def resize(image, annotation):
#     image = np.float32(image)
#     im = cv2.resize(image[annotation[LIDC_ANN_Y0]:annotation[LIDC_ANN_Y1], annotation[LIDC_ANN_X0]:annotation[LIDC_ANN_X1]], (IMAGE_SIZE, IMAGE_SIZE))
#     return np.int16(im)

def get_optimizer(name, learning_rate, weight_decay, momentum):
    if name == 'AdamW':
        return tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    elif name == 'SGDW':
        return tfa.optimizers.SGDW(learning_rate=learning_rate, weight_decay=weight_decay)
    elif name == 'SGD':
        return tf.keras.optimizers.SGD(
            learning_rate=learning_rate, momentum=momentum, nesterov=False, name='SGD')


def img_transformer(image_size, channels, crop_image):
    def resize(image, annotation):
        image = np.float32(image)
        if crop_image and (annotation[LIDC_ANN_Y0] != 0 or annotation[LIDC_ANN_Y1] != 0 or annotation[LIDC_ANN_X0] != 0 or annotation[LIDC_ANN_X1] != 0):
            im = cv2.resize(image[annotation[LIDC_ANN_Y0]:annotation[LIDC_ANN_Y1], annotation[LIDC_ANN_X0]:annotation[LIDC_ANN_X1]], (image_size, image_size))
        elif crop_image and annotation[LIDC_ANN_Y0] == 0 and annotation[LIDC_ANN_Y1] == 0 and annotation[LIDC_ANN_X0] == 0 and annotation[LIDC_ANN_X1] != 0:
            error('No image to be cropped')
            exit(-1)
        else:
            im = cv2.resize(image, (image_size, image_size))
        if channels == 3:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        return np.int16(im)
    return resize
