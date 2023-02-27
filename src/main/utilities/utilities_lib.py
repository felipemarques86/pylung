import cv2
import numpy as np
# from tensorflow.keras import metrics
import tensorflow as tf
import tensorflow_addons as tfa
from colorama import Fore, Style
from matplotlib import pyplot as plt, patches

LIDC_ANN_Y0 = 0
LIDC_ANN_Y1 = 1
LIDC_ANN_X0 = 2
LIDC_ANN_X1 = 3
LIDC_ANN_ML = 4
LIDC_ANN_SP = 5
NORMALIZE = True
ALPHA_NAMES = ['Alpha','Bravo','Charlie','Delta','Echo','Foxtrot','Golf','Hotel','India','Juliett','Kilo','Lima','Mike',
               'November','Oscar','Papa','Quebec','Romeo','Sierra','Tango','Uniform','Victor','Whiskey','X-ray',
               'Yankee','Zulu']
ANIMALS_NAMES = ['Caracal', 'Cheetah', 'Eland', 'Giraffe', 'Zebra', 'Hartebeest', 'Impala', 'Jackal', 'Leopard',
                 'Lion', 'Ostrich', 'Rhinoceros', 'Hyena', 'Warthog', 'Wildebeest', 'Flamingo', 'Gecko', 'Tortoise',
                 'Crocodile', 'Mongoose', 'Gazelle', 'Stork', 'Meerkat', 'Elephant']

def warning(message: str):
    print(f"{Fore.YELLOW}{Style.BRIGHT}[WARNING] {Style.RESET_ALL}{Fore.YELLOW}{message}{Style.RESET_ALL}")


def error(message: str):
    print(f"{Fore.RED}{Style.BRIGHT}[ERROR] {Style.RESET_ALL}{Fore.RED}{message}{Style.RESET_ALL}")


def info(message: str):
    print(f"{Fore.BLUE}{Style.BRIGHT}[INFO] {Style.RESET_ALL}{Fore.BLUE}{message}{Style.RESET_ALL}")



def get_data_transformer(data_transformer_name, detection=False):
    metrics = []
    data_transformer = globals()[data_transformer_name]
    loss = 'categorical_crossentropy'
    if data_transformer_name.startswith('binary_'):
        num_classes = 1
        loss = 'binary_crossentropy'
        metrics = [tf.keras.metrics.BinaryAccuracy()]
    elif data_transformer_name.startswith('one_hot_two'):
        num_classes = 2
        metrics.append("accuracy")
    elif data_transformer_name.startswith('one_hot_three'):
        num_classes = 3
        metrics.append("accuracy")
    elif data_transformer_name.startswith('one_hot_four'):
        num_classes = 4
        metrics.append("accuracy")
    elif data_transformer_name.startswith('one_hot_five'):
        num_classes = 5
        metrics.append("accuracy")
    elif data_transformer_name.startswith('one_hot_six'):
        num_classes = 6
        metrics.append("accuracy")
    elif data_transformer_name.startswith('bbox'):
        num_classes = 4
        metrics.append("accuracy")
    else:
        raise Exception(f'{data_transformer_name} not found!')

    if not detection:
        metrics.append(tf.keras.metrics.FalsePositives())
        metrics.append(tf.keras.metrics.FalseNegatives())
        metrics.append(tf.keras.metrics.TrueNegatives())
        metrics.append(tf.keras.metrics.TruePositives())


    return num_classes, data_transformer, loss, metrics


def filter_out_class3_malignancy(data):
    return data[LIDC_ANN_ML] == 3

def filter_out_class0_malignancy(data):
    return data[LIDC_ANN_ML] == 0


# Problem reduction functions - these functions determine which classes are going to be used and how they are grouped

# Any image with ML != 0 is considered an image with nodule
def binary_non_module(data, _1, _2):
    if data[LIDC_ANN_ML] == 0:
        return [0]
    return [1]

# Images with ML = [0,1,2] are BENIGN=[1, 0] and [3,4,5] are MALIGNANT=[0, 1]
def one_hot_two_non_nodule(data, _1, _2):
    if data[LIDC_ANN_ML] > 0:
        return [0, 1]  # malignant
    return [1, 0]

# Images with ML = [0,1,2] are BENIGN=[0] and [3,4,5] are MALIGNANT=[1]
def binary_malignancy_3benign(data, _1, _2):
    clazz = 0
    if data[LIDC_ANN_ML] > 3:
        clazz = 1
    return [clazz]

def one_hot_two_malignancy_3benign(data, _1, _2):
    if data[LIDC_ANN_ML] > 3:
        return [0, 1]  # malignant
    return [1, 0]

def binary_malignancy_cut0_3benign(data, _1, _2):
    if data[LIDC_ANN_ML] == 0:
        return None
    clazz = 0
    if data[LIDC_ANN_ML] > 3:
        clazz = 1
    return [clazz]

def binary_malignancy_3malignant(data, _1, _2):
    if data[LIDC_ANN_ML] > 2:
        return [1]  # malignant
    return [0]


def one_hot_two_malignancy_3malignant(data, _1, _2):
    if data[LIDC_ANN_ML] > 2:
        return [0, 1]  # malignant
    return [1, 0]

def binary_malignancy_cut3(data, _1, _2):
    if data[LIDC_ANN_ML] == 3:
        return None
    if data[LIDC_ANN_ML] > 2:
        return [1]
    return [0]

def one_hot_two_malignancy_cut3(data, _1, _2):
    if data[LIDC_ANN_ML] == 3:
        return None
    if data[LIDC_ANN_ML] > 2:
        return [0, 1]
    return [1, 0]

def binary_malignancy_cut0and3(data, _1, _2):
    if data[LIDC_ANN_ML] == 3 or data[LIDC_ANN_ML] == 0:
        return None
    if data[LIDC_ANN_ML] > 2:
        return [1]  # malignant
    return [0]

def one_hot_two_malignancy_cut0and3(data, _1, _2):
    if data[LIDC_ANN_ML] == 3 or data[LIDC_ANN_ML] == 0:
        return None
    if data[LIDC_ANN_ML] > 2:
        return [0, 1] # malignant
    return [1, 0]


def binary_malignancy_cut0_3benign(data, _1, _2):
    if data[LIDC_ANN_ML] == 0:
        return None
    if data[LIDC_ANN_ML] > 3:
        return [1]  # malignant
    return [0]


def one_hot_two_malignancy_cut0_3benign(data, _1, _2):
    if data[LIDC_ANN_ML] == 0:
        return None
    if data[LIDC_ANN_ML] > 3:
        return [0, 1]  # malignant
    return [1, 0]


def binary_malignancy_cut0_3malignant(data, _1, _2):
    if data[LIDC_ANN_ML] == 0:
        return None
    if data[LIDC_ANN_ML] > 2:
        return [1]  # malignant
    return [0]

def one_hot_two_malignancy_cut0_3malignant(data, _1, _2):
    if data[LIDC_ANN_ML] == 0:
        return None
    if data[LIDC_ANN_ML] > 2:
        return [0, 1]  # malignant
    return [1, 0]

def one_hot_six(data, _1, _2):
    ret = [0, 0, 0, 0, 0, 0]
    ret[int(data[LIDC_ANN_ML]-1)] = 1
    return ret

def one_hot_five(data, _1, _2):
    ret = [0, 0, 0, 0, 0]
    if data[LIDC_ANN_ML] == 0:
        return ret
    if data[LIDC_ANN_ML] == 1 or data[LIDC_ANN_ML] == 2:
        return [0, 1, 0, 0, 0]
    ret[int(data[LIDC_ANN_ML]-1)] = 1
    return ret

def one_hot_five_cut0(data, _1, _2):
    if data[LIDC_ANN_ML] == 0:
        return None
    ret = [0, 0, 0, 0, 0]
    ret[int(data[LIDC_ANN_ML]-1)] = 1
    return ret

def one_hot_five_cut3(data, _1, _2):
    if data[LIDC_ANN_ML] == 3:
        return None
    if data[LIDC_ANN_ML] == 0:
        return [0, 0, 0, 0, 0]
    if data[LIDC_ANN_ML] == 1:
        return [0, 1, 0, 0, 0]
    elif data[LIDC_ANN_ML] == 2:
        return [0, 0, 1, 0, 0]
    elif data[LIDC_ANN_ML] == 4:
        return [0, 0, 0, 1, 0]
    else:
        return [0, 0, 0, 0, 1]

def one_hot_four(data, _1, _2):
    ret = [0, 0, 0, 0]
    if data[LIDC_ANN_ML] == 0:
        return ret
    if data[LIDC_ANN_ML] == 1 or data[LIDC_ANN_ML] == 2:
        return [0, 1, 0, 0]
    elif data[LIDC_ANN_ML] == 3 or data[LIDC_ANN_ML] == 4:
        return [0, 0, 1, 0]
    else:
        return [0, 0, 0, 1]

def one_hot_four_cut0and3(data, _1, _2):
    if data[LIDC_ANN_ML] == 3 or data[LIDC_ANN_ML] == 0:
        return None
    if data[LIDC_ANN_ML] == 1:
        return [1, 0, 0, 0]
    if data[LIDC_ANN_ML] == 2:
        return [0, 1, 0, 0]
    elif data[LIDC_ANN_ML] == 4:
        return [0, 0, 1, 0]
    else:
        return [0, 0, 0, 1]

def bbox(data, _, image_size):
    if data[LIDC_ANN_ML] == 0:
        return None
    return (data[LIDC_ANN_Y0] / 512) * image_size, (data[LIDC_ANN_Y1] / 512) * image_size, \
        (data[LIDC_ANN_X0] / 512) * image_size, (data[LIDC_ANN_X1] / 512) * image_size


def get_optimizer(name, learning_rate, weight_decay, momentum):
    if name == 'AdamW':
        return tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    elif name == 'SGDW':
        return tfa.optimizers.SGDW(learning_rate=learning_rate, weight_decay=weight_decay)
    elif name == 'SGD':
        return tf.keras.optimizers.SGD(
            learning_rate=learning_rate, momentum=momentum, nesterov=False, name='SGD')


def img_transformer(image_size, channels, isolate_nodule_image):
    def resize(image, annotation, _):
        image = np.float32(image)
        if NORMALIZE:
            image[image < -1000] = -1000
            image[image > 600] = 600
            image = (image + 1000) / (600 + 1000)
            image = image * 255

        if isolate_nodule_image and (annotation[LIDC_ANN_Y0] != 0 or annotation[LIDC_ANN_Y1] != 0 or annotation[LIDC_ANN_X0] != 0 or annotation[LIDC_ANN_X1] != 0):
            im = cv2.resize(image[annotation[LIDC_ANN_Y0]:annotation[LIDC_ANN_Y1], annotation[LIDC_ANN_X0]:annotation[LIDC_ANN_X1]], (image_size, image_size))
        elif isolate_nodule_image and annotation[LIDC_ANN_Y0] == 0 and annotation[LIDC_ANN_Y1] == 0 and annotation[LIDC_ANN_X0] == 0 and annotation[LIDC_ANN_X1] == 0:
            error('No image to be cropped. Please only "cut0" type of problem reduction function!')
            exit(-1)
        else:
            im = cv2.resize(image, (image_size, image_size))
        if channels == 3:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        return np.int16(im)
    return resize

def get_channels(model_type):
    if model_type == 'vit' or model_type == 'cait':
        return 1
    return 3

def shuffle_data(images, annotations):
    images_annotations = list(zip(images, annotations))
    import random
    random.shuffle(images_annotations)
    res1, res2 = zip(*images_annotations)
    images, annotations = list(res1), list(res2)
    return images, annotations


def display_image(image, annotations, N=1):
    for k in range(0, N):
        fig, ax1 = plt.subplots(1, 1, figsize=(15, 15))
        ax1.imshow(image[k], cmap=plt.cm.gray)
        data = '-'
        for i in range(0, len(annotations[k])):
            data = data + str(annotations[k][i]) + '-'

        ax1.set_xlabel("Data: " + data)
        plt.show()


def display_original_image_bbox(image, annotations, extra_text=''):
    if len(annotations) < 4 or annotations[0] == 0 and annotations[1] == 0:
        fig, ax1 = plt.subplots(1, 1, figsize=(15, 15))
        # Display the image
        ax1.imshow(image, cmap=plt.cm.gray)
        ax1.set_xlabel(
            "Original Data: "
            + str(annotations)
            + "\n"
            + extra_text
        )
        plt.show()
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
        # Display the image
        ax1.imshow(image, cmap=plt.cm.gray)
        ax2.imshow(image[int(annotations[0]):int(annotations[1]), int(annotations[2]):int(annotations[3])],
                   cmap=plt.cm.gray)
        rect = patches.Rectangle(
            (int(annotations[2]), int(annotations[0])),
            int(annotations[3] - annotations[2]),
            int(annotations[1] - annotations[0]),
            facecolor="none",
            edgecolor="red",
            linewidth=2,
        )
        # Add the bounding box to the image
        ax1.add_patch(rect)
        ax1.set_xlabel(
            "Original Data: "
            + str(annotations)
            + "\n"
            + extra_text
        )
        plt.show()


def get_experiment_codename(pos: int):
    pos2 = int(pos/len(ALPHA_NAMES))
    return ALPHA_NAMES[pos % len(ALPHA_NAMES)] + ' ' + ANIMALS_NAMES[pos2]
