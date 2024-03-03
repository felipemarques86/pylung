import sys
from abc import abstractmethod
import os
import random
from statistics import mean, variance, stdev

import numpy as np
from keras.applications.densenet import layers
from numpy import empty
from prettytable import PrettyTable

from main.common.config_classes import ConfigurableObject
import tensorflow as tf

from main.utilities.utilities_lib import info
from collections import Counter

class Dataset(ConfigurableObject):
    def __init__(self, name, type) -> None:
        super().__init__()
        self.name = name
        self.type = type


class DatasetTransformer:
    def __init__(self, function) -> None:
        super().__init__()
        self.function = function

    def execute(self, param1, param2=None, param3=None, param4=None):
        return self.function(param1, param2, param3, param4)

    def walk(self, param):
        self.function(param)


class MutableCollectionReader(ConfigurableObject):
    def __init__(self):
        super().__init__()
        self.dataset_image_transformers = []
        self.dataset_data_transformers = []

    def add_data_normalizer(self, function):
        self.dataset_image_transformers.append(DatasetTransformer(function))

    def add_data_transformer(self, function):
        self.dataset_data_transformers.append(DatasetTransformer(function))


class DatasetReader(MutableCollectionReader):
    def __init__(self, dataset, location) -> None:
        super().__init__()
        self.filter = None
        if not os.path.exists(location):
            raise Exception("The specified location " + location + " does not exist")
        self.dataset = dataset
        self.location = location
        self.images = None
        self.annotations = None

    @abstractmethod
    def load(self):
        pass

    def shuffle_data(self):
        all = list(zip(self.images, self.annotations))
        import random
        random.shuffle(all)
        images, annotations = zip(*all)
        self.images = images
        self.annotations = annotations

    def augment(self, percentage=0.1, random_rotation=.5, in_imgs=None, in_anns=None, amnt=0, out_imgs=None, out_anns=None, in_ann_raw=[], out_ann_raw=[]):
        modify_image = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(random_rotation)
        ])

        if in_imgs is not None and in_anns is not None:
            images_to_be_augmented = range(0, amnt)
            s = 0
            for i in images_to_be_augmented:
                if len(in_imgs) == s:
                    s = 0
                new_image = modify_image(tf.cast(tf.expand_dims(in_imgs[s], 0), tf.int32))
                out_imgs.append(new_image[0])
                out_anns.append(in_anns[s])
                out_ann_raw.append(in_ann_raw[s])
                s += 1
        else:
            images_to_be_augmented = random.sample(range(0, len(self.images)), int(len(self.images)*percentage))
            new_image = None
            for i in images_to_be_augmented:
                new_image = modify_image(tf.cast(tf.expand_dims(self.images[i], 0), tf.int32))
                self.images.append(new_image[0])
                self.annotations.append(self.annotations[i])
            return new_image[0]

    def filter_out(self, function):
        self.filter = function

    def calculate_stats(self, images, annotations):
        table = PrettyTable(['Parameter', 'Value'])
        table.add_row(['Total images', len(images)])
        table.add_row(['Total annotations', len(annotations)])
        aggr = []
        _mean = []
        _variance = []
        _std = []
        _ann_max = []
        _ann_min = []
        for k in range(0, 10): # len(annotations[0])
            aggr.append([])

        _max = -sys.maxsize - 1
        _min = sys.maxsize

        for i in range(0, len(images)):
            m = np.max(images[i])
            n = np.min(images[i])
            if m > _max:
                _max = m
            if n < _min:
                _min = n
            for j, a in enumerate(annotations[i]):
                aggr[j].append(a)

        table.add_row(['Max value of images', _max])
        table.add_row(['Min value of images', _min])
        table.add_row(['Example annotation', annotations[0]])
        # _aggr_counter = Counter()

        for a in aggr:
            if len(a) > 1 and not isinstance(a[0], (str, bool)):
                _ann_max.append(max(a))
                _ann_min.append(min(a))
                try:
                    _mean.append(mean(a))
                except:
                    print('Cannot calculate mean')
                if len(a) > 1:
                    try:
                        _variance.append(variance(a))
                    except:
                        print('Cannot calculate variance')
                else:
                    _variance.append('N/A')
                if len(a) > 1:
                    try:
                        _std.append(stdev(a))
                    except:
                        print('Cannot calculate stdev')
                else:
                    _std.append('N/A')

        table.add_row(['Annotations mean', _mean])
        table.add_row(['Annotations max', _ann_max])
        table.add_row(['Annotations min', _ann_min])
        table.add_row(['Annotations standard deviation', _std])
        table.add_row(['Annotations variance', _variance])
        # table.add_row(['Annotations count', _aggr_counter])

        print(table)

    def dump_to_file(self, data, file):
        with open(file, 'w') as my_file:
            for i in data:
                np.savetxt(my_file, i)
    def statistics(self, first=1.0):

        if first < 1.0:
            (first_images), (first_annotations) = (
                self.images[: int(len(self.images) * first)],
                self.annotations[: int(len(self.annotations) * first)])
            (last_images), (last_annotations) = (
                np.asarray(self.images[int(len(self.images) * first):]),
                np.asarray(self.annotations[int(len(self.annotations) * first):]))

            info(f'Statistics for the first {first*100}% images')
            # if data_augmentation:
            #     _aggr_counter = Counter()
            #     for a in first_annotations:
            #         _aggr_counter[tuple(a)] += 1
            #
            #     max_aggr = max(_aggr_counter, key=_aggr_counter.get)
            #
            #     for i in _aggr_counter:
            #         if _aggr_counter[i] < _aggr_counter[max_aggr]:
            #             add_n_images = _aggr_counter[max_aggr] - _aggr_counter[i]
            #             first_images_of_clazz = [first_images[j] for j in range(len(first_images)) if first_annotations[j] == list(i)]
            #             first_annotations_of_clazz = [first_annotations[j] for j in range(len(first_annotations)) if first_annotations[j] == list(i)]
            #             self.augment(in_imgs=first_images_of_clazz, in_anns=first_annotations_of_clazz, amnt=add_n_images, out_imgs=first_images, out_anns=first_annotations)

            self.calculate_stats(first_images, first_annotations)

            #self.dump_to_file(first_annotations, 'first_annotations.txt')

            info(f'Statistics for the last {(1-first) * 100}% images')
            self.calculate_stats(last_images, last_annotations)
            #self.dump_to_file(last_annotations, 'last_annotations.txt')

        info(f'Statistics for all images')
        self.calculate_stats(self.images, self.annotations)


class DataCollection(MutableCollectionReader):
    def __init__(self, name, images, annotations):
        super().__init__()
        self.name = name
        self.images = images
        self.annotations = annotations

    def display_image(self, position, show_box=False, show_array=False):
        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib.patches as patches

        input_image = self.images[position]
        im = input_image
        input_image = np.expand_dims(input_image, axis=0)
        p = self.dataset_image_transformers[0].execute(input_image)

        predicted = self.dataset_data_transformers[0].execute((p[0]))
        truth = self.dataset_data_transformers[0].execute((self.annotations[position]))

        if show_box:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 15))
            # Display the image
            ax1.imshow(im, cmap=plt.cm.gray)
            ax2.imshow(im, cmap=plt.cm.gray)

            box_predicted = (predicted[0], predicted[1], predicted[2], predicted[3])
            box_truth = (truth[0], truth[1], truth[2], truth[3])

            ax3.imshow(im[int(box_truth[0]):int(box_truth[1]), int(box_truth[2]):int(box_truth[3])], cmap=plt.cm.gray)
            ax4.imshow(im[int(box_predicted[0]):int(box_predicted[1]), int(box_predicted[2]):int(box_predicted[3])],
                       cmap=plt.cm.gray)

            # Create the bounding box
            rect = patches.Rectangle(
                (int(box_predicted[2]), int(box_predicted[0])),
                int(box_predicted[3] - box_predicted[2]),
                int(box_predicted[1] - box_predicted[0]),
                facecolor="none",
                edgecolor="blue",
                linewidth=2,
            )
            # Add the bounding box to the image
            ax1.add_patch(rect)
            ax1.set_xlabel(
                "Predicted: "
                + str(int(box_predicted[2]))
                + ", "
                + str(int(box_predicted[0]))
                + ", "
                + str(int(box_predicted[3]))
                + ", "
                + str(int(box_predicted[1]))
            )

            # Create the bounding box
            # y0 y1 x0 x1
            # 0  1  2  3
            rect = patches.Rectangle(
                (int(box_truth[2]), int(box_truth[0])),
                int(box_truth[3] - box_truth[2]),
                int(box_truth[1] - box_truth[0]),
                facecolor="none",
                edgecolor="green",
                linewidth=2,
            )
            # Add the bounding box to the image
            ax2.add_patch(rect)
            iou = self.__bounding_box_intersection_over_union(box_predicted, box_truth)
            ax2.set_xlabel(
                "Target: "
                + str(int(box_truth[2]))
                + ", "
                + str(int(box_truth[0]))
                + ", "
                + str(int(box_truth[3]))
                + ", "
                + str(int(box_truth[1]))
                + "\n"
                + "IoU"
                + str(iou)
            )
            return iou

        if show_array:
            fig, (ax1) = plt.subplots(1, 1, figsize=(15, 15))
            im = input_image

            # Display the image
            ax1.imshow(im, cmap=plt.cm.gray)

            ax1.set_xlabel(
                "Predicted: " + str(predicted) + ", Original: " + str(truth)
            )

    def __bounding_box_intersection_over_union(self, box_predicted, box_truth):
        # get (x, y) coordinates of intersection of bounding boxes
        top_x_intersect = max(box_predicted[0], box_truth[0])
        top_y_intersect = max(box_predicted[1], box_truth[1])
        bottom_x_intersect = min(box_predicted[2], box_truth[2])
        bottom_y_intersect = min(box_predicted[3], box_truth[3])

        # calculate area of the intersection bb (bounding box)
        intersection_area = max(0, bottom_x_intersect - top_x_intersect + 1) * max(
            0, bottom_y_intersect - top_y_intersect + 1
        )

        # calculate area of the prediction bb and ground-truth bb
        box_predicted_area = (box_predicted[2] - box_predicted[0] + 1) * (
                box_predicted[3] - box_predicted[1] + 1
        )
        box_truth_area = (box_truth[2] - box_truth[0] + 1) * (
                box_truth[3] - box_truth[1] + 1
        )

        # calculate intersection over union by taking intersection
        # area and dividing it by the sum of predicted bb and ground truth
        # bb areas subtracted by  the interesection area

        # return ioU
        return intersection_area / float(
            box_predicted_area + box_truth_area - intersection_area
        )
