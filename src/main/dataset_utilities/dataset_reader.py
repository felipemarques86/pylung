from abc import abstractmethod


class Dataset:
    def __init__(self, name, type) -> None:
        self.name = name
        self.type = type


class DatasetTransformer:
    def __init__(self, function) -> None:
        super().__init__()
        self.function = function

    def execute(self, param):
        return self.function(param)


class MutableCollectionReader:
    def __init__(self) -> None:
        self.dataset_image_transformers = []
        self.dataset_data_transformers = []

    def add_data_normalizer(self, function):
        self.dataset_image_transformers.append(DatasetTransformer(function))

    def add_data_transformer(self, function):
        self.dataset_data_transformers.append(DatasetTransformer(function))


class DatasetReader(MutableCollectionReader):
    def __init__(self, dataset, location, part=0, max_parts=1) -> None:
        self.dataset = dataset
        self.location = location
        self.images = None
        self.annotations = None
        self.part = part
        self.max_parts = max_parts

    @abstractmethod
    def load(self, dry_run=False):
        pass

    def shuffle_data(self):
        all = list(zip(self.images, self.annotations))
        import random
        random.shuffle(all)
        res1, res2 = zip(*all)
        images, annotations = list(res1), list(res2)
        self.images = images
        self.annotations = annotations


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
