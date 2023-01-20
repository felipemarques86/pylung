import pickle

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from common.ds_reader import get_ds_single_file_name

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
        images_1 = load_ds(512, 0, 1, 'img-consensus-pt-' + str(i))
        annotations_1 = load_ds(512, 0, 1, 'ann4-consensus-pt-' + str(i))
        images, annotations = populate_data(normalization_func, box_reader, images, annotations, images_1,
                                            annotations_1)

    return images, annotations


# normalize an LIDC-IDRI image to grayscale
def normalize(img):
    img[img < MIN_BOUND] = -1000
    img[img > MAX_BOUND] = 600
    img = (img + 1000) / (600 + 1000)
    return np.array(255 * img, dtype="uint8")


def transform_bbox_normal(bb):
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



i, mean_iou = 0, 0
N = 10
for i in range(0, 1):
    input_image = images[i]
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 15))
    im = input_image

    # Display the image
    ax1.imshow(im, cmap=plt.cm.gray)

    input_image = np.expand_dims(input_image, axis=0)

    circle = transform_bbox_to_circle(annotations[i])
    square = transform_bbox_to_square(annotations[i])
    bbox = transform_bbox_normal(annotations[i])

    print("xxxxxxxxxx Circle xxxxxxxxx")
    print(circle)
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxx")

    print("xxxxxxxxxx Square xxxxxxxxx")
    print(square)
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxx")

    print("xxxxxxxxxx BBox xxxxxxxxx")
    print(bbox)
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxx")

    circle = transform_circle_to_bbox(circle)
    square = transform_square_to_bbox(square)

    print("xxxxxxxxxx Circle as BBox xxxxxxxxx")
    print(circle)
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxx")

    print("xxxxxxxxxx Square as BBox xxxxxxxxx")
    print(square)
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxx")

    ax2.imshow(im[int(bbox[0]):int(bbox[1]), int(bbox[2]):int(bbox[3])], cmap=plt.cm.gray)
    ax3.imshow(im[int(square[0]):int(square[1]), int(square[2]):int(square[3])], cmap=plt.cm.gray)
    ax4.imshow(im[int(circle[0]):int(circle[1]), int(circle[2]):int(circle[3])], cmap=plt.cm.gray)

    # Create the bounding box
    rect = patches.Rectangle(
        (bbox[2], bbox[0]),
        bbox[3] - bbox[2],
        bbox[1] - bbox[0],
        facecolor="none",
        edgecolor="blue",
        linewidth=1,
    )
    print(rect)
    # Add the bounding box to the image
    ax1.add_patch(rect)

    rect = patches.Rectangle(
        (square[2], square[0]),
        square[3] - square[2],
        square[1] - square[0],
        facecolor="none",
        edgecolor="green",
        linewidth=1,
    )
    # Add the bounding box to the image
    ax1.add_patch(rect)

    rect = patches.Rectangle(
        (circle[2], circle[0]),
        circle[3] - circle[2],
        circle[1] - circle[0],
        facecolor="none",
        edgecolor="red",
        linewidth=1,
    )
    # Add the bounding box to the image
    ax1.add_patch(rect)

plt.show()
