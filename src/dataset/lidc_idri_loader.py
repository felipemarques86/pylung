import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pylidc as pl
from PIL import Image
from pylidc.utils import consensus
from tensorflow import keras
import matplotlib.pyplot as plt


LIDC_IMAGE_ = 'lidc_image_'
SCALED_BOX_ = 'lidc_scaled_box_'
PRE_PROCESSED_FOLDER_PREFIX = 'LIDC-IDRI-size-'
DATA_FILE_EXT = '.pkl'
MIN_BOUND = -1000.0
MAX_BOUND = 600.0
PIXEL_MEAN = 0.25


def load_pre_processed_lidc_idri(image_size, base_path, pad=0):
    if image_size != 128 and image_size != 256 and image_size != 512:
        raise Exception('image_size parameter must have the value: 128, 256 or 512. ' + str(image_size) +
                        ' is not a valid value')
    if base_path is None:
        raise Exception('base_path is mandatory')

    path = base_path + PRE_PROCESSED_FOLDER_PREFIX + str(image_size) + '-' + str(pad) + os.sep
    path_high_res = base_path + PRE_PROCESSED_FOLDER_PREFIX + '512' + '-' + str(pad) + os.sep

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
        image_filename = path + LIDC_IMAGE_ + str(i) + DATA_FILE_EXT
        with open(image_filename, 'rb') as filePointer:  # Overwrites any existing file.
            if len(x_train) < train_to:
                raw_img = pickle.load(filePointer)
                x_train.append(raw_img)
            else:
                image_filename = path_high_res + LIDC_IMAGE_ + str(i) + DATA_FILE_EXT
                with open(image_filename, 'rb') as filePointer2:  # Overwrites any existing file.
                    x_test.append(pickle.load(filePointer2))

        bbox_filename = path + SCALED_BOX_ + str(i) + DATA_FILE_EXT
        with open(bbox_filename, 'rb') as filePointer:  # Overwrites any existing file.
            if (len(y_train) < train_to):
                y_train.append(pickle.load(filePointer))
            else:
                bbox_filename = path_high_res + SCALED_BOX_ + str(i) + DATA_FILE_EXT
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


def build_pre_processed_dataset(image_size, base_path, annotation_size_perc=1, image_count=-1, pad=0):
    if image_size != 64 and image_size != 128 and image_size != 256 and image_size != 512:
        raise Exception('image_size parameter must have the value: 64, 128, 256 or 512. ' + str(image_size) +
                        ' is not a valid value')
    if base_path is None:
        raise Exception('base_path is mandatory')

    Path(base_path + PRE_PROCESSED_FOLDER_PREFIX + str(image_size) + '-' + str(pad)).mkdir(parents=True, exist_ok=True)
    total_images = 0
    annotation_list = pl.query(pl.Annotation)
    annotations_count = int(annotation_list.count() * annotation_size_perc)
    for i in range(0, annotations_count):
        annotation = annotation_list[i]
        annotation_bbox = annotation.bbox(pad)
        vol = annotation.scan.to_volume(verbose=False)

        y0, y1 = annotation_bbox[0].start, annotation_bbox[0].stop
        x0, x1 = annotation_bbox[1].start, annotation_bbox[1].stop

        # Get the central slice of the computed bounding box.
        i, j, k = annotation.centroid
        z = max(int(annotation_bbox[2].stop - k) - 1, 0)
        (w, h) = vol[:, :, int(k)].shape

        image = Image.fromarray(vol[:, :, int(z)])
        image = image.resize((image_size, image_size))

        scaled_bbox = (float(x0) / w, float(y0) / h, float(x1) / w, float(y1) / h)

        __save_object(scaled_bbox,
                      base_path + PRE_PROCESSED_FOLDER_PREFIX + str(image_size) + '-' + str(pad) + os.sep + SCALED_BOX_
                      + str(total_images) + DATA_FILE_EXT)
        __save_object(keras.utils.img_to_array(image),
                      base_path + PRE_PROCESSED_FOLDER_PREFIX + str(image_size) + '-' + str(pad) + os.sep + LIDC_IMAGE_
                      + str(total_images) + DATA_FILE_EXT)

        total_images = total_images + 1
        if image_count > 0 and total_images >= image_count:
            break



def load_lidc_idri_annotations_only(pad=0):
    total_images = 0
    annotation_list = pl.query(pl.Annotation)
    annotations_count = annotation_list.count()
    annotations = []
    w = 512
    h = 512
    for i in range(0, annotations_count):
        annotation = annotation_list[i]
        annotation_bbox = annotation.bbox(pad)

        y0, y1 = annotation_bbox[0].start, annotation_bbox[0].stop
        x0, x1 = annotation_bbox[1].start, annotation_bbox[1].stop

        scaled_bbox = (float(x0) / w, float(y0) / h, float(x1) / w, float(y1) / h)

        # apply relative scaling to bounding boxes as per given image and append to list
        annotations.append(scaled_bbox)
        total_images = total_images + 1
        print(total_images)

    # Convert the list to numpy array, split to train and test dataset
    (ytrain) = (
        np.asarray(annotations[: int(len(annotations) * 0.8)]),
    )
    (ytest) = (
        np.asarray(annotations[int(len(annotations) * 0.8):]),
    )

    return ytrain, ytest



def load_lidc_idri_per_annotation(image_size, annotation_size_perc=1, pad=0):
    total_images = 0
    annotation_list = pl.query(pl.Annotation)
    annotations_count = int(annotation_list.count() * annotation_size_perc)
    images = []
    annotations = []
    for i in range(0, annotations_count):
        annotation = annotation_list[i]
        annotation_bbox = annotation.bbox(pad)
        vol = annotation.scan.to_volume(verbose=False)

        y0, y1 = annotation_bbox[0].start, annotation_bbox[0].stop
        x0, x1 = annotation_bbox[1].start, annotation_bbox[1].stop

        # Get the central slice of the computed bounding box.
        i, j, k = annotation.centroid
        z = max(int(annotation_bbox[2].stop - k) - 1, 0)
        (w, h) = vol[:, :, int(k)].shape

        image = Image.fromarray(__normalize(vol[:, :, int(z)]))
        image = image.resize((image_size, image_size))

        images.append(keras.utils.img_to_array(image))
        scaled_bbox = (float(x0) / w, float(y0) / h, float(x1) / w, float(y1) / h)

        # apply relative scaling to bounding boxes as per given image and append to list
        annotations.append(scaled_bbox)
        total_images = total_images + 1

    # Convert the list to numpy array, split to train and test dataset
    (xtrain), (ytrain) = (
        np.asarray(images[: int(len(images) * 0.8)]),
        np.asarray(annotations[: int(len(annotations) * 0.8)]),
    )
    (xtest), (ytest) = (
        np.asarray(images[int(len(images) * 0.8):]),
        np.asarray(annotations[int(len(annotations) * 0.8):]),
    )

    return xtrain, ytrain, xtest, ytest

def build_pre_processed_dataset_consensus(image_size, base_path, size_perc=1, image_count=-1, pad=0,consensus_level=0.5):
    if image_size != 64 and image_size != 128 and image_size != 256 and image_size != 512:
        raise Exception('image_size parameter must have the value: 64, 128, 256 or 512. ' + str(image_size) +
                        ' is not a valid value')
    if base_path is None:
        raise Exception('base_path is mandatory')

    Path(base_path + PRE_PROCESSED_FOLDER_PREFIX + str(image_size) + '-' + str(pad) + '-consensus-' + str(consensus_level)).mkdir(parents=True, exist_ok=True)
    scan_list = pl.query(pl.Scan)
    scan_list_size = int(scan_list.count() * size_perc)
    annotations = []
    total_images = 0
    for i in range(0, scan_list_size):
        scan = scan_list[i]
        nodules = scan.cluster_annotations()
        vol = scan.to_volume(verbose=False)
        nodules_count = len(nodules)
        for j in range(0, nodules_count):
            nodule = nodules[j]
            a, cbbox, b = consensus(nodule, clevel=consensus_level, pad=[(pad, pad), (pad, pad), (0, 0)])
            y0, y1 = cbbox[0].start, cbbox[0].stop
            x0, x1 = cbbox[1].start, cbbox[1].stop
            k = int(0.5 * (cbbox[2].stop - cbbox[2].start))
            z = max(int(x1 - k) - 1, 0)
            (w, h) = vol[:, :, int(k)].shape

            image = vol[:, :, int(k)]
            #image = Image.fromarray(__normalize(vol[:, :, int(k)]))
            #if image_size != 512:
            #    image = image.resize((image_size, image_size))
            scaled_bbox = (float(x0) / w, float(y0) / h, float(x1) / w, float(y1) / h)

            total_images = total_images + 1

            __save_object(scaled_bbox,
                          base_path + PRE_PROCESSED_FOLDER_PREFIX + str(image_size) + '-' + str(pad) + os.sep + SCALED_BOX_
                          + str(total_images) + DATA_FILE_EXT)
            __save_object(image,
                          base_path + PRE_PROCESSED_FOLDER_PREFIX + str(image_size) + '-' + str(pad) + os.sep + LIDC_IMAGE_
                          + str(total_images) + DATA_FILE_EXT)

            total_images = total_images + 1
            if image_count > 0 and total_images >= image_count:
                break

    return total_images



def load_lidc_idri_per_consensus(image_size, size_perc=1, pad=0, consensus_level=0.5):
    scan_list = pl.query(pl.Scan)
    scan_list_size = int(scan_list.count() * size_perc)
    images = []
    annotations = []
    total_images = 0
    for i in range(0, scan_list_size):
        scan = scan_list[i]
        nodules = scan.cluster_annotations()
        vol = scan.to_volume(verbose=False)
        nodules_count = len(nodules)
        for j in range(0, nodules_count):
            nodule = nodules[j]
            a, cbbox, b = consensus(nodule, clevel=consensus_level, pad=[(pad, pad), (pad, pad), (0, 0)])
            y0, y1 = cbbox[0].start, cbbox[0].stop
            x0, x1 = cbbox[1].start, cbbox[1].stop
            k = int(0.5 * (cbbox[2].stop - cbbox[2].start))
            z = max(int(x1 - k) - 1, 0)
            (w, h) = vol[:, :, int(k)].shape

            image = Image.fromarray(normalize(vol[:, :, int(k)]))
            image = image.resize((image_size, image_size))
            images.append(keras.utils.img_to_array(image))
            scaled_bbox = (float(x0) / w, float(y0) / h, float(x1) / w, float(y1) / h)

            annotations.append(scaled_bbox)
            total_images = total_images + 1


    # Convert the list to numpy array, split to train and test dataset
    (xtrain), (ytrain) = (
        np.asarray(images[: int(len(images) * 0.8)]),
        np.asarray(annotations[: int(len(annotations) * 0.8)]),
    )
    (xtest), (ytest) = (
        np.asarray(images[int(len(images) * 0.8):]),
        np.asarray(annotations[int(len(annotations) * 0.8):]),
    )

    return xtrain, ytrain, xtest, ytest


def load_lidc_idri_per_consensus_no_norm(start=0, end=0, pad=0, consensus_level=0.5):
    scan_list = pl.query(pl.Scan)
    scan_list_size = min(scan_list.count(), end)
    images = []
    annotations = []
    for i in range(start, scan_list_size):
        scan = scan_list[i]
        nodules = scan.cluster_annotations()
        vol = scan.to_volume(verbose=False)
        nodules_count = len(nodules)
        for j in range(0, nodules_count):
            nodule = nodules[j]
            m_val = max(ann.malignancy for ann in nodule) #malignancy - how "cancerous" it is
            s_val = min(ann.subtlety for ann in nodule) #subtlety - how easy to detect
            a, cbbox, b = consensus(nodule, clevel=consensus_level, pad=[(pad, pad), (pad, pad), (0, 0)])
            y0, y1 = cbbox[0].start, cbbox[0].stop
            x0, x1 = cbbox[1].start, cbbox[1].stop
            k = int(0.5 * (cbbox[2].stop - cbbox[2].start))
            z = cbbox[2].start + k

            images.append(vol[:, :, int(z)])

            annotations.append((y0, y1, x0, x1))

    return images, annotations


# normalize an LIDC-IDRI image to grayscale
def normalize2(im):
    im[im < MIN_BOUND] = -1000
    im[im > MAX_BOUND] = 600
    im = (im + 1000) / (600 + 1000)
    return np.array(255 * im, dtype="uint8")

def load_lidc_idri_per_consensus_resize(image_size=512, start=0, end=0, pad=0, consensus_level=0.5):
    scan_list = pl.query(pl.Scan)
    scan_list_size = min(scan_list.count(), end)
    images = []
    annotations = []
    for i in range(start, scan_list_size):
        scan = scan_list[i]
        nodules = scan.cluster_annotations()
        vol = scan.to_volume(verbose=False)
        nodules_count = len(nodules)
        for j in range(0, nodules_count):
            nodule = nodules[j]
            a, cbbox, b = consensus(nodule, clevel=consensus_level, pad=[(pad, pad), (pad, pad), (0, 0)])
            y0, y1 = cbbox[0].start, cbbox[0].stop
            x0, x1 = cbbox[1].start, cbbox[1].stop
            k = int(0.5 * (cbbox[2].stop - cbbox[2].start))
            z = cbbox[2].start + k
            (w, h) = vol[:, :, int(z)].shape

            image = Image.fromarray(normalize2(vol[:, :, int(z)]))
            image = image.resize((image_size, image_size))
            images.append(keras.utils.img_to_array(image))

            annotations.append((int((y0 / h) * image_size), int((y1 / h) * image_size), int((x0 / w) * image_size), int((x1 / w) * image_size)))

    return images, annotations


def load_lidc_idri_per_consensus_class_resize(image_size=512, start=0, end=0, pad=0, consensus_level=0.5):
    scan_list = pl.query(pl.Scan)
    scan_list_size = min(scan_list.count(), end)
    images = []
    annotations = []
    for i in range(start, scan_list_size):
        scan = scan_list[i]
        nodules = scan.cluster_annotations()
        vol = scan.to_volume(verbose=False)
        nodules_count = len(nodules)
        for j in range(0, nodules_count):
            nodule = nodules[j]
            m_val = max(ann.malignancy for ann in nodule)  # malignancy - how "cancerous" it is
            s_val = min(ann.subtlety for ann in nodule)  # subtlety - how easy to detect
            a, cbbox, b = consensus(nodule, clevel=consensus_level, pad=[(pad, pad), (pad, pad), (0, 0)])
            y0, y1 = cbbox[0].start, cbbox[0].stop
            x0, x1 = cbbox[1].start, cbbox[1].stop
            k = int(0.5 * (cbbox[2].stop - cbbox[2].start))
            z = cbbox[2].start + k
            (w, h) = vol[:, :, int(z)].shape

            #image = Image.fromarray(normalize2(vol[:, :, int(z)]))
            #image = image.resize((image_size, image_size))
            #images.append(keras.utils.img_to_array(image))
            images.append(vol[:, :, int(z)])

            annotations.append((y0, y1 , x0, x1, m_val, s_val))

    return images, annotations

def __normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > (1 - PIXEL_MEAN)] = 1.
    image[image < (0 - PIXEL_MEAN)] = 0.
    return np.array(255 * image, dtype="uint8")


def __save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def normalize(im):
    im[im > 400] = 400
    im[im < -1000] = -1000

    im = (255 - 0)/(400 - (-1000)) * (im - 400) + 255
    return im.astype(np.uint8)
