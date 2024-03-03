import ast
import os
import pickle
import configparser
from collections import Counter

import cv2
import numpy as np
import pylidc as pl
from prettytable import PrettyTable
from pylidc.utils import consensus

from main.dataset_utilities.dataset_reader_classes import DatasetReader, Dataset
from main.utilities.utilities_lib import error


class LidcDatasetReader(DatasetReader):
    def __init__(self, location: str, image_size: int, consensus_level: float, zip=True, first=0, last=0, data_transformer=None, im_transformer=None) -> None:
        """

        :param location: 
        :param image_size: 
        :param consensus_level: 
        """
        self.annotations_raw = []
        dataset = Dataset("LIDC-IDRI", "lib")
        self.image_size = image_size
        self.consensus_level = consensus_level
        self.total_scan = 0
        self.zip = zip
        self.first = first
        self.last = last
        self.data_transformer = data_transformer
        self.im_transformer = im_transformer

        super().__init__(dataset, location)

    def has_image(self) -> bool:
        """

        :rtype: bool
        :return:
        """
        return self.images is not None or self.images.size == 0

    def clear(self):
        self.images = []
        self.annotations = []

    def save(self, start=0):
        if self.zip:
            file_name_annotations = self.get_ds_single_file_name('annotations-pt')
            file_name_images = self.get_ds_single_file_name('images-pt')
            self.save_ds_single_file(file_name_annotations, self.annotations)
            self.save_ds_single_file(file_name_images, self.images)
        else:
            self.save_separated_files('image-', self.images, '.raw')
            self.save_ds_single_file(self.location + 'annotations.ann', self.annotations)
            self.save_ds_single_file(self.location + 'raw_annotations.ann', self.annotations_raw)
        return len(self.images)

    def load(self):
        scan_list = pl.query(pl.Scan)
        #self.total_scan = scan_list.count()
        start = int(min(scan_list.count(), self.first))
        end = int(min(scan_list.count(), self.last))
        images = []
        annotations = []
        annotations_raw = []
        if end > start:
            for i in range(start, end):
                scan = scan_list[i]
                nodules = scan.cluster_annotations()
                vol = scan.to_volume(verbose=False)
                nodules_count = len(nodules)
                print(scan)
                if nodules_count == 0:
                    for j in range(0, vol.shape[2]):
                        image = vol[:, :, j]
                        annotation = None
                        annotation_raw = (0, 0, 0, 0, 0, 0, j, j, False)
                        if self.data_transformer is not None:
                            annotation = self.data_transformer.execute(annotation_raw, image, self.image_size, self.image_size)

                        if annotation is not None:
                            annotations.append(annotation)
                            annotations_raw.append(annotation_raw)

                            if self.im_transformer is not None:
                                image = self.im_transformer.execute(image, annotation)
                            images.append(image)
                else:
                    for j in range(0, nodules_count):
                        nodule = nodules[j]
                        m_val = max(ann.malignancy for ann in nodule)  # malignancy - how "cancerous" it is
                        s_val = min(ann.subtlety for ann in nodule)  # subtlety - how easy to detect
                        a, cbbox, b = consensus(nodule, clevel=self.consensus_level,
                                                pad=[(0, 0), (0, 0), (0, 0)])
                        y0, y1 = cbbox[0].start, cbbox[0].stop
                        x0, x1 = cbbox[1].start, cbbox[1].stop
                        A = cbbox[2].start
                        B = cbbox[2].stop
                        k = int(0.5 * (B - A))
                        centroid = cbbox[2].start + k
                        for x in range(A, B):
                            image = vol[:, :, x]
                            annotation = None
                            annotation_raw = (y0, y1, x0, x1, m_val, s_val, x, scan.id, x == centroid)
                            if self.data_transformer is not None:
                                annotation = self.data_transformer.execute(annotation_raw, image, self.image_size, self.image_size)

                            if annotation is not None:
                                annotations.append(annotation)
                                annotations_raw.append(annotation_raw)

                                if self.im_transformer is not None:
                                    image = self.im_transformer.execute(image, annotation_raw)
                                images.append(image)


        #data augmentation

        _aggr_counter = Counter()
        for a in annotations:
            _aggr_counter[tuple(a)] += 1

        max_aggr = max(_aggr_counter, key=_aggr_counter.get)

        out_imgs = []
        out_anns = []
        out_anns_raw = []

        for i in _aggr_counter:
            if _aggr_counter[i] < _aggr_counter[max_aggr]:
                add_n_images = _aggr_counter[max_aggr] - _aggr_counter[i]
                first_images_of_clazz = [images[j] for j in range(len(images)) if
                                         annotations[j] == list(i)]
                first_annotations_of_clazz = [annotations[j] for j in range(len(annotations)) if
                                              annotations[j] == list(i)]
                self.augment(in_imgs=first_images_of_clazz, in_anns=first_annotations_of_clazz, amnt=add_n_images,
                             out_imgs=out_imgs, out_anns=out_anns, in_ann_raw=annotations_raw, out_ann_raw=out_anns_raw)

        images.extend(out_imgs)
        annotations.extend(out_anns)
        annotations_raw.extend(out_anns_raw)

        self.images = images
        self.annotations = annotations
        self.annotations_raw = annotations_raw

        print(annotations)
        print(annotations_raw)

        return len(self.annotations)


    # private methods
    def get_ds_single_file_name(self, type, other_location=None):
        location = self.location
        if other_location is not None:
            location = other_location
        if self.image_size == -1:
            return location + 'LIDC-RAW-' + type + '.pkl'
        return location + 'LIDC-' + str(self.image_size) + '-' + type + '.pkl'


    def save_ds_single_file(self, filename, obj):
        print('Saving file: ' + filename)
        print('Size: ' + str(len(obj)))
        with open(filename, 'wb') as outp:
            pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
        outp.close()

    def save_separated_files(self, prefix, obj, ext, start=0):
        mode = 'wb'

        location = self.location
        for i in range(0, len(obj)):
            with open(location + f'{prefix}{str(i+start)}{ext}', mode) as file:
                pickle.dump(obj[i], file, pickle.HIGHEST_PROTOCOL)
        file.close()

    def resize(self, image):
        im = np.float32(image)
        image = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        return np.int16(image)


class CustomLidcDatasetReader(LidcDatasetReader):

    def __init__(self, name, location, image_size, consensus_level) -> None:
        super().__init__(location, image_size, consensus_level)
        self.dataset = Dataset(name, "generated")

    def __init__(self, location):
        self.config = configparser.ConfigParser()
        config_file = self.config.read(location + 'config.ini')
        if len(config_file) == 0:
            raise Exception("config.ini file not found")

        self.name = self.config['DEFAULT']['name']
        self.image_size = int(self.config['DEFAULT']['image_size'])
        self.consensus_level = float(self.config['DEFAULT']['consensus_level'])
        self.type = self.config['DEFAULT']['type']
        self.dataset = Dataset(self.name, "generated")
        super().__init__(location, self.image_size, self.consensus_level)

    def export_as_table(self):
        table = PrettyTable(['Property', 'Value'])
        table.add_row(['Name', self.name])
        table.add_row(['Image size', self.image_size])
        table.add_row(['Consensus Level', self.consensus_level])
        table.add_row(['Type', self.type])
        return table

    def load_single(self, index, image_size =-1, dry_run=False, walk=False):
        self.clear()
        img_size = self.image_size
        if image_size > 0:
            img_size = image_size

        with open(self.location + f'image-{index}.raw', 'rb') as file:
            image = pickle.load(file)
        with open(self.location + f'annotation-{index}.txt', 'rb') as file:
            annotation = pickle.load(file)

        ann = None
        if self.filter is None or not self.filter(annotation):
            if len(self.dataset_data_transformers) > 0:
                for j in range(0, len(self.dataset_data_transformers)):
                        ann = self.dataset_data_transformers[j].execute(annotation, image, self.image_size, img_size)
                        if ann is not None:
                            self.annotations.append(ann)
            else:
                ann = annotation
                self.annotations.append(annotation)

            if len(self.dataset_image_transformers) > 0:
                for j in range(0, len(self.dataset_image_transformers)):
                    if ann is not None:
                        self.images.append(self.dataset_image_transformers[j].execute(image, annotation))
            else:
                if ann is not None:
                    self.images.append(image)
        if len(self.images) == 0:
            error('Dataset has no images/data!')
            exit(-1)

    def load_custom(self, image_size =-1, dry_run=False, walk=False):
        self.clear()
        img_size = self.image_size
        if image_size > 0:
            img_size = image_size

        file_name_annotations = self.get_ds_single_file_name('annotations-pt')
        file_name_images = self.get_ds_single_file_name('images-pt')

        images = self.__load_ds(file_name_images)
        annotations = self.__load_ds(file_name_annotations)

        for i in range(0, len(images)):
            ann = None
            if self.filter is None or not self.filter(annotations[i]):
                if len(self.dataset_data_transformers) > 0:
                    for j in range(0, len(self.dataset_data_transformers)):
                            ann = self.dataset_data_transformers[j].execute(annotations[i], images[i], self.image_size, img_size)
                            if ann is not None:
                                self.annotations.append(ann)
                else:
                    ann = annotations[i]
                    self.annotations.append(annotations[i])

                if len(self.dataset_image_transformers) > 0:
                    for j in range(0, len(self.dataset_image_transformers)):
                        if ann is not None:
                            self.images.append(self.dataset_image_transformers[j].execute(images[i], annotations[i]))
                else:
                    if ann is not None:
                        self.images.append(images[i])
        if len(self.images) == 0:
            error('Dataset has no images/data!')
            exit(-1)

    def load_simple(self):
        self.annotations = self.__load_ds(self.location + 'annotations.ann')
        self.annotations_raw = self.__load_ds(self.location + 'raw_annotations.ann')

        all_dot_raw_files_in_dir = [f for f in os.listdir(self.location) if f.endswith('.raw')]
        images = []
        for index in range(0, len(all_dot_raw_files_in_dir)):
            with open(self.location + 'image-' + str(index) + '.raw', 'rb') as file:
                image = pickle.load(file)
                images.append(image)
        self.images = images

        if len(self.images) != len(self.annotations):
            error('Dataset images size does not match annotations size!')
            exit(-1)

    def __load_ds(self, filename):
        with open(filename, 'rb') as filePointer:
            import pickle
            data = pickle.load(filePointer)
        return data


