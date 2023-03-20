import pickle
import configparser

import cv2
import numpy as np
import pylidc as pl
from prettytable import PrettyTable
from pylidc.utils import consensus

from main.dataset_utilities.dataset_reader_classes import DatasetReader, Dataset
from main.utilities.utilities_lib import error


class LidcDatasetReader(DatasetReader):
    def __init__(self, location: str, image_size: int, consensus_level: float, pad: int, part: int = 0, part_size: int = 1, num_parts: int = 0, zip = True) -> None:
        """

        :param location: 
        :param image_size: 
        :param consensus_level: 
        :param pad: 
        :param part: 
        :param part_size: 
        :param num_parts: 
        """
        dataset = Dataset("LIDC-IDRI", "lib")
        self.image_size = image_size
        self.consensus_level = consensus_level
        self.pad = pad
        self.total_scan = 0
        self.zip = zip

        super().__init__(dataset, location, part, part_size, num_parts)

    def has_image(self) -> bool:
        """

        :rtype: bool
        :return:
        """
        return self.images is not None or self.images.size == 0

    def clear(self):
        self.images = []
        self.annotations = []

    def save(self, start = 0, other_location = None):
        if self.zip:
            file_name_annotations = self.get_ds_single_file_name('annotations-pt-' + str(self.part), other_location)
            file_name_images = self.get_ds_single_file_name('images-pt-' + str(self.part), other_location)
            self.save_ds_single_file(file_name_annotations, self.annotations)
            self.save_ds_single_file(file_name_images, self.images)
        else:
            self.save_separated_files('image-', self.images, '.raw', start, other_location)
            self.save_separated_files('annotation-', self.annotations, '.txt', start, other_location)

        return len(self.images)

    def next(self):
        return LidcDatasetReader(self.location, self.image_size, self.consensus_level, self.pad, self.part + 1,
                                 self.part_size, zip=self.zip)

    def load(self, iterative=False, walk=False):
        scan_list = pl.query(pl.Scan)
        self.total_scan = scan_list.count()
        start = int(self.part_size + 1) * self.part
        end = start + int(self.part_size)
        end = min(self.total_scan, end)
        images = []
        annotations = []
        if end > start:
            for i in range(start, end):
                scan = scan_list[i]
                nodules = scan.cluster_annotations()
                vol = scan.to_volume(verbose=False)
                nodules_count = len(nodules)
                if nodules_count == 0:
                    for j in range(0, vol.shape[2]):
                        images.append(vol[:, :, j])
                        annotations.append((0, 0, 0, 0, 0, 0, j, j, False))
                else:
                    for j in range(0, nodules_count):
                        nodule = nodules[j]
                        m_val = max(ann.malignancy for ann in nodule)  # malignancy - how "cancerous" it is
                        s_val = min(ann.subtlety for ann in nodule)  # subtlety - how easy to detect
                        a, cbbox, b = consensus(nodule, clevel=self.consensus_level,
                                                pad=[(self.pad, self.pad), (self.pad, self.pad), (0, 0)])
                        y0, y1 = cbbox[0].start, cbbox[0].stop
                        x0, x1 = cbbox[1].start, cbbox[1].stop
                        A = cbbox[2].start
                        B = cbbox[2].stop
                        k = int(0.5 * (B - A))
                        centroid = cbbox[2].start + k
                        for x in range(A, B):
                            images.append(vol[:, :, x])
                            annotations.append((y0, y1, x0, x1, m_val, s_val, x, centroid, x == centroid))

        self.images = images
        self.annotations = annotations
        return len(self.annotations) > 0


    # private methods
    def get_ds_single_file_name(self, type, other_location=None):
        location = self.location
        if other_location is not None:
            location = other_location
        if self.image_size == -1:
            return location + 'LIDC-RAW-' + str(self.pad) + '-' + type + '.pkl'
        return location + 'LIDC-' + str(self.image_size) + '-' + str(self.pad) + '-' + type + '.pkl'

    def save_ds_single_file(self, filename, obj):
        with open(filename, 'wb') as outp:
            pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

    def save_separated_files(self, prefix, obj, ext, start, other_location):
        mode = 'wb'
        if ext == 'txt' or ext == 'csv':
            mode = 'w'

        location = self.location
        if other_location is not None:
            location = other_location

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

    def __init__(self, name, location, image_size, consensus_level, pad, part, part_size, num_parts=0) -> None:
        super().__init__(location, image_size, consensus_level, pad, part, part_size, num_parts)
        self.dataset = Dataset(name, "generated")

    def __init__(self, location):
        config = configparser.ConfigParser()
        config_file = config.read(location + 'config.ini')
        if len(config_file) == 0:
            raise Exception("config.ini file not found")

        self.name = config['DEFAULT']['name']
        self.image_size = int(config['DEFAULT']['image_size'])
        self.consensus_level = float(config['DEFAULT']['consensus_level'])
        self.pad = int(config['DEFAULT']['pad'])
        self.part_size = int(config['DEFAULT']['part_size'])
        self.type = config['DEFAULT']['type']
        self.dataset = Dataset(self.name, "generated")
        self.num_parts = int(config['DEFAULT']['num_parts'])
        super().__init__(location, self.image_size, self.consensus_level, self.pad, 0, self.part_size, self.num_parts)

    def export_as_table(self):
        table = PrettyTable(['Property', 'Value'])
        table.add_row(['Name', self.name])
        table.add_row(['Image size', self.image_size])
        table.add_row(['Consensus Level', self.consensus_level])
        table.add_row(['PAD', self.pad])
        table.add_row(['Part Size', self.part_size])
        table.add_row(['Type', self.type])
        table.add_row(['Num parts', self.num_parts])
        return table

    def load_custom(self, image_size =-1, dry_run=False, walk=False):
        self.clear()
        img_size = self.image_size
        if image_size > 0:
            img_size = image_size
        for part in range(self.part, self.num_parts):
            file_name_annotations = self.get_ds_single_file_name('annotations-pt-' + str(part))
            file_name_images = self.get_ds_single_file_name('images-pt-' + str(part))

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

    def __load_ds(self, filename):
        with open(filename, 'rb') as filePointer:
            import pickle
            data = pickle.load(filePointer)
        return data

