import configparser

import cv2
import numpy as np
import pylidc as pl
from prettytable import PrettyTable
from pylidc.utils import consensus

from main.dataset_utilities.dataset_reader_classes import DatasetReader, Dataset
from main.utilities.utilities_lib import error


class LidcDatasetReader(DatasetReader):
    def __init__(self, location, image_size, consensus_level, pad, part=0, part_size=1, num_parts=0) -> None:
        dataset = Dataset("LIDC-IDRI", "lib")
        self.image_size = image_size
        self.consensus_level = consensus_level
        self.pad = pad
        self.total_scan = 0

        super().__init__(dataset, location, part, part_size, num_parts)

    def has_image(self):
        return self.images != None or self.images.size == 0

    def clear(self):
        self.images = []
        self.annotations = []

    def save(self):
        file_name_annotations = self.get_ds_single_file_name('annotations-pt-' + str(self.part))
        file_name_images = self.get_ds_single_file_name('images-pt-' + str(self.part))
        self.save_ds_single_file(file_name_annotations, self.annotations)
        self.save_ds_single_file(file_name_images, self.images)

    def next(self):
        return LidcDatasetReader(self.location, self.image_size, self.consensus_level, self.pad, self.part + 1,
                                 self.part_size)

    def load(self, dry_run=False, walk=False):
        scan_list = pl.query(pl.Scan)
        self.total_scan = scan_list.count()
        start = int(self.part_size + 1) * self.part
        end = start + int(self.part_size)
        end = min(self.total_scan, end)
        images = []
        annotations = []
        if end > start:
            if (dry_run == False):
                for i in range(start, end):
                    scan = scan_list[i]
                    nodules = scan.cluster_annotations()
                    vol = scan.to_volume(verbose=False)
                    nodules_count = len(nodules)
                    for j in range(0, nodules_count):
                        nodule = nodules[j]
                        m_val = max(ann.malignancy for ann in nodule)  # malignancy - how "cancerous" it is
                        s_val = min(ann.subtlety for ann in nodule)  # subtlety - how easy to detect
                        a, cbbox, b = consensus(nodule, clevel=self.consensus_level,
                                                pad=[(self.pad, self.pad), (self.pad, self.pad), (0, 0)])
                        y0, y1 = cbbox[0].start, cbbox[0].stop
                        x0, x1 = cbbox[1].start, cbbox[1].stop
                        k = int(0.5 * (cbbox[2].stop - cbbox[2].start))
                        z = cbbox[2].start + k
                        metadata = 'slice=' + str(z) + ",scan=" + str(scan.id) + ",annotations="
                        for n in range(0, len(nodule)):
                            metadata += str(nodule[n].id) + "#"
                        (w, h) = vol[:, :, int(z)].shape
                        if w == self.image_size:
                            images.append(vol[:, :, int(z)])
                        else:
                            images.append(self.resize(vol[:, :, int(z)]))

                        annotations.append((y0, y1, x0, x1, m_val, s_val, metadata))

            self.images = images
            self.annotations = annotations
            return True
        return False

    # private methods
    def get_ds_single_file_name(self, type):
        if self.image_size == -1:
            return self.location + 'LIDC-RAW-' + str(self.pad) + '-' + type + '.pkl'
        return self.location + 'LIDC-' + str(self.image_size) + '-' + str(self.pad) + '-' + type + '.pkl'

    def save_ds_single_file(self, filename, obj):
        with open(filename, 'wb') as outp:
            import pickle
            pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

    def resize(self, image):
        im = np.float32(image)
        # im[im < MIN_BOUND] = -1000
        # im[im > MAX_BOUND] = 600
        # im = (im + 1000) / (600 + 1000)
        # im = im * 255

        #image = Image.fromarray(im)

        image = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

        image = cv2.resize(image, (self.image_size, self.image_size))
        #image = keras.utils.img_to_array(image)



        return np.int8(image)


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

    def load_custom(self, dry_run=False, walk=False):
        self.clear()
        for part in range(self.part, self.num_parts):
            file_name_annotations = self.get_ds_single_file_name('annotations-pt-' + str(part))
            file_name_images = self.get_ds_single_file_name('images-pt-' + str(part))

            images = self.__load_ds(file_name_images)
            annotations = self.__load_ds(file_name_annotations)

            for i in range(0, len(images)):
                if len(self.dataset_image_transformers) > 0:
                    for j in range(0, len(self.dataset_image_transformers)):
                        if walk:
                            self.dataset_image_transformers[j].walk(images[i])
                        else:
                            self.images.append(self.dataset_image_transformers[j].execute(images[i]))
                else:
                    self.images.append(images[i])
                if len(self.dataset_data_transformers) > 0:
                    for j in range(0, len(self.dataset_data_transformers)):
                        if walk:
                            self.dataset_data_transformers[j].walk(annotations[i])
                        else:
                            self.annotations.append(self.dataset_data_transformers[j].execute(annotations[i]))
                else:
                    self.annotations.append(annotations[i])
        if len(self.images) == 0:
            error('Dataset has no images/data!')
            exit(-1)

    def split(self, value):
        pass

    def __load_ds(self, filename):
        with open(filename, 'rb') as filePointer:
            import pickle
            data = pickle.load(filePointer)
        return data
