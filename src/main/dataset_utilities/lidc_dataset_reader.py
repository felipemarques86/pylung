import pylidc as pl
from pylidc.utils import consensus

from main.dataset_utilities.dataset_reader import DatasetReader, Dataset


class LidcDatasetReader(DatasetReader):
    def __init__(self, location, image_size, consensus_level, pad, part=0, max_parts=1) -> None:
        dataset = Dataset("LIDC-IDRI", "lib")
        self.image_size = image_size
        self.consensus_level = consensus_level
        self.pad = pad
        self.total_scan = 0
        super().__init__(dataset, location, part, max_parts)

    def has_image(self):
        return self.images != None or self.images.size == 0

    def clear(self):
        del self.images
        del self.annotations

    def save(self):
        file_name_annotations = self.__get_ds_single_file_name('annotations-pt-' + str(self.part))
        file_name_images = self.__get_ds_single_file_name('images-pt-' + str(self.part))
        self.__save_ds_single_file(file_name_annotations, self.annotations)
        self.__save_ds_single_file(file_name_images, self.images)

    def next(self):
        return LidcDatasetReader(self.location, self.image_size, self.consensus_level, self.pad, self.part + 1, self.max_parts)

    def load(self, dry_run=False):
        scan_list = pl.query(pl.Scan)
        self.total_scan = scan_list.count()
        start = int(self.max_parts + 1) * self.part
        end = start + int(self.max_parts)
        end = min(self.total_scan, end)
        images = []
        annotations = []
        if end > start:
            print("[INFO] Total scan " + str(self.total_scan))
            print("[INFO] Start from " + str(start) + " to " + str(end))
            if( dry_run == False):
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

                        images.append(vol[:, :, int(z)])

                        annotations.append((y0, y1, x0, x1, m_val, s_val, metadata))

            self.images = images
            self.annotations = annotations
            return True
        return False

    # private methods
    def __get_ds_single_file_name(self, type):
        if self.image_size == -1:
            return self.location + 'LIDC-RAW-' + str(self.pad) + '-' + type + '.pkl'
        return self.location + 'LIDC-' + str(self.image_size) + '-' + str(self.pad) + '-' + type + '.pkl'

    def __save_ds_single_file(self, filename, obj):
        with open(filename, 'wb') as outp:
            import pickle
            pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


class CustomLidcDatasetReader(LidcDatasetReader):

    def __init__(self, name, location, image_size, consensus_level, pad, part, max_parts) -> None:
        super().__init__(location, image_size, consensus_level, pad, part, max_parts)
        dataset = Dataset(name, "generated")

    def load(self):
        file_name_annotations = self.__get_ds_single_file_name('annotations-pt-' + str(self.part))
        file_name_images = self.__get_ds_single_file_name('images-pt-' + str(self.part))
        self.clear()

        images = self.__load_ds(file_name_images)
        annotations = self.__load_ds(file_name_annotations)

        has_normalizers = len(self.dataset_image_transformers) > 0
        has_transformers = len(self.dataset_data_transformers) > 0

        if not has_normalizers:
            self.images = images
        if not has_transformers:
            self.annotations = annotations

        if has_normalizers or has_transformers:
            for i in range(0, len(images)):
                for j in range(0, len(self.dataset_image_transformers)):
                        self.images.append(self.dataset_image_transformers[j].execute(images[i]))
                for j in range(0, len(self.dataset_data_transformers)):
                    self.annotations.append(self.dataset_data_transformers[j].execute(annotations[i]))

    def split(self, value):
        pass

    def __load_ds(self, filename):
        with open(filename, 'rb') as filePointer:
            import pickle
            data = pickle.load(filePointer)
        return data
