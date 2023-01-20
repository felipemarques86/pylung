import logging
import unittest
from pathlib import Path

from main.dataset_utilities.lidc_dataset_reader import LidcDatasetReader

logging.basicConfig(level=logging.DEBUG)

class TestDataSetLoader(unittest.TestCase):

    def test_build_pre_processed(self):
        source_path = Path(__file__).resolve()
        source_dir = str(source_path.parent.parent)

        lidc_dataset_reader = LidcDatasetReader(
            location=source_dir + '\\data\\temp\\',
            image_size=512,
            consensus_level=0.5,
            pad=0,
            part=0,
            max_parts=100
        )
        lidc_dataset_reader.load(dry_run=True)
        lidc_dataset_reader.save()

        self.assertIsNotNone(lidc_dataset_reader.images, "Train images is None")
        self.assertIsNotNone(lidc_dataset_reader.annotations, "Train bounding boxes is None")

        lidc_dataset_reader = lidc_dataset_reader.next()

        while lidc_dataset_reader.load(dry_run=True):
            self.assertIsNotNone(lidc_dataset_reader.images, "Train images is None")
            self.assertIsNotNone(lidc_dataset_reader.annotations, "Train bounding boxes is None")
            lidc_dataset_reader.save()
            lidc_dataset_reader = lidc_dataset_reader.next()

if __name__ == '__main__':
    unittest.main()