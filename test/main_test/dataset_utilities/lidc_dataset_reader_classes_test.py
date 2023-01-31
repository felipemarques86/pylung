import logging
import unittest
from pathlib import Path

from main.dataset_utilities.lidc_dataset_reader_classes import LidcDatasetReader

logging.basicConfig(level=logging.DEBUG)

class LidcDatasetReaderClassesTest(unittest.TestCase):

    def test_build_pre_processed(self):
        source_path = Path(__file__).resolve()
        source_dir = str(source_path.parent.parent)

        lidc_dataset_reader = LidcDatasetReader(
            location=source_dir + '\\data\\temp\\',
            image_size=512,
            consensus_level=0.5,
            pad=0,
            part=2,
            part_size=100
        )
        lidc_dataset_reader.load()
        lidc_dataset_reader.save()

        lidc_dataset_reader = lidc_dataset_reader.next()

        while lidc_dataset_reader.load():
            lidc_dataset_reader.save()
            lidc_dataset_reader = lidc_dataset_reader.next()

if __name__ == '__main__':
    unittest.main()
