import tensorflow as tf
from tensorflow.python.client import device_lib
import pylidc as pl

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
device_lib.list_local_devices()

#scans = pl.query(pl.Scan)

#print(scans)