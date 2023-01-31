import pylidc as pl
from pylidc.utils import consensus

scan_list = pl.query(pl.Scan)
images = []
annotations = []
i = 38
scan = scan_list[i]
nodules = scan.cluster_annotations()
vol = scan.to_volume(verbose=False)
nodules_count = len(nodules)
if nodules_count == 0:
    for j in range(0, vol.shape[2]):
        images.append(vol[:, :, j])
        metadata = 'slice=' + str(j) + ",scan=" + str(scan.id) + ",annotations=None"
        annotations.append((0, 0, 0, 0, 0, 0, metadata))
else:
    for j in range(0, nodules_count):
        nodule = nodules[j]
        m_val = max(ann.malignancy for ann in nodule)  # malignancy - how "cancerous" it is
        s_val = min(ann.subtlety for ann in nodule)  # subtlety - how easy to detect
        a, cbbox, b = consensus(nodule, clevel=0.5)
        y0, y1 = cbbox[0].start, cbbox[0].stop
        x0, x1 = cbbox[1].start, cbbox[1].stop
        k = int(0.5 * (cbbox[2].stop - cbbox[2].start))
        z = cbbox[2].start + k
        metadata = 'slice=' + str(z) + ",scan=" + str(scan.id) + ",annotations="
        for n in range(0, len(nodule)):
            metadata += str(nodule[n].id) + "#"
        (w, h) = vol[:, :, int(z)].shape
        images.append(vol[:, :, int(z)])
        annotations.append((y0, y1, x0, x1, m_val, s_val, metadata))
