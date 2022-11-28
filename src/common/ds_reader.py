import pickle

def save_ds_single_file(filename, obj):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
    pass

def load_ds_single_file(image_size, pad, scan_count_perc, type):
    with open(get_ds_single_file_name(image_size, pad, scan_count_perc, type), 'rb') as filePointer:
        xtrain, ytrain, _, _ = pickle.load(filePointer)

    with open(get_ds_single_file_name(512, pad, scan_count_perc, type), 'rb') as filePointer:
        _, _, xtest, ytest = pickle.load(filePointer)

    return xtrain, ytrain, xtest, ytest

def load_annotation_file(pad):
    with open(get_annotation_file_name(pad), 'rb') as filePointer:
        ytrain, ytest = pickle.load(filePointer)
    return ytrain, ytest

def get_ds_single_file_name(image_size, pad, scan_count_perc, type):
    return 'C:\\temp\\LIDC-' + str(image_size) + '-' + str(pad) + '-' + str(scan_count_perc) + '-' + type + '.pkl'

def get_annotation_file_name(pad):
    return 'C:\\temp\\LIDC-' + str(pad) + '-annotations.pkl'

