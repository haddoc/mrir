import os
import json
import h5py
import numpy as np


def read_raw(file_name=None):
    """Import a raw data file and return metadata and raw datasets"""
    if not os.path.isfile(file_name) or os.path.splitext(file_name)[-1] != '.h5':
        print("'{}' is not a valid raw file".format(file_name))
        raise FileNotFoundError
    try:
        ds = h5py.File(file_name, "r")
    except:
        print("Could not open raw data file, check format.")
    data = {
        "scan": ds['data/0/lines'][()],
        "noise": ds['data/1/lines'][()],
    }
    index = {
        "scan": json.loads(ds['data/0/index'][()]),
        "noise": json.loads(ds['data/1/index'][()]),
    }
    metadata = json.loads(ds['metadata'][()])

    return data, index, metadata
