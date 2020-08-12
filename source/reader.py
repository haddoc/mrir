import os
import json
import h5py
import numpy as np


def read_raw(file_name=None):
    """Import a raw data file and return metadata and raw datasets"""
    if not os.path.isfile(file_name) or not os.path.splitext(file_name)[-1] == '.h5':
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

    # Trim extra points
    extra_points = int(metadata['sequence']['readout']['readouts'][0]['extrapoints'])
    if extra_points > 0:
        for _k, _v in data.items():
            data[_k] = _v[:, :, extra_points:-extra_points]

    # Take only 1st series
    if 'set_echo' in index['scan']:
        sel_echo = np.array(index['scan']['set_echo']) == 0
        data['scan'] = data['scan'][sel_echo, :, :]
        for _k, _v in index['scan'].items():
            index['scan'][_k] = np.array(_v)[sel_echo]

    return data, index, metadata
