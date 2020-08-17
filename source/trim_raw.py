import sys
import os
import json
import numpy as np
import h5py

def trim_dataset(file_name=None):
    ds = h5py.File(file_name,'r')
    file_name_trimmed = os.path.splitext(file_name)[0] + "_trim.h5"
    ds2 = h5py.File(file_name_trimmed,'w')
    # copy metadata str
    ds2['metadata'] = ds['metadata'][()]
    # create data groups 0=imaging, 1=noise
    ds2.create_group('data')
    ds2.create_group('data/0')
    ds2.create_group('data/1')
    # only keep echo 0
    index_scan = json.loads(ds['data/0/index'][()])
    set_echo = np.array(index_scan['set_echo'])
    data_scan = ds['data/0/lines'][()][set_echo==0, :, :]
    for _k, _v in index_scan.items():
        index_scan[_k] = np.array(_v)[set_echo==0].tolist()
    # save dataset 0
    ds2['data/0/lines'] = data_scan
    ds2['data/0/index'] = index_scan
    ds2['data/0/index'] = json.dumps(index_scan)
    # save dataset 1
    ds2['data/1/lines'] = ds['data/1/lines'][()]
    ds2['data/1/index'] = ds['data/1/index'][()]
    ds2.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Must specify file path")
        return
    fname = sys.argv[1]
    trim_dataset(file_name=fname)
