#!/usr/bin/env python
import sys
import os
import json
import numpy as np
import h5py


def trim_new_dataset(h5_old, h5_file):
    # copy metadata str
    h5_file['metadata'] = h5_old['metadata'][()]
    # create data groups 0=imaging, 1=noise
    h5_file.create_group('data')
    h5_file.create_group('data/0')
    h5_file.create_group('data/1')
    # only keep echo 0
    index_scan = json.loads(h5_old['data/0/index'][()])
    if 'set_echo' in index_scan and np.max(np.abs(np.diff(index_scan['set_echo']))) > 0:
        set_echo = np.array(index_scan['set_echo'])
        data_scan = h5_old['data/0/lines'][set_echo==0, :, :]
        for _k, _v in index_scan.items():
            index_scan[_k] = np.array(_v)[set_echo==0].tolist()
    # save dataset 0
    h5_file['data/0/lines'] = data_scan
    h5_file['data/0/index'] = json.dumps(index_scan)
    # save dataset 1
    h5_file['data/1/lines'] = h5_old['data/1/lines'][()]
    h5_file['data/1/index'] = h5_old['data/1/index'][()]


def trim_old_dataset(data_dict, h5_file):
    import numpy as np
    import copy
    # copy metadata str
    dict_meta = copy.deepcopy(data_dict['seqdata'])
    dict_meta = format_meta(dict_meta)
    h5_file['metadata'] = json.dumps(dict_meta)
    # create data groups 0=imaging, 1=noise
    h5_file.create_group('data')
    h5_file.create_group('data/0')
    h5_file.create_group('data/1')
    # From spectral to temporal
    data_time = np.fft.ifft(np.fft.ifftshift(data_dict['data']['freq_domain'], axes=-1), axis=-1)
    extra_points = dict_meta['sequence']['readout']['readouts'][0]['extrapoints']
    matrix_size = dict_meta['sequence']['readout']['readouts'][0]['matrix']
    data_time = data_time[:, :, extra_points:(extra_points + matrix_size)]
    # Get the imaging group from index
    index_all = data_dict['seqdata']['recon']['line_index']
    sel_0 = np.array(index_all['group']) == 0
    index_scan = {_k: np.array(_v)[sel_0] for _k, _v in index_all.items()}
    data_scan = data_time[sel_0, :, :]
    # only keep echo 0
    sel_echo = index_scan['set_echo'] == 0
    data_scan = data_scan[sel_echo, :, :]
    data_scan = np.transpose(data_scan, (1, 2, 0))
    index_scan = {_k: _v[sel_echo].tolist() for _k, _v in index_scan.items()}
    # save dataset 0
    h5_file['data/0/lines'] = data_scan
    h5_file['data/0/index'] = json.dumps(index_scan)
    # save dataset 1
    sel_1 = np.array(index_all['group']) == 1
    index_prescan = {_k: np.array(_v)[sel_1].tolist() for _k, _v in index_all.items()}
    data_prescan = data_time[sel_1, :, :]
    data_prescan = np.transpose(data_prescan, (1, 2, 0))
    h5_file['data/1/lines'] = data_prescan
    h5_file['data/1/index'] = json.dumps(index_prescan)


def format_meta(mdata):
        """Check data for proper JSON format, remove or replace numpy data"""
        if isinstance(mdata, dict):
            for _k, _v in mdata.items():
                mdata[_k] = format_meta(_v)
        elif isinstance(mdata, list):
            for _i, _v in enumerate(mdata):
                mdata[_i] = format_meta(_v)
        elif isinstance(mdata, np.ndarray):
            mdata = format_numpy(mdata)
        else:
            mdata = format_numpy(mdata)

        return mdata


def format_numpy(var):
    if isinstance(var, np.ndarray):
        var_list = var.tolist()
        return [format_numpy(_v) for _v in var_list]
    elif isinstance(var, np.bool_):
        return bool(var)
    elif isinstance(var, np.int_):
        return int(var)
    elif isinstance(var, np.float_):
        return float(var)
    elif isinstance(var, (complex, np.complex_)):
        return {'real': np.real(var), 'imag': np.imag(var)}
    else:
        return var


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Must specify file path")
        sys.exit(1)
    file_name = sys.argv[1]
    file_name_trimmed = os.path.splitext(file_name)[0] + "_trim.h5"
    h5_old = h5py.File(file_name,'r')
    h5_new = h5py.File(file_name_trimmed,'w')
    if 'seqdata' in h5_old:
        import deepdish as dd
        data_old = dd.io.load(file_name)
        trim_old_dataset(data_old, h5_new)
    else:
        trim_new_dataset(h5_old, h5_new)
    h5_new.close()
