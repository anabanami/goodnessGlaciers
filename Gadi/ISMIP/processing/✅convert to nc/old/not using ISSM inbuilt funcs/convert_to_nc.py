#!/usr/bin/env python3
"""
Convert ISSM .outbin files to NetCDF format (flat or ISSM-style grouped).

Adds support for writing variables under groups: results/TransientSolution
so downstream tools expecting ISSM-like layout can read them.

Usage:
    python convert_to_nc_grouped.py input.outbin [output.nc] [--layout grouped|flat] [--analyse-only]
"""

import numpy as np
import netCDF4 as nc
import struct
import os
import argparse
from datetime import datetime
import sys


class ISSMOutbinReader:
    """Reader for ISSM .outbin binary format"""
    
    def __init__(self, filename):
        self.filename = filename
        self.file_handle = None
        self.results = []
        
    def __enter__(self):
        self.file_handle = open(self.filename, 'rb')
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_handle:
            self.file_handle.close()
    
    def read_cstring(self, length):
        data = self.file_handle.read(length)
        if len(data) != length:
            raise ValueError(f"Expected {length} bytes, got {len(data)}")
        return data.rstrip(b'\x00').decode('utf-8')
    
    def read_result_header(self):
        try:
            length_data = self.file_handle.read(4)
            if len(length_data) != 4:
                return None  # EOF
            name_length = struct.unpack('<i', length_data)[0]
            if name_length <= 0 or name_length > 1000:
                raise ValueError(f"Invalid name length: {name_length}")
            result_name = self.read_cstring(name_length)
            time_data = self.file_handle.read(8)
            if len(time_data) != 8:
                raise ValueError("Could not read time")
            time = struct.unpack('<d', time_data)[0]
            step_data = self.file_handle.read(4)
            if len(step_data) != 4:
                raise ValueError("Could not read step")
            step = struct.unpack('<i', step_data)[0]
            return {'name': result_name, 'time': time, 'step': step}
        except (struct.error, ValueError) as e:
            print(f"Error reading header: {e}")
            return None
    
    def read_result_data(self, header):
        try:
            type_data = self.file_handle.read(4)
            if len(type_data) != 4:
                raise ValueError("Could not read type")
            data_type = struct.unpack('<i', type_data)[0]
            result = header.copy()
            result['type'] = data_type
            
            if data_type == 1:
                size_data = self.file_handle.read(4)
                _ = struct.unpack('<i', size_data)[0]
                value_data = self.file_handle.read(8)
                value = struct.unpack('<d', value_data)[0]
                result['data'] = value
                result['shape'] = ()
            elif data_type == 2:
                length_data = self.file_handle.read(4)
                length = struct.unpack('<i', length_data)[0]
                string_value = self.read_cstring(length)
                result['data'] = string_value
                result['shape'] = ()
            elif data_type == 3:
                rows_data = self.file_handle.read(4)
                cols_data = self.file_handle.read(4)
                rows = struct.unpack('<i', rows_data)[0]
                cols = struct.unpack('<i', cols_data)[0]
                array_size = rows * cols
                array_data = self.file_handle.read(array_size * 8)
                values = struct.unpack(f'<{array_size}d', array_data)
                result['data'] = np.array(values).reshape(rows, cols)
                result['shape'] = (rows, cols)
            elif data_type == 4:
                rows_data = self.file_handle.read(4)
                cols_data = self.file_handle.read(4)
                rows = struct.unpack('<i', rows_data)[0]
                cols = struct.unpack('<i', cols_data)[0]
                array_size = rows * cols
                array_data = self.file_handle.read(array_size * 4)
                values = struct.unpack(f'<{array_size}i', array_data)
                result['data'] = np.array(values).reshape(rows, cols)
                result['shape'] = (rows, cols)
            elif data_type == 5:
                rows_data = self.file_handle.read(4)
                cols_data = self.file_handle.read(4)
                rows = struct.unpack('<i', rows_data)[0]
                cols = struct.unpack('<i', cols_data)[0]
                array_size = rows * cols
                real_data = self.file_handle.read(array_size * 8)
                real_values = struct.unpack(f'<{array_size}d', real_data)
                imag_data = self.file_handle.read(array_size * 8)
                imag_values = struct.unpack(f'<{array_size}d', imag_data)
                complex_values = [complex(r, i) for r, i in zip(real_values, imag_values)]
                result['data'] = np.array(complex_values).reshape(rows, cols)
                result['shape'] = (rows, cols)
            else:
                raise ValueError(f"Unknown data type: {data_type}")
            return result
        except (struct.error, ValueError) as e:
            print(f"Error reading data: {e}")
            return None
    
    def read_all_results(self):
        self.results = []
        while True:
            header = self.read_result_header()
            if header is None:
                break
            result = self.read_result_data(header)
            if result is None:
                break
            self.results.append(result)
            print(f"Read result: {result['name']} (step={result['step']}, time={result['time']:.6f}, type={result['type']})")
        return self.results

def create_array_variable(target, var_name, shape, data_type):
    dim1_name = f'{var_name}_dim1'
    dim2_name = f'{var_name}_dim2'
    if dim1_name not in target.dimensions:
        target.createDimension(dim1_name, shape[0])
    if dim2_name not in target.dimensions:
        target.createDimension(dim2_name, shape[1])
    dtype_map = {3: 'f8', 4: 'i4', 5: 'c16'}
    return target.createVariable(var_name, dtype_map[data_type], ('time', dim1_name, dim2_name))

def fill_array_variable(var, var_results, steps):
    for r in var_results:
        step_idx = steps.index(r['step'])
        var[step_idx] = r['data']

def create_netcdf_from_outbin(input_file, output_file=None, layout="grouped"):
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}.nc"
    with ISSMOutbinReader(input_file) as reader:
        results = reader.read_all_results()
    steps = sorted(set(r['step'] for r in results))
    times = {r['step']: r['time'] for r in results}
    with nc.Dataset(output_file, 'w', format='NETCDF4') as ncfile:
        ncfile.title = "ISSM Simulation Output"
        ncfile.description = f"Converted from ISSM .outbin format: {os.path.basename(input_file)}"
        ncfile.creation_date = datetime.now().isoformat()
        ncfile.source = "ISSM"
        ncfile.converter = "convert_to_nc_grouped.py"
        ncfile.original_file = input_file
        if layout == "grouped":
            g_results = ncfile.createGroup("results")
            target = g_results.createGroup("TransientSolution")
        else:
            target = ncfile
        target.createDimension('time', len(steps))
        time_var = target.createVariable('time', 'f8', ('time',))
        time_var[:] = [times[s] for s in steps]
        step_var = target.createVariable('step', 'i4', ('time',))
        step_var[:] = steps
        var_groups = {}
        for r in results:
            var_groups.setdefault(r['name'], []).append(r)
        for var_name, var_results in var_groups.items():
            shape = var_results[0]['shape']
            data_type = var_results[0]['type']
            if data_type in [3,4,5]:
                var = create_array_variable(target, var_name, shape, data_type)
                fill_array_variable(var, var_results, steps)
                var.issm_type = data_type
                var.original_shape = shape
            elif data_type == 1:
                var = target.createVariable(var_name, 'f8', ('time',))
                for r in var_results:
                    var[steps.index(r['step'])] = r['data']
            elif data_type == 2:
                max_len = max(len(str(r['data'])) for r in var_results)
                dim_name = f'{var_name}_strlen'
                target.createDimension(dim_name, max_len)
                var = target.createVariable(var_name, 'S1', ('time', dim_name))
                for r in var_results:
                    s = str(r['data']).encode('utf-8')
                    padded = s + b'\x00' * (max_len - len(s))
                    var[steps.index(r['step']), :] = np.frombuffer(padded, dtype='S1')
    print(f"Conversion complete: {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('output_file', nargs='?')
    parser.add_argument('--analyse-only', action='store_true')
    parser.add_argument('--layout', choices=['grouped','flat'], default='grouped')
    args = parser.parse_args()
    if args.analyse_only:
        with ISSMOutbinReader(args.input_file) as reader:
            reader.read_all_results()
    else:
        create_netcdf_from_outbin(args.input_file, args.output_file, layout=args.layout)

if __name__ == "__main__":
    main()
