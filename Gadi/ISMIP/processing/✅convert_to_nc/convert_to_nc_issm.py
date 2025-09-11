import os
import re
import argparse
import numpy as np
from model import model
from loadmodel import loadmodel
from export_netCDF import export_netCDF

# --- Import ISSM inbuilt functions ---
from loadresultsfromdisk import loadresultsfromdisk
from results import results

# naming
parser = argparse.ArgumentParser(description='Converter: outbin to nc')
parser.add_argument('input_file', help='Input file path (e.g., IsmipF_S*_*_*-Transient.outbin)')

args = parser.parse_args()
input_file = args.input_file

print(f"Input file: {input_file}")

# # Parse scenario and resolution from input filename
# match = re.match(r'IsmipF_(S\d)_(\d+\.?\d*)_(\d+\.?\d*)-Transient\.outbin', os.path.basename(input_file))
pattern = r'([a-zA-Z0-9_]+)_(S\d+)_(\d+\.?\d*)_(\d+\.?\d*)-Transient\.outbin'
match = re.match(pattern, os.path.basename(input_file))

if match:
    profile, scenario, h_res, v_res = match.groups()
    # bc_file = f"Boundary_conditions/{scenario}_F/IsmipF_{scenario}_{h_res}_{v_res}-BoundaryCondition.nc"
    bc_file = f"Boundary_conditions/{profile}/{profile}_{scenario}_{h_res}_{v_res}-BoundaryCondition.nc"

    if os.path.exists(bc_file):
        print(f"Loading model from: {bc_file}")
        md = loadmodel(bc_file)

# Continue with existing loadresultsfromdisk
md = loadresultsfromdisk(md, input_file)

output_file = input_file.replace('.outbin', '.nc')
print(f"{input_file} -> {output_file}")

export_netCDF(md, output_file)






