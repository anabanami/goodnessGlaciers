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

# Parse scenario and resolution from input filename
# Support both old format: profile_scenario_h_v-type.outbin
# and new format with wavelength: profile_scenario_h_v_wX-type.outbin

# Try new format first (with wavelength factor)
pattern_new = r'([a-zA-Z0-9_]+)_(S\d+)_(\d+\.?\d*)_(\d+\.?\d*)_w(\d+\.?\d*)-(.+)\.outbin'
match = re.match(pattern_new, os.path.basename(input_file))

if match:
    profile, scenario, h_res, v_res, wavelength_factor, sol_type = match.groups()
    print(f"Detected new format with wavelength factor: w{wavelength_factor}")
else:
    # Fall back to old format (without wavelength factor)
    pattern_old = r'([a-zA-Z0-9_]+)_(S\d+)_(\d+\.?\d*)_(\d+\.?\d*)-(.+)\.outbin'
    match = re.match(pattern_old, os.path.basename(input_file))
    
    if match:
        profile, scenario, h_res, v_res, sol_type = match.groups()
        wavelength_factor = None
        print(f"Detected old format (no wavelength factor)")

if match:
    # bc_file = f"Boundary_conditions/{scenario}_F/IsmipF_{scenario}_{h_res}_{v_res}-BoundaryCondition.nc"
    if wavelength_factor:
        bc_file = f"Boundary_conditions/{profile}/{profile}_{scenario}_{h_res}_{v_res}_w{wavelength_factor}-BoundaryCondition.nc"
    else:
        bc_file = f"Boundary_conditions/{profile}/{profile}_{scenario}_{h_res}_{v_res}-BoundaryCondition.nc"

    if os.path.exists(bc_file):
        print(f"Loading model from: {bc_file}")
        md = loadmodel(bc_file)

# Continue with existing loadresultsfromdisk
md = loadresultsfromdisk(md, input_file)

output_file = input_file.replace('.outbin', '.nc')
print(f"{input_file} -> {output_file}")

export_netCDF(md, output_file)






