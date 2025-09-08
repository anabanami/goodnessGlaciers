#!/usr/bin/env python3
"""
A script to convert ISSM .outbin files to NetCDF format with nested structure preservation.
Uses ISSM's built-in loadresultsfromdisk or loadresultsfromcluster functions.

Handles files with naming pattern: IsmipF_SN_X_Y-Transient.outbin
where N ∈ {1,2,3,4}, X ∈ {0.5,1,1.5,2}, Y ∈ {0.5,1,1.5,2}

Output: IsmipF_SN_X_Y-Transient.nc

Usage:
    python outbin_to_nc_ISSM_converter.py [input_file] [--batch] [--from-cluster]
    
    Options:
    input_file     - Specific .outbin file to convert (optional if --batch)
    --batch        - Process all matching .outbin files in current directory
    --from-cluster - Use loadresultsfromcluster instead of loadresultsfromdisk
"""

import os
import sys
import argparse
import glob
import re
from datetime import datetime
import numpy as np
import netCDF4 as nc

# --- Import ISSM functions ---
try:
    from loadresultsfromdisk import loadresultsfromdisk
    from loadresultsfromcluster import loadresultsfromcluster
    from results import results
    # Import model dependencies  
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'm', 'classes'))
    from model import model
except ImportError as e:
    print(f"Error: Could not import ISSM module: {e}")
    print("Attempting to create minimal mock model class...")
    
    # Fallback mock model if ISSM model class not available
    class MockObject:
        def __repr__(self): 
            return f"<MockObject {self.__dict__}>"
    
    class model:
        """Minimal mock of ISSM model class for loadresultsfromdisk."""
        def __init__(self):
            self.qmu = MockObject()
            self.qmu.isdakota = False
            self.results = results()
            self.settings = MockObject()
            self.settings.io_gather = False
            self.miscellaneous = MockObject()
            self.miscellaneous.name = ''
            self.private = MockObject()
            self.private.solution = None
            self.private.runtimename = ''
            self.cluster = MockObject()

def is_valid_filename(filename):
    """Check if filename matches the expected pattern: IsmipF_SN_X_Y-Transient.outbin"""
    pattern = r'^IsmipF_S[1-4]_[0-9.]+_[0-9.]+-Transient\.outbin$'
    return re.match(pattern, os.path.basename(filename)) is not None

def get_output_filename(input_file):
    """Convert .outbin filename to .nc filename"""
    base_name = os.path.splitext(input_file)[0]
    return f"{base_name}.nc"

def find_matching_outbin_files():
    """Find all .outbin files matching the expected pattern in current directory"""
    pattern = "IsmipF_S[1-4]_*_*-Transient.outbin"
    files = glob.glob(pattern)
    valid_files = [f for f in files if is_valid_filename(f)]
    return sorted(valid_files)

def convert_outbin_to_nc(input_file, output_file=None, use_cluster=False):
    """
    Convert ISSM .outbin file to NetCDF format preserving nested structure.
    
    Args:
        input_file: Path to .outbin file
        output_file: Output .nc file path (auto-generated if None)
        use_cluster: Use loadresultsfromcluster instead of loadresultsfromdisk
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return False

    if not is_valid_filename(input_file):
        print(f"Warning: '{input_file}' doesn't match expected naming pattern.")
        print("Expected pattern: IsmipF_SN_X_Y-Transient.outbin")
        
    if output_file is None:
        output_file = get_output_filename(input_file)

    print(f"Converting: {input_file} -> {output_file}")
    
    # Create ISSM model object
    md = model()
    md.miscellaneous.name = os.path.splitext(input_file)[0]
    
    # Load results using appropriate ISSM function
    print("Loading results using ISSM built-in functions...")
    try:
        if use_cluster:
            md = loadresultsfromcluster(md)
        else:
            md = loadresultsfromdisk(md, input_file)
    except Exception as e:
        print(f"Error loading results: {type(e).__name__}: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure all ISSM Python modules are in PYTHONPATH")
        print("2. Check that .outbin file is complete and valid")
        print("3. Verify ISSM installation is complete")
        return False

    # Verify results were loaded
    solution_type = md.private.solution
    if solution_type is None:
        print("Error: No solution found in results.")
        return False
    
    print(f"Solution type: {solution_type}")
    solution_results = getattr(md.results, solution_type)
    
    # Handle different result structures based on solution type
    if solution_type == 'TransientSolution':
        # For TransientSolution, the results are in a solution object with steps attribute
        if hasattr(solution_results, 'steps'):
            solution_results = solution_results.steps
        elif isinstance(solution_results, list):
            # Already a list of steps
            pass
        else:
            print("Warning: Unexpected TransientSolution structure")
            solution_results = [solution_results]
    else:
        # For non-transient solutions, ensure it's in list format for uniform processing
        if not isinstance(solution_results, list):
            solution_results = [solution_results]
    
    print(f"Found {len(solution_results)} time step(s)")
    
    # Create NetCDF file with nested structure
    try:
        _write_netcdf_with_nested_structure(solution_results, solution_type, input_file, output_file)
        print(f"Successfully converted to: {output_file}")
        return True
    except Exception as e:
        print(f"Error writing NetCDF file: {type(e).__name__}: {e}")
        return False

def _write_netcdf_with_nested_structure(solution_results, solution_type, input_file, output_file):
    """Write results to NetCDF preserving ISSM's nested structure"""
    
    with nc.Dataset(output_file, 'w', format='NETCDF4') as ncfile:
        # Global attributes
        ncfile.title = "ISSM Simulation Output"
        ncfile.source = "ISSM"
        ncfile.original_file = os.path.basename(input_file)
        ncfile.creation_date = datetime.now().isoformat(timespec='seconds')
        ncfile.solution_type = solution_type
        ncfile.time_steps = len(solution_results)
        
        # Create nested group structure: results/[solution_type]/
        results_group = ncfile.createGroup('results')
        solution_group = results_group.createGroup(solution_type)
        
        num_steps = len(solution_results)
        solution_group.createDimension('time', num_steps)
        
        # Analyze first result to determine dimensions
        first_result = solution_results[0]
        result_vars = vars(first_result)
        
        # Find maximum array dimension
        max_vertices = 0
        array_vars = {}
        scalar_vars = {}
        
        for var_name, var_value in result_vars.items():
            if var_name.startswith('_') or var_name in ['SolutionType', 'errlog', 'outlog', 'step', 'time']:
                continue
                
            if isinstance(var_value, np.ndarray):
                if var_value.ndim > 0:
                    max_vertices = max(max_vertices, var_value.shape[0])
                    array_vars[var_name] = var_value
                else:
                    scalar_vars[var_name] = var_value
            elif isinstance(var_value, (int, float, np.number)):
                scalar_vars[var_name] = var_value
        
        # Create dimensions
        if max_vertices > 0:
            solution_group.createDimension('vertices', max_vertices)
        
        # Create time coordinate variable
        if hasattr(first_result, 'time'):
            time_var = solution_group.createVariable('time', 'f8', ('time',))
            time_var.units = 'years'
            time_var.long_name = 'time'
            time_var[:] = [getattr(res, 'time', i) for i, res in enumerate(solution_results)]
        
        # Create step variable if available
        if hasattr(first_result, 'step'):
            step_var = solution_group.createVariable('time_step', 'i4', ('time',))
            step_var.long_name = 'time step number'
            step_var[:] = [getattr(res, 'step', i) for i, res in enumerate(solution_results)]
        
        # Create variables for arrays
        for var_name, sample_array in array_vars.items():
            if sample_array.ndim == 1:
                var = solution_group.createVariable(var_name, 'f8', ('time', 'vertices'))
                var.long_name = var_name
                
                for i, res in enumerate(solution_results):
                    data = getattr(res, var_name, np.array([]))
                    if len(data) > 0:
                        # Pad or truncate to fit max_vertices
                        if len(data) > max_vertices:
                            data = data[:max_vertices]
                        elif len(data) < max_vertices:
                            padded = np.full(max_vertices, np.nan)
                            padded[:len(data)] = data
                            data = padded
                        var[i, :] = data
            elif sample_array.ndim == 2:
                # Handle 2D arrays - create additional dimension
                dim2_size = sample_array.shape[1]
                dim2_name = f'{var_name}_dim2'
                if dim2_name not in solution_group.dimensions:
                    solution_group.createDimension(dim2_name, dim2_size)
                
                var = solution_group.createVariable(var_name, 'f8', ('time', 'vertices', dim2_name))
                var.long_name = var_name
                
                for i, res in enumerate(solution_results):
                    data = getattr(res, var_name, np.array([]))
                    if data.size > 0:
                        var[i, :data.shape[0], :] = data
        
        # Create variables for scalars
        for var_name, sample_scalar in scalar_vars.items():
            var = solution_group.createVariable(var_name, 'f8', ('time',))
            var.long_name = var_name
            
            for i, res in enumerate(solution_results):
                var[i] = getattr(res, var_name, np.nan)
        
        # Add log information if available
        if hasattr(first_result, 'errlog') and first_result.errlog:
            errlog_str = '\n'.join(first_result.errlog) if isinstance(first_result.errlog, list) else str(first_result.errlog)
            solution_group.error_log = errlog_str
            
        if hasattr(first_result, 'outlog') and first_result.outlog:
            outlog_str = '\n'.join(first_result.outlog) if isinstance(first_result.outlog, list) else str(first_result.outlog)
            solution_group.output_log = outlog_str

def main():
    parser = argparse.ArgumentParser(
        description="Convert ISSM .outbin files to NetCDF format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file
  python outbin_to_nc_ISSM_converter.py IsmipF_S1_0.5_2-Transient.outbin
  
  # Convert single file with custom output name  
  python outbin_to_nc_ISSM_converter.py IsmipF_S1_0.5_2-Transient.outbin output.nc
  
  # Batch convert all matching files in current directory
  python outbin_to_nc_ISSM_converter.py --batch
  
  # Use cluster loading instead of disk loading
  python outbin_to_nc_ISSM_converter.py IsmipF_S1_0.5_2-Transient.outbin --from-cluster
        """)
    
    parser.add_argument('input_file', nargs='?', help="The input .outbin file")
    parser.add_argument('output_file', nargs='?', default=None, 
                       help="The output .nc file (auto-generated if not provided)")
    parser.add_argument('--batch', action='store_true', 
                       help="Process all matching .outbin files in current directory")
    parser.add_argument('--from-cluster', action='store_true',
                       help="Use loadresultsfromcluster instead of loadresultsfromdisk")
    
    args = parser.parse_args()
    
    # Handle batch mode
    if args.batch:
        print("Batch processing mode: searching for .outbin files...")
        outbin_files = find_matching_outbin_files()
        
        if not outbin_files:
            print("No matching .outbin files found in current directory.")
            print("Expected pattern: IsmipF_S[1-4]_*_*-Transient.outbin")
            return
        
        print(f"Found {len(outbin_files)} file(s) to process:")
        for f in outbin_files:
            print(f"  - {f}")
        
        successful = 0
        failed = 0
        
        for input_file in outbin_files:
            print(f"\n{'-'*50}")
            output_file = get_output_filename(input_file)
            
            if convert_outbin_to_nc(input_file, output_file, args.from_cluster):
                successful += 1
            else:
                failed += 1
        
        print(f"\n{'='*50}")
        print(f"Batch processing complete:")
        print(f"  Successfully converted: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Total processed: {len(outbin_files)}")
        
    else:
        # Single file mode
        if not args.input_file:
            parser.error("input_file is required when not using --batch mode")
        
        convert_outbin_to_nc(args.input_file, args.output_file, args.from_cluster)

if __name__ == "__main__":
    main()