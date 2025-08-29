
"""
Usage:

python extract_results.py flowline_27_S4_non_periodic_static.nc

OR

python extract_results.py --pattern='flowline_27_S4_*.nc'

Will create:

flowline_27_S4_periodic_transient/
├── Vx/
│   ├── Vx_step00.png
│   ├── Vx_step01.png
│   └── ...
├── Vel/
│   └── Vel_step00.png
├── Pressure/
│   └── Pressure_step00.png
"""

import glob
import os
import sys
import time
from multiprocessing import Pool, cpu_count
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from bamgflowband import bamgflowband
from matplotlib.colors import TwoSlopeNorm

# Get parent directory and add to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from bedrock_generator import SyntheticBedrockModelConfig
# from domain_utils import find_optimal_domain_length

FIELD_CONFIG = {
    'Vx': {
        'cmap': 'coolwarm',
        'norm': lambda d: TwoSlopeNorm(vcenter=0,
                                       vmin=-np.max(np.abs(d)),
                                       vmax= np.max(np.abs(d)))
    },
    'Vel': {
        'cmap': 'viridis',
        'norm': lambda d: None
    },
    'Pressure': {
        'cmap': 'plasma',
        'norm': lambda d: None
    },
    # …add more fields here as needed…
}

# Constants matching flowline.py
YTS = 31556926  # seconds per year
# Domain parameters will be determined dynamically from bedrock config
PROFILE_DIR = os.path.join(parent_dir, "bedrock_profiles")


def get_profile_id(filename):
    """Extract profile ID from filename."""
    base = os.path.splitext(os.path.basename(filename))[0]
    profile_str = base.split('_')[0]
    try:
        return int(profile_str)
    except ValueError:
        raise ValueError(f"Couldn't parse profile ID from '{base}'")


def to_m_per_year(var, arr):
    """Convert velocity-like arrays from m/s to m/yr if needed."""
    units = getattr(var, 'units','').lower()
    if ('m/s' in units) or ('m s-1' in units) or ('m s^-1' in units):
        return arr * YTS, 'm/yr'
    # already m/yr or unknown
    if ('m/a' in units) or ('m/yr' in units) or ('m a-1' in units) or ('m yr-1' in units):
        return arr, 'm/yr'
    return arr, units or ''


def get_resolution_factor(filename):
    """Extract resolution factor from filename (e.g., '165_S3_1.25_*.nc' -> 1.25)."""
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split('_')

    # Expected format: {profile_id}_{experiment}_{resolution_factor}_*
    if len(parts) >= 3:
        try:
            return float(parts[2])
        except ValueError:
            pass

    # Fallback to default if not found
    print(f"⚠️  Warning: Could not extract resolution factor from {base}")
    return None


def build_mesh(profile_id, resolution_factor=None):
    """Build mesh using bedrock config with adaptive resolution matching flowline.py."""
    bedrock_config = SyntheticBedrockModelConfig(profile_id, PROFILE_DIR)

    # Default resolution factor based on flowline.py (you may need to adjust this)
    if resolution_factor is None:
        resolution_factor = 1.0  # Match the active line in flowline.py

    # Use same domain optimization as flowline.py
    L = 210e3  # Updated to match flowline.py
    # L = find_optimal_domain_length(bedrock_config, target_L)

    # Adjust nx proportionally to maintain resolution (matching flowline.py)
    nx = 2100  # Updated to match flowline.py 
    # nx = int(target_nx * L / target_L)

    x_1D = np.linspace(0, L, nx)
    b0 = bedrock_config.get_bedrock_elevation(x_1D)
    s0 = b0 + bedrock_config.ice_thickness

    # Get bedrock wavelength from config for adaptive meshing
    bed_wavelength = bedrock_config.profile_params['wavelength']
    ice_thickness = bedrock_config.ice_thickness

    # Exact adaptive_bamg logic from flowline.py
    wavelength_thickness_ratio = bed_wavelength / ice_thickness

    if bed_wavelength < 15000:
        refinement_factor = 50
    else:
        refinement_factor = 200

    hmax = (bed_wavelength / refinement_factor) * resolution_factor

    # Use adaptive bamg meshing with exact parameters from flowline.py
    md = bamgflowband(None, x_1D, s0, b0,
                      'hmax', hmax,
                      'anisomax', 3,
                      'vertical', 1)

    print(f"Mesh built: L={L/1000:.3f}km, {md.mesh.numberofvertices} vertices, wavelength={bed_wavelength:.0f}m, hmax={hmax:.1f}m")
    return md.mesh.x, md.mesh.y, md.mesh.elements - 1


def plot_field(x, y, elements, data, field, title, filename, triang=None, cbar_label=None):
    # Reuse triangulation if provided (major optimization)
    if triang is None:
        triang = tri.Triangulation(x*1e-3, y*1e-3, triangles=elements)

    # fetch the config, falling back to a sane default
    props = FIELD_CONFIG.get(field, {
        'cmap': 'viridis',
        'norm': lambda d: None
    })
    cmap = props['cmap']
    norm = props['norm'](data)

    fig, ax = plt.subplots(figsize=(16, 5))
    tc = ax.tripcolor(
        triang, data,
        cmap=cmap,
        norm=norm,
        shading='gouraud'
    )
    cbar = fig.colorbar(tc, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label)

    ax.set_title(title)
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.grid(True, linestyle=":")
    ax.set_aspect(23)

    # Lower DPI for faster save, higher quality isn't needed for many plots
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_single_field(args):
    """Worker function for parallel plotting"""
    x, y, elements, data, field, title, filename_out, cbar_label = args

    # Create triangulation in worker process
    triang = tri.Triangulation(x*1e-3, y*1e-3, triangles=elements)

    # fetch the config, falling back to a sane default
    props = FIELD_CONFIG.get(field, {
        'cmap': 'viridis',
        'norm': lambda d: None
    })
    cmap = props['cmap']
    norm = props['norm'](data)

    fig, ax = plt.subplots(figsize=(16, 5))
    tc = ax.tripcolor(
        triang, data,
        cmap=cmap,
        norm=norm,
        shading='gouraud'
    )
    cbar = fig.colorbar(tc, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label)

    ax.set_title(title)
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.grid(True, linestyle=":")
    ax.set_aspect(23)

    fig.savefig(filename_out, dpi=150)
    plt.close(fig)

    return f"Plotted {filename_out}"

def visualise_file(filename):
    """Process and visualize a NetCDF file."""
    start_time = time.time()

    print(f"\n📂 Processing: {filename}")
    profile_id = get_profile_id(filename)
    resolution_factor = get_resolution_factor(filename)

    mesh_start = time.time()
    x, y, elements = build_mesh(profile_id, resolution_factor)
    mesh_time = time.time() - mesh_start
    print(f"⏱️  Mesh building: {mesh_time:.1f}s")
    print(f"📊 Mesh info: {len(x)} vertices, {len(elements)} triangles, resolution_factor={resolution_factor}")

    ds = nc.Dataset(filename, 'r')
    out_dir = os.path.splitext(os.path.basename(filename))[0]
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n📂 Processing: {filename}")

    # INVESTIGATION: What's actually in the NetCDF file?
    print(f"\n🔍 NETCDF FILE STRUCTURE INVESTIGATION:")
    print(f"Root dimensions: {list(ds.dimensions.keys())}")
    print(f"Root variables: {list(ds.variables.keys())}")
    print(f"Groups: {list(ds.groups.keys())}")

    # Check for mesh information
    if 'Mesh' in ds.groups:
        mesh_group = ds.groups['Mesh']
        print(f"Mesh group variables: {list(mesh_group.variables.keys())}")
        for var_name in mesh_group.variables:
            var = mesh_group.variables[var_name]
            print(f"  {var_name}: shape {var.shape}")

    # Check dimensions of data variables
    if 'results' in ds.groups and 'TransientSolution' in ds.groups['results'].groups:
        tsol = ds.groups['results'].groups['TransientSolution']
        print(f"TransientSolution dimensions: {list(tsol.dimensions.keys())}")
        for dim_name, dim in tsol.dimensions.items():
            print(f"  {dim_name}: {len(dim)}")
    print(f"🔍 END INVESTIGATION\n")

    # Check if we have the grouped structure
    if 'results' in ds.groups and 'TransientSolution' in ds.groups['results'].groups:
        # Grouped structure from convert_to_nc.py
        tsol = ds.groups['results'].groups['TransientSolution']
        times = tsol.variables['time'][:]
        n_steps = len(times)
        data_source = tsol.variables
        structure_type = "grouped"
    else:
        # Legacy flat structure fallback
        if 'time' in ds.dimensions:
            n_steps = len(ds.dimensions['time'])
            times = ds.variables['time'][:]
        else:
            n_steps = 1
            times = np.array([0])
        data_source = ds.variables
        structure_type = "flat"

    print(f"Structure: {structure_type}, Time steps: {n_steps}")

    # Determine time in years for plot titles
    if n_steps > 1:
        # Check time units if available
        time_var = tsol.variables['time'] if structure_type == "grouped" else ds.variables['time']
        tunits = getattr(time_var, 'units', '').lower()
        
        # Smart detection: if times are reasonable year values (0.001 to 10000), assume they're already in years
        max_time = np.max(times)
        if 0.001 <= max_time <= 10000:  # Reasonable range for years in glaciology
            t_years = times
        elif 'sec' in tunits:
            t_years = times / YTS
        elif 'yr' in tunits or 'year' in tunits:
            t_years = times
        else:
            # Only convert from seconds if times are very large (typical seconds range)
            if max_time > 100000:  # Likely seconds if > 100k
                t_years = times / YTS
            else:
                t_years = times  # Assume already in years
    else:
        t_years = times.astype(float)

    # Load all data at once (and convert units where appropriate)
    data_start = time.time()
    print("Loading all data...")
    all_data = {}
    units_map = {}
    if n_steps == 1:
        # Static case
        fields_to_load = ['Vx','Vel','Pressure','Surface','Base']
        for field in fields_to_load:
            if field in data_source:
                var = data_source[field]
                data_array = np.squeeze(var[:])
                # convert units for velocities
                if field in ('Vx','Vy','Vel'):
                    data_array, ulabel = to_m_per_year(var, data_array)
                    units_map[field] = ulabel or getattr(var, 'units','')
                else:
                    units_map[field] = getattr(var, 'units','')
                print(f"  Loading {field} (static): shape {data_array.shape}")
                all_data[field] = data_array
    else:
        # Transient case
        fields_to_load = ['Vx','Vel','Pressure']
        for field in fields_to_load:
            if field in data_source:
                var = data_source[field]
                if structure_type == "grouped":
                    data_array = var[:, :]  # (Time, VertNum)
                else:
                    data_array = var[:]      # Load all at once
                # convert units for velocities
                if field in ('Vx','Vy','Vel'):
                    data_array, ulabel = to_m_per_year(var, data_array)
                    units_map[field] = ulabel or getattr(var, 'units','')
                else:
                    units_map[field] = getattr(var, 'units','')
                print(f"  Loading {field} (all {n_steps} timesteps): shape {data_array.shape}")
                all_data[field] = data_array

    ds.close()  # Close file early after loading data
    data_time = time.time() - data_start
    print(f"⏱️  Data loading: {data_time:.1f}s")

    # Validate dimensions match between mesh and data
    n_vertices = len(x)
    for field, data_array in all_data.items():
        if n_steps == 1:
            data_length = len(data_array)
        else:
            data_length = data_array.shape[1] if len(data_array.shape) > 1 else len(data_array)

        if data_length != n_vertices:
            print(f"❌ DIMENSION MISMATCH INVESTIGATION:")
            print(f"   Field: {field}")
            print(f"   Mesh vertices: {n_vertices}")
            print(f"   Data points: {data_length}")
            print(f"   Difference: {data_length - n_vertices}")
            print(f"   Let's investigate why this happens...")

            # Don't exit - continue to see if the NetCDF has mesh info

    print("✅ Dimension investigation completed!")

    plot_start = time.time()
    print("Starting visualization...")

    # Create output directories
    for field in all_data.keys():
        field_dir = os.path.join(out_dir, field)
        os.makedirs(field_dir, exist_ok=True)

    # Prepare all plotting tasks
    plot_tasks = []
    if n_steps == 1:
        # Static case plotting
        for field, data in all_data.items():
            filename_out = os.path.join(out_dir, field, f"{field}_step00.png")
            cbar_label = units_map.get(field) or None
            plot_tasks.append((x, y, elements, data, field, f"{field} (static)", filename_out, cbar_label))
    else:
        # Transient case plotting
        for i in range(1, n_steps):
            for field, data_array in all_data.items():
                data = np.squeeze(data_array[i, :]) if structure_type == "grouped" else np.squeeze(data_array[i])
                title = f"{field} at {t_years[i]:.3f} yr"
                cbar_label = units_map.get(field) or None
                filename_out = os.path.join(out_dir, field, f"{field}_step{i:02d}.png")
                plot_tasks.append((x, y, elements, data, field, title, filename_out, cbar_label))

    # Use parallel processing for plotting
    n_cores = min(cpu_count(), len(plot_tasks), 8)  # Limit to 8 cores max
    print(f"Starting parallel plotting with {n_cores} cores for {len(plot_tasks)} plots...")

    with Pool(n_cores) as pool:
        results = pool.map(plot_single_field, plot_tasks)

    print(f"Completed {len(results)} plots in parallel")

    plot_time = time.time() - plot_start
    total_time = time.time() - start_time
    print(f"⏱️  Plotting: {plot_time:.1f}s")
    print(f"⏱️  Total time: {total_time:.1f}s")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python extract_results.py <file.nc>")
        print("  python extract_results.py --pattern='*.nc'")
        sys.exit(1)

    if sys.argv[1].startswith("--pattern="):
        pattern = sys.argv[1].split("=", 1)[1]
        files = glob.glob(pattern)
        if not files:
            print(f"No files match pattern '{pattern}'")
            sys.exit(1)
        for file in files:
            visualise_file(file)
    else:
        visualise_file(sys.argv[1])
