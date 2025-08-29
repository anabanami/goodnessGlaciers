# goodnessGlaciers
Playing with ISSM


# Generate 625 bedrock profiles in 1D
**usage** `bedrock_generator.py`
```bash
python bedrock_generator.py
```

# RUN Simulation
**usage** `flowline9_synthetic.py`:
```bash
# process profile_001 as default example
python flowline9_synthetic.py

#Process a single file for different profile
python flowline9_synthetic.py --profile 42 --output results
```

**usage** `run_batch.py`
```bash
python run_batch.py --profiles "1-10" --output batch_results

#specific profiles
python run_batch.py --profiles "1,5,25,42" --output selected_profiles

#parallel processing
python run_batch.py --profiles "1-50" --parallel 4 --output parallel_results
```

#### Convert from `.outbin` to `.nc` file
**usage**: `basal_friction_Budd.py`
```bash
python convert_to_netCDF.py --input flowline9_profile_XXX.outbin
```


#### Analysis stuff
**usage**: `basal_friction_Budd.py`
```bash
# Analyze a single profile
python basal_friction_Budd.py -p 5

# Analyze a range of profiles
python basal_friction_Budd.py -b 1-10

# Analyze specific profiles
python basal_friction_Budd.py -b 1,5,10,25
```

**usage** `extract_results.py`
```bash
# Process a single file
python extract_results.py flowline9_profile_XXX.nc

# Process all profile files with output to a specific directory
python extract_results.py --pattern='flowline9_profile_*.nc' --output-dir=results
```

**usage** `phase_analysis.py`
```bash
# Process a single file
python phase_analysis.py results/ice_flow_results_profile_XXX.nc

# Process multiple files in one command
python phase_analysis.py results/ice_flow_results_profile_XXX.nc results/ice_flow_results_profile_002.nc

# Specify a custom output directory
python phase_analysis.py results/ice_flow_results_profile_*.nc --output-dir=phase_analysis
```
