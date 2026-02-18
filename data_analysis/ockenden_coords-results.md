(.venv) ana@MU00236940:~/Desktop/code/Data/Bedmap/Ockenden$ python ockenden_coords.py 
================================================================================
SUBSET BEDMAP DATA FOR OCKENDEN et al. (2025) COMPARISON
================================================================================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ASB_ICECAP_2010  (UTIG_2010_ICECAP_AIR_BM3.csv)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Rows: 9,372,658  |  Valid bed picks: 4,983,753
  Lat:  [-84.42, -56.66]
  Lon:  [-180.00, 180.00]

  ✓ Fig4_Aurora_SB: 386,009 valid points
    Aurora Subglacial Basin — classified as low-relief
    Subset lat: [-76.000, -71.000]
    Subset lon: [105.000, 125.000]
    Bed elev:   [-1771, 930] m
    Trajectories: 73

    ┌─ COPY INTO YOUR ANALYSIS PIPELINE ───────────────┐
    {
        'file': 'UTIG_2010_ICECAP_AIR_BM3.csv',
        'label': 'ASB_ICECAP_2010_Fig4_Aurora_SB',
        'subset': lambda df, _r={
            'lat_min': -76.0, 'lat_max': -71.0,
            'lon_min': 105.0, 'lon_max': 125.0,
        }: df[
            (df['latitude (degree_north)'] >= _r['lat_min']) &
            (df['latitude (degree_north)'] <= _r['lat_max']) &
            (df['longitude (degree_east)']  >= _r['lon_min']) &
            (df['longitude (degree_east)']  <= _r['lon_max'])
        ].copy(),
    },
    └────────────────────────────────────────────────────┘

  ✓ Fig2F_Resolution_SH: 41,371 valid points
    Resolution Subglacial Highlands — alpine valleys
    Subset lat: [-76.000, -73.000]
    Subset lon: [135.001, 150.000]
    Bed elev:   [-1256, 793] m
    Trajectories: 7

    ┌─ COPY INTO YOUR ANALYSIS PIPELINE ───────────────┐
    {
        'file': 'UTIG_2010_ICECAP_AIR_BM3.csv',
        'label': 'ASB_ICECAP_2010_Fig2F_Resolution_SH',
        'subset': lambda df, _r={
            'lat_min': -76.0, 'lat_max': -73.0,
            'lon_min': 135.0, 'lon_max': 150.0,
        }: df[
            (df['latitude (degree_north)'] >= _r['lat_min']) &
            (df['latitude (degree_north)'] <= _r['lat_max']) &
            (df['longitude (degree_east)']  >= _r['lon_min']) &
            (df['longitude (degree_east)']  <= _r['lon_max'])
        ].copy(),
    },
    └────────────────────────────────────────────────────┘

  ✓ Fig2G_Highland_A: 124,409 valid points
    Highland A — paleo-river landscape
    Subset lat: [-76.000, -73.000]
    Subset lon: [118.000, 132.000]
    Bed elev:   [-3446, 930] m
    Trajectories: 31

    ┌─ COPY INTO YOUR ANALYSIS PIPELINE ───────────────┐
    {
        'file': 'UTIG_2010_ICECAP_AIR_BM3.csv',
        'label': 'ASB_ICECAP_2010_Fig2G_Highland_A',
        'subset': lambda df, _r={
            'lat_min': -76.0, 'lat_max': -73.0,
            'lon_min': 118.0, 'lon_max': 132.0,
        }: df[
            (df['latitude (degree_north)'] >= _r['lat_min']) &
            (df['latitude (degree_north)'] <= _r['lat_max']) &
            (df['longitude (degree_east)']  >= _r['lon_min']) &
            (df['longitude (degree_east)']  <= _r['lon_max'])
        ].copy(),
    },
    └────────────────────────────────────────────────────┘

  ✓ Fig2H_Golicyna_SH: 118,273 valid points
    Golicyna Subglacial Highlands — dendritic valleys
    Subset lat: [-75.000, -72.000]
    Subset lon: [103.000, 117.000]
    Bed elev:   [-1454, 635] m
    Trajectories: 19

    ┌─ COPY INTO YOUR ANALYSIS PIPELINE ───────────────┐
    {
        'file': 'UTIG_2010_ICECAP_AIR_BM3.csv',
        'label': 'ASB_ICECAP_2010_Fig2H_Golicyna_SH',
        'subset': lambda df, _r={
            'lat_min': -75.0, 'lat_max': -72.0,
            'lon_min': 103.0, 'lon_max': 117.0,
        }: df[
            (df['latitude (degree_north)'] >= _r['lat_min']) &
            (df['latitude (degree_north)'] <= _r['lat_max']) &
            (df['longitude (degree_east)']  >= _r['lon_min']) &
            (df['longitude (degree_east)']  <= _r['lon_max'])
        ].copy(),
    },
    └────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ASB_ICECAP_2008  (UTIG_2008_ICECAP_AIR_BM2.csv)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Rows: 1,072,811  |  Valid bed picks: 1,072,811
  Lat:  [-76.09, -66.27]
  Lon:  [88.29, 136.76]

  ✓ Fig4_Aurora_SB: 260,123 valid points
    Aurora Subglacial Basin — classified as low-relief
    Subset lat: [-76.000, -71.000]
    Subset lon: [105.000, 125.000]
    Bed elev:   [-1791, 725] m
    Trajectories: 260123

    ┌─ COPY INTO YOUR ANALYSIS PIPELINE ───────────────┐
    {
        'file': 'UTIG_2008_ICECAP_AIR_BM2.csv',
        'label': 'ASB_ICECAP_2008_Fig4_Aurora_SB',
        'subset': lambda df, _r={
            'lat_min': -76.0, 'lat_max': -71.0,
            'lon_min': 105.0, 'lon_max': 125.0,
        }: df[
            (df['latitude (degree_north)'] >= _r['lat_min']) &
            (df['latitude (degree_north)'] <= _r['lat_max']) &
            (df['longitude (degree_east)']  >= _r['lon_min']) &
            (df['longitude (degree_east)']  <= _r['lon_max'])
        ].copy(),
    },
    └────────────────────────────────────────────────────┘

  ✗ Fig2F_Resolution_SH: no overlap

  ✓ Fig2G_Highland_A: 46,733 valid points
    Highland A — paleo-river landscape
    Subset lat: [-75.449, -73.000]
    Subset lon: [118.000, 129.877]
    Bed elev:   [-1029, 927] m
    Trajectories: 46733

    ┌─ COPY INTO YOUR ANALYSIS PIPELINE ───────────────┐
    {
        'file': 'UTIG_2008_ICECAP_AIR_BM2.csv',
        'label': 'ASB_ICECAP_2008_Fig2G_Highland_A',
        'subset': lambda df, _r={
            'lat_min': -76.0, 'lat_max': -73.0,
            'lon_min': 118.0, 'lon_max': 132.0,
        }: df[
            (df['latitude (degree_north)'] >= _r['lat_min']) &
            (df['latitude (degree_north)'] <= _r['lat_max']) &
            (df['longitude (degree_east)']  >= _r['lon_min']) &
            (df['longitude (degree_east)']  <= _r['lon_max'])
        ].copy(),
    },
    └────────────────────────────────────────────────────┘

  ✓ Fig2H_Golicyna_SH: 112,332 valid points
    Golicyna Subglacial Highlands — dendritic valleys
    Subset lat: [-75.000, -72.000]
    Subset lon: [103.000, 117.000]
    Bed elev:   [-1469, 616] m
    Trajectories: 112332

    ┌─ COPY INTO YOUR ANALYSIS PIPELINE ───────────────┐
    {
        'file': 'UTIG_2008_ICECAP_AIR_BM2.csv',
        'label': 'ASB_ICECAP_2008_Fig2H_Golicyna_SH',
        'subset': lambda df, _r={
            'lat_min': -75.0, 'lat_max': -72.0,
            'lon_min': 103.0, 'lon_max': 117.0,
        }: df[
            (df['latitude (degree_north)'] >= _r['lat_min']) &
            (df['latitude (degree_north)'] <= _r['lat_max']) &
            (df['longitude (degree_east)']  >= _r['lon_min']) &
            (df['longitude (degree_east)']  <= _r['lon_max'])
        ].copy(),
    },
    └────────────────────────────────────────────────────┘




<><><><><><><><><><><>< YOU ARE HERE
=================================================




━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Rec_Catch  (BAS_2012_ICEGRAV_AIR_BM3.csv)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Rows: 2,633,648  |  Valid bed picks: 2,420,450
  Lat:  [-84.25, -75.28]
  Lon:  [-34.53, 27.22]

  ✓ Fig2D_Recovery_SB: 165,642 valid points
    Recovery Subglacial Basin — geological boundary
    Subset lat: [-82.399, -80.500]
    Subset lon: [-33.212, -15.000]
    Bed elev:   [-2192, 1689] m
    Trajectories: 9

    ┌─ COPY INTO YOUR ANALYSIS PIPELINE ───────────────┐
    {
        'file': 'BAS_2012_ICEGRAV_AIR_BM3.csv',
        'label': 'Rec_Catch_Fig2D_Recovery_SB',
        'subset': lambda df, _r={
            'lat_min': -83.5, 'lat_max': -80.5,
            'lon_min': -35.0, 'lon_max': -15.0,
        }: df[
            (df['latitude (degree_north)'] >= _r['lat_min']) &
            (df['latitude (degree_north)'] <= _r['lat_max']) &
            (df['longitude (degree_east)']  >= _r['lon_min']) &
            (df['longitude (degree_east)']  <= _r['lon_max'])
        ].copy(),
    },
    └────────────────────────────────────────────────────┘

  ✓ Fig1_Pensacola_Pole: 1,022 valid points
    Pensacola-Pole Basin — main comparison region (Fig 1)
    Subset lat: [-82.047, -82.000]
    Subset lon: [-20.587, -20.000]
    Bed elev:   [45, 1099] m
    Trajectories: 1

    ┌─ COPY INTO YOUR ANALYSIS PIPELINE ───────────────┐
    {
        'file': 'BAS_2012_ICEGRAV_AIR_BM3.csv',
        'label': 'Rec_Catch_Fig1_Pensacola_Pole',
        'subset': lambda df, _r={
            'lat_min': -88.0, 'lat_max': -82.0,
            'lon_min': -60.0, 'lon_max': -20.0,
        }: df[
            (df['latitude (degree_north)'] >= _r['lat_min']) &
            (df['latitude (degree_north)'] <= _r['lat_max']) &
            (df['longitude (degree_east)']  >= _r['lon_min']) &
            (df['longitude (degree_east)']  <= _r['lon_max'])
        ].copy(),
    },
    └────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  DML_AniRES  (AWI_2018_ANIRES_AIR_BM3.csv)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Rows: 121,401  |  Valid bed picks: 121,401
  Lat:  [-72.93, -70.55]
  Lon:  [-16.78, -3.33]

  ✗ Fig2A_Maud_SB: no overlap

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  POLARGAP_2015  (BAS_2015_POLARGAP_AIR_BM3.csv)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Rows: 911,113  |  Valid bed picks: 869,413
  Lat:  [-90.00, -80.72]
  Lon:  [-172.37, 167.25]

  ✓ Fig1_Pensacola_Pole: 118,982 valid points
    Pensacola-Pole Basin — main comparison region (Fig 1)
    Subset lat: [-88.000, -82.000]
    Subset lon: [-60.000, -20.001]
    Bed elev:   [-2131, 1325] m
    Trajectories: 21

    ┌─ COPY INTO YOUR ANALYSIS PIPELINE ───────────────┐
    {
        'file': 'BAS_2015_POLARGAP_AIR_BM3.csv',
        'label': 'POLARGAP_2015_Fig1_Pensacola_Pole',
        'subset': lambda df, _r={
            'lat_min': -88.0, 'lat_max': -82.0,
            'lon_min': -60.0, 'lon_max': -20.0,
        }: df[
            (df['latitude (degree_north)'] >= _r['lat_min']) &
            (df['latitude (degree_north)'] <= _r['lat_max']) &
            (df['longitude (degree_east)']  >= _r['lon_min']) &
            (df['longitude (degree_east)']  <= _r['lon_max'])
        ].copy(),
    },
    └────────────────────────────────────────────────────┘

  ✓ Fig2C_Hercules_Dome: 22,139 valid points
    Hercules Dome — U-shaped valleys
    Subset lat: [-87.500, -85.000]
    Subset lon: [-120.000, -100.077]
    Bed elev:   [-1014, 1834] m
    Trajectories: 4

    ┌─ COPY INTO YOUR ANALYSIS PIPELINE ───────────────┐
    {
        'file': 'BAS_2015_POLARGAP_AIR_BM3.csv',
        'label': 'POLARGAP_2015_Fig2C_Hercules_Dome',
        'subset': lambda df, _r={
            'lat_min': -87.5, 'lat_max': -85.0,
            'lon_min': -120.0, 'lon_max': -100.0,
        }: df[
            (df['latitude (degree_north)'] >= _r['lat_min']) &
            (df['latitude (degree_north)'] <= _r['lat_max']) &
            (df['longitude (degree_east)']  >= _r['lon_min']) &
            (df['longitude (degree_east)']  <= _r['lon_max'])
        ].copy(),
    },
    └────────────────────────────────────────────────────┘


================================================================================
SUMMARY
================================================================================
  ASB_ICECAP_2010      × Fig4_Aurora_SB            →  386,009 pts
  ASB_ICECAP_2010      × Fig2F_Resolution_SH       →   41,371 pts
  ASB_ICECAP_2010      × Fig2G_Highland_A          →  124,409 pts
  ASB_ICECAP_2010      × Fig2H_Golicyna_SH         →  118,273 pts
  ASB_ICECAP_2008      × Fig4_Aurora_SB            →  260,123 pts
  ASB_ICECAP_2008      × Fig2G_Highland_A          →   46,733 pts
  ASB_ICECAP_2008      × Fig2H_Golicyna_SH         →  112,332 pts
  Rec_Catch            × Fig2D_Recovery_SB         →  165,642 pts
  Rec_Catch            × Fig1_Pensacola_Pole       →    1,022 pts
  POLARGAP_2015        × Fig1_Pensacola_Pole       →  118,982 pts
  POLARGAP_2015        × Fig2C_Hercules_Dome       →   22,139 pts

NEXT STEPS:
  1. If a region shows 0 overlap, try expanding the box by 1-2 degrees.
  2. The Ockenden Metrics_v2/ NetCDFs contain gridded roughness metrics
     you can sample at your flight line locations for direct comparison:
       - ifpa_rms_slope.nc, ifpa_b1_5km.nc (fractal dimension)
       - ifpa_count_max_50.nc (hill counts)
       - bedmach_rms_slope.nc, etc. (BedMachine equivalents)
  3. BedMachine Antarctica v4 now includes IFPA topography:
     https://doi.org/10.5067/POJQI54A45HX
