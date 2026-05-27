import pandas as pd

# # test
# path = "v22/Ockenden-regions/window_csvs/ASB_ICECAP_2010_Fig4_Aurora_SB_w50km_window_stats.csv"
# df = pd.read_csv(path)
# ct = pd.crosstab(df.bed_class, df.relief_class, normalize='all') * 100
# print(ct.round(1))


# Loop over regions:

# NOTE THAT: if a bed_class category (e.g. "soft") has zero windows, its row is omitted entirely.   
from pathlib import Path

for f in sorted(Path("v22/Ockenden-regions/window_csvs").glob("*_window_stats.csv")): 
    df = pd.read_csv(f)
    name = f.stem.replace("_w50km_window_stats", "")
    print(f"\n=== {name} ===")
    print(pd.crosstab(df.bed_class, df.relief_class, normalize='all').mul(100).round(2))

  # On interpreting these crosstabs for geological character:

  # The crosstab gives you the joint distribution of spectral roughness character (beta-based)
  #  and relief amplitude. Here's how to read the dominant combinations:

# ┌──────────────────┬──────────────────────────────────────────────────────────────────┐
# │    Dominant      │                          Interpretation                          │
# │   combination    │                                                                  │
# ├──────────────────┼──────────────────────────────────────────────────────────────────┤
# │ hard +           │ Crystalline basement / shield terrain — rough at short           │
# │ subdued/flat     │ wavelengths but low total relief. Think cratonic bedrock.        │
# ├──────────────────┼──────────────────────────────────────────────────────────────────┤
# │ hard +           │ Active/recent orogen or volcanic terrain — rough texture and big │
# │ mountainous      │  topography.                                                     │
# ├──────────────────┼──────────────────────────────────────────────────────────────────┤
# │ soft +           │ Sediment-draped mountains — high relief but smoothed at short    │
# │ mountainous      │ wavelengths (marine sediments over buried topography, or glacial │
# │                  │  erosion smoothing peaks).                                       │
# ├──────────────────┼──────────────────────────────────────────────────────────────────┤
# │ soft +           │ Sedimentary basin — smooth at all scales. Classic soft-sediment  │
# │ flat/subdued     │ signature.                                                       │
# ├──────────────────┼──────────────────────────────────────────────────────────────────┤
# │ transitional +   │ Mixed — could be partially eroded mountains, or heterogeneous    │
# │ mountainous      │ geology.                                                         │
# ├──────────────────┼──────────────────────────────────────────────────────────────────┤
# │ chaotic +        │ Noisy/disordered — possibly heavily fractured, or data quality   │
# │ anything         │ issues.                                                          │
# └──────────────────┴──────────────────────────────────────────────────────────────────┘

#   Looking at your regions:

#   - Recovery SB (both datasets): dominated by transitional + mountainous (~30-34%) with some
#    soft — consistent with a subglacial basin margin with mixed geology.
#   - Resolution SH, Highland A, Golicyna SH, Aurora SB: all dominated by hard + subdued
#   (35-44%) — classic East Antarctic cratonic shield signature.
#   - Pensacola/Pole: transitional + subdued (26%) and hard + subdued (25%) — mixed character,
#    makes sense for the transition zone.
#   - Hercules Dome: transitional + subdued/mountainous dominant — transitional character
#   consistent with subglacial highlands.

#   A quick way to collapse this into a single "dominant terrain type" label per region: just
#   take the cell with the highest percentage. You could also compute a weighted summary
#   metric if you want something more continuous — e.g., a "hardness index" (fraction
#   hard+chaotic vs soft) and a "relief index" (fraction mountainous vs flat) as two
#   orthogonal axes, then plot regions in that 2D space.
