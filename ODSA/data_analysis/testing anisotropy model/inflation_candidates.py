"""Both inflation formulas for each batch, so the write-up can name the one it uses."""
import numpy as np
import pandas as pd

BATCHES = ["null", "site_e_masked", "site_f_masked", "dubawnt_masked"]

print(f"{'batch':18s} {'SE':>8s} {'SD':>8s} {'robust_half_z':>13s} {'SD/median_se':>13s}")
for name in BATCHES:
    d = pd.read_csv(f"outputs/{name}_delta_beta.csv")
    z = d.delta / d.delta_se
    se = (np.percentile(d.delta, 84) - np.percentile(d.delta, 16)) / 2
    sd = d.delta.std()
    robust_half_z = (np.percentile(z, 84) - np.percentile(z, 16)) / 2
    sd_over_se = sd / d.delta_se.median()
    print(f"{name:18s} {se:8.4f} {sd:8.4f} {robust_half_z:13.4f} {sd_over_se:13.4f}")
