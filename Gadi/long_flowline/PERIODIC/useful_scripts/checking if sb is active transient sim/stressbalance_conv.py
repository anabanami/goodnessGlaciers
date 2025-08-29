import netCDF4 as nc, numpy as np

fn = "165_S1_0.5.nc"
with nc.Dataset(fn) as ds:
    g = ds["results/TransientSolution"]
    s = g["StressbalanceConvergenceNumSteps"][:]  # shape (time,)
nz = np.flatnonzero(s>0)
print("SB ran at steps:", nz.tolist())
if nz.size>1:
    print("Δ between SB steps:", np.diff(nz).tolist(), " (mode ≈ coupling frequency)")


with nc.Dataset(fn) as ds:
    Vx = ds["results/TransientSolution/Vx"][:]  # (t, n, 1)
Vx = np.squeeze(Vx, axis=-1)
dV = np.max(np.abs(np.diff(Vx, axis=0)), axis=1)  # L∞ change per step
chg = np.flatnonzero(dV > (np.nanmax(dV)*1e-6 + 1e-12)) + 1  # conservative threshold
print("Steps with appreciable ΔVx:", chg.tolist())