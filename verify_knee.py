"""Verify knee angle values in the slip vs normal states files."""
import os
import numpy as np
import opensim as osim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "ResultsSlip")

col_path = "/jointset/knee_r/knee_angle_r/value"

# Read both the .mot (degrees) and .sto (radians) files
for label, fname in [("degrees .mot", "slip_simulation_states_degrees.mot"),
                     ("radians .sto", "slip_simulation_states.sto")]:
    print(f"\n=== {label} ===")
    for condition, subdir in [("Normal", "normal"), ("Slip", "")]:
        fpath = os.path.join(RESULTS_DIR, subdir, fname)
        stor = osim.Storage(fpath)
        col = osim.ArrayDouble()
        stor.getDataColumn(col_path, col)

        n = col.getSize()
        t0 = stor.getStateVector(0).getTime()
        t_end = stor.getStateVector(n - 1).getTime()

        print(f"  {condition:8s}: first={col.get(0):.6f}  last={col.get(n-1):.6f}"
              f"  t=[{t0:.6f}, {t_end:.6f}]  nFrames={n}")

# Also print a few time points side by side from the .mot file
print("\n=== Time series comparison (degrees .mot) ===")
print(f"{'time':>10s} {'Normal':>12s} {'Slip':>12s} {'Diff':>10s}")
print("-" * 48)

normal_stor = osim.Storage(os.path.join(RESULTS_DIR, "normal",
                                         "slip_simulation_states_degrees.mot"))
slip_stor = osim.Storage(os.path.join(RESULTS_DIR,
                                       "slip_simulation_states_degrees.mot"))

normal_col = osim.ArrayDouble()
slip_col = osim.ArrayDouble()
normal_stor.getDataColumn(col_path, normal_col)
slip_stor.getDataColumn(col_path, slip_col)

n = min(normal_col.getSize(), slip_col.getSize())
for i in range(0, n, max(1, n // 10)):
    t = normal_stor.getStateVector(i).getTime()
    nv = normal_col.get(i)
    sv = slip_col.get(i)
    print(f"{t:10.6f} {nv:12.4f} {sv:12.4f} {sv - nv:10.4f}")

# Print last row
t = normal_stor.getStateVector(n - 1).getTime()
nv = normal_col.get(n - 1)
sv = slip_col.get(n - 1)
print(f"{t:10.6f} {nv:12.4f} {sv:12.4f} {sv - nv:10.4f}  <-- LAST")
