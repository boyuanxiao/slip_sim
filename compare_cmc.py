"""Compare our CMC actuation forces against the reference output."""
import os
import numpy as np
import opensim as osim
from scipy.interpolate import interp1d

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_sto_as_dict(filepath):
    storage = osim.Storage(filepath)
    n_rows = storage.getSize()
    time = np.zeros(n_rows)
    for i in range(n_rows):
        time[i] = storage.getStateVector(i).getTime()
    data = {"time": time}
    col_labels = storage.getColumnLabels()
    for j in range(col_labels.getSize()):
        label = col_labels.get(j)
        if label == "time":
            continue
        col_data = osim.ArrayDouble()
        storage.getDataColumn(label, col_data)
        arr = np.zeros(col_data.getSize())
        for i in range(col_data.getSize()):
            arr[i] = col_data.get(i)
        data[label] = arr
    return data


our_file = os.path.join(BASE_DIR, "ResultsCMC", "subject01_walk1_Actuation_force.sto")
ref_file = os.path.join(
    BASE_DIR, "OutputReference", "ResultsCMC", "subject01_walk1_Actuation_force.sto"
)

our_data = load_sto_as_dict(our_file)
ref_data = load_sto_as_dict(ref_file)

our_cols = set(our_data.keys()) - {"time"}
ref_cols = set(ref_data.keys()) - {"time"}
common_cols = sorted(our_cols & ref_cols)

print(
    f"Our time range:  {our_data['time'][0]:.4f} - {our_data['time'][-1]:.4f} s"
    f" ({len(our_data['time'])} pts)"
)
print(
    f"Ref time range:  {ref_data['time'][0]:.4f} - {ref_data['time'][-1]:.4f} s"
    f" ({len(ref_data['time'])} pts)"
)

t_start = max(our_data["time"][0], ref_data["time"][0])
t_end = min(our_data["time"][-1], ref_data["time"][-1])
ref_mask = (ref_data["time"] >= t_start) & (ref_data["time"] <= t_end)
t_compare = ref_data["time"][ref_mask]

key_actuators = [
    "glut_med1_r", "soleus_r", "tib_ant_r", "med_gas_r",
    "vas_int_r", "rect_fem_r", "bifemlh_r",
    "glut_med1_l", "soleus_l", "tib_ant_l",
    "FX", "FY", "FZ",
]

print(f"\nComparison range: {t_start:.4f} - {t_end:.4f} s ({len(t_compare)} pts)")
print(f"\n{'Actuator':<28s} {'RMSE':>10s} {'nRMSE(%)':>10s} {'Max Ref':>10s}")
print("-" * 62)

all_nrmse = []
for col in common_cols:
    interp_func = interp1d(
        our_data["time"], our_data[col], kind="linear", fill_value="extrapolate"
    )
    our_interp = interp_func(t_compare)
    ref_vals = ref_data[col][ref_mask]
    rmse = np.sqrt(np.mean((our_interp - ref_vals) ** 2))
    max_ref = np.max(np.abs(ref_vals))
    nrmse = (rmse / max_ref * 100) if max_ref > 1e-6 else 0.0
    all_nrmse.append(nrmse)
    if col in key_actuators:
        print(f"{col:<28s} {rmse:>10.2f} {nrmse:>9.2f}% {max_ref:>10.2f}")

print("-" * 62)
print(f"{'Median nRMSE (all cols)':<28s} {'':>10s} {np.median(all_nrmse):>9.2f}%")
print(f"{'Mean nRMSE (all cols)':<28s} {'':>10s} {np.mean(all_nrmse):>9.2f}%")
print(f"{'Max nRMSE (all cols)':<28s} {'':>10s} {np.max(all_nrmse):>9.2f}%")
