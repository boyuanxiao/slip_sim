"""
Run the OpenSim gait analysis pipeline: Scale -> IK -> RRA -> CMC
Then compare CMC actuation forces against reference output.
"""

import os
import sys
import numpy as np
import opensim as osim

# All paths relative to this script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

REF_DIR = os.path.join(BASE_DIR, "OutputReference")


def run_scale():
    """Run the Scale Tool to produce the subject-specific model."""
    print("\n" + "=" * 60)
    print("STEP 1: SCALING")
    print("=" * 60)

    tool = osim.ScaleTool(os.path.join(BASE_DIR, "subject01_Setup_Scale.xml"))
    tool.run()

    output_model = os.path.join(BASE_DIR, "subject01_simbody.osim")
    if not os.path.exists(output_model):
        raise RuntimeError("Scale tool did not produce subject01_simbody.osim")
    print(f"  -> Produced: subject01_simbody.osim")


def run_ik():
    """Run Inverse Kinematics to get joint angles from marker data."""
    print("\n" + "=" * 60)
    print("STEP 2: INVERSE KINEMATICS")
    print("=" * 60)

    tool = osim.InverseKinematicsTool(
        os.path.join(BASE_DIR, "subject01_Setup_IK.xml")
    )
    tool.run()

    output_mot = os.path.join(BASE_DIR, "subject01_walk1_ik.mot")
    if not os.path.exists(output_mot):
        raise RuntimeError("IK tool did not produce subject01_walk1_ik.mot")
    print(f"  -> Produced: subject01_walk1_ik.mot")


def run_rra():
    """Run Residual Reduction Algorithm to adjust kinematics and model."""
    print("\n" + "=" * 60)
    print("STEP 3: RESIDUAL REDUCTION ALGORITHM (RRA)")
    print("=" * 60)

    # Create results directory if needed
    os.makedirs(os.path.join(BASE_DIR, "ResultsRRA"), exist_ok=True)

    tool = osim.RRATool(os.path.join(BASE_DIR, "subject01_Setup_RRA.xml"))
    tool.run()

    adjusted_model = os.path.join(BASE_DIR, "subject01_simbody_adjusted.osim")
    if not os.path.exists(adjusted_model):
        raise RuntimeError("RRA did not produce subject01_simbody_adjusted.osim")
    print(f"  -> Produced: subject01_simbody_adjusted.osim")
    print(f"  -> Results in: ResultsRRA/")


def run_cmc():
    """Run Computed Muscle Control to get muscle activations/forces."""
    print("\n" + "=" * 60)
    print("STEP 4: COMPUTED MUSCLE CONTROL (CMC)")
    print("=" * 60)

    # Create results directory if needed
    os.makedirs(os.path.join(BASE_DIR, "ResultsCMC"), exist_ok=True)

    tool = osim.CMCTool(os.path.join(BASE_DIR, "subject01_Setup_CMC.xml"))
    tool.run()

    actuation_file = os.path.join(
        BASE_DIR, "ResultsCMC", "subject01_walk1_Actuation_force.sto"
    )
    if not os.path.exists(actuation_file):
        raise RuntimeError("CMC did not produce actuation force file")
    print(f"  -> Produced: ResultsCMC/subject01_walk1_Actuation_force.sto")


def load_sto_as_dict(filepath):
    """Load an OpenSim .sto file into a dict of {column_name: numpy_array}."""
    storage = osim.Storage(filepath)
    n_rows = storage.getSize()

    # Get time column
    time = np.zeros(n_rows)
    for i in range(n_rows):
        time[i] = storage.getStateVector(i).getTime()

    data = {"time": time}

    # Get all data columns
    col_labels = storage.getColumnLabels()
    n_cols = col_labels.getSize()
    for j in range(n_cols):
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


def compare_cmc_results():
    """Compare our CMC actuation forces against the reference output."""
    print("\n" + "=" * 60)
    print("COMPARISON: CMC Actuation Forces vs Reference")
    print("=" * 60)

    our_file = os.path.join(
        BASE_DIR, "ResultsCMC", "subject01_walk1_Actuation_force.sto"
    )
    ref_file = os.path.join(
        REF_DIR, "ResultsCMC", "subject01_walk1_Actuation_force.sto"
    )

    if not os.path.exists(our_file):
        print("  ERROR: Our CMC output not found.")
        return
    if not os.path.exists(ref_file):
        print("  ERROR: Reference CMC output not found.")
        return

    our_data = load_sto_as_dict(our_file)
    ref_data = load_sto_as_dict(ref_file)

    # Find common columns (excluding time)
    our_cols = set(our_data.keys()) - {"time"}
    ref_cols = set(ref_data.keys()) - {"time"}
    common_cols = sorted(our_cols & ref_cols)

    if not common_cols:
        print("  No common columns found!")
        return

    # Interpolate our data onto reference time points for comparison
    from scipy.interpolate import interp1d

    print(f"\n  Our time range:  {our_data['time'][0]:.4f} - {our_data['time'][-1]:.4f} s")
    print(f"  Ref time range:  {ref_data['time'][0]:.4f} - {ref_data['time'][-1]:.4f} s")
    print(f"  Our data points: {len(our_data['time'])}")
    print(f"  Ref data points: {len(ref_data['time'])}")

    # Use overlapping time range
    t_start = max(our_data["time"][0], ref_data["time"][0])
    t_end = min(our_data["time"][-1], ref_data["time"][-1])
    ref_mask = (ref_data["time"] >= t_start) & (ref_data["time"] <= t_end)
    t_compare = ref_data["time"][ref_mask]

    if len(t_compare) == 0:
        print("  No overlapping time range!")
        return

    print(f"  Comparison range: {t_start:.4f} - {t_end:.4f} s ({len(t_compare)} points)")

    # Select some key muscles to display
    key_muscles = [
        "glut_med1_r", "soleus_r", "tib_ant_r", "med_gas_r",
        "vas_int_r", "rect_fem_r", "bifemlh_r",
        "glut_med1_l", "soleus_l", "tib_ant_l",
    ]

    print(f"\n  {'Actuator':<28s} {'RMSE':>10s} {'nRMSE(%)':>10s} {'Max Ref':>10s}")
    print("  " + "-" * 62)

    all_nrmse = []
    for col in common_cols:
        # Interpolate our result onto reference time
        interp_func = interp1d(
            our_data["time"], our_data[col],
            kind="linear", fill_value="extrapolate"
        )
        our_interp = interp_func(t_compare)
        ref_vals = ref_data[col][ref_mask]

        rmse = np.sqrt(np.mean((our_interp - ref_vals) ** 2))
        max_ref = np.max(np.abs(ref_vals))
        nrmse = (rmse / max_ref * 100) if max_ref > 1e-6 else 0.0
        all_nrmse.append(nrmse)

        if col in key_muscles:
            print(f"  {col:<28s} {rmse:>10.2f} {nrmse:>9.2f}% {max_ref:>10.2f}")

    print("  " + "-" * 62)
    print(f"  {'ALL COLUMNS (median nRMSE)':<28s} {'':>10s} {np.median(all_nrmse):>9.2f}%")
    print(f"  {'ALL COLUMNS (mean nRMSE)':<28s} {'':>10s} {np.mean(all_nrmse):>9.2f}%")
    print(f"  {'ALL COLUMNS (max nRMSE)':<28s} {'':>10s} {np.max(all_nrmse):>9.2f}%")


if __name__ == "__main__":
    print("OpenSim Gait Analysis Pipeline")
    print(f"OpenSim version: {osim.GetVersion()}")
    print(f"Working directory: {BASE_DIR}")

    try:
        run_scale()
        run_ik()
        run_rra()
        run_cmc()
        compare_cmc_results()
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
