"""
Slip Simulation Script

Simulates what happens when friction is suddenly reduced during walking.
Uses CMC-derived muscle controls (frozen activations) and modified GRF
to run a forward dynamics simulation.

Inputs:
    - CMC states file:   provides initial conditions (joint angles, velocities, muscle states)
    - CMC controls file: provides frozen muscle activations
    - GRF file:          ground reaction forces to be modified

Usage:
    python slip_sim.py --mu_slip 0.1 --foot right --slip_time 0.9 --duration 0.05
"""

import argparse
import os
import shutil

import numpy as np
import opensim as osim


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="Slip simulation")
    parser.add_argument(
        "--cmc_states",
        default=os.path.join(BASE_DIR, "ResultsCMC", "subject01_walk1_states.sto"),
        help="CMC states file (.sto)",
    )
    parser.add_argument(
        "--cmc_controls",
        default=os.path.join(
            BASE_DIR, "ResultsCMC", "subject01_walk1_controls.xml"
        ),
        help="CMC controls file (.xml)",
    )
    parser.add_argument(
        "--grf_file",
        default=os.path.join(BASE_DIR, "subject01_walk1_grf.mot"),
        help="Ground reaction force file (.mot)",
    )
    parser.add_argument(
        "--grf_xml",
        default=os.path.join(BASE_DIR, "subject01_walk1_grf.xml"),
        help="External loads XML file",
    )
    parser.add_argument(
        "--model_file",
        default=os.path.join(BASE_DIR, "subject01_simbody_adjusted.osim"),
        help="OpenSim model file (.osim)",
    )
    parser.add_argument(
        "--actuator_file",
        default=os.path.join(BASE_DIR, "gait2354_CMC_Actuators.xml"),
        help="CMC actuators XML file",
    )
    parser.add_argument(
        "--mu_slip",
        type=float,
        default=0.1,
        help="Reduced friction coefficient during slip (default: 0.1)",
    )
    parser.add_argument(
        "--foot",
        choices=["right", "left"],
        default="right",
        help="Which foot slips (default: right)",
    )
    parser.add_argument(
        "--slip_time",
        type=float,
        default=None,
        help="Time of slip onset in seconds. If not set, auto-detects heel contact.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.05,
        help="Simulation duration after slip onset in seconds (default: 0.05)",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(BASE_DIR, "ResultsSlip"),
        help="Output directory",
    )
    return parser.parse_args()


def load_grf(grf_file):
    """Load the GRF .mot file into time array and data columns."""
    # Count header lines
    with open(grf_file, "r") as f:
        for i, line in enumerate(f):
            if line.strip() == "endheader":
                header_end = i + 1
                break

    # Read column names (line after endheader)
    with open(grf_file, "r") as f:
        lines = f.readlines()
    col_names = lines[header_end].strip().split("\t")

    data = np.loadtxt(grf_file, skiprows=header_end + 1)
    return col_names, data


def detect_heel_contact(col_names, data, foot, t_min=0.0, t_max=np.inf):
    """Detect heel contact as the first frame where vertical GRF exceeds threshold."""
    time = data[:, 0]
    if foot == "right":
        fy_col = col_names.index("ground_force_vy")
    else:
        fy_col = col_names.index("1_ground_force_vy")

    fy = data[:, fy_col]
    threshold = 20.0

    for i in range(1, len(time)):
        if time[i] < t_min or time[i] > t_max:
            continue
        if fy[i - 1] < threshold and fy[i] >= threshold:
            return time[i]

    return None


def create_modified_grf(col_names, data, foot, slip_time, mu_slip, output_file):
    """
    Create a modified GRF file where horizontal forces are capped by
    Coulomb friction: |F_horizontal| <= mu_slip * F_vertical,
    from slip_time onward.
    """
    time = data[:, 0]

    if foot == "right":
        fx_col = col_names.index("ground_force_vx")
        fy_col = col_names.index("ground_force_vy")
        fz_col = col_names.index("ground_force_vz")
    else:
        fx_col = col_names.index("1_ground_force_vx")
        fy_col = col_names.index("1_ground_force_vy")
        fz_col = col_names.index("1_ground_force_vz")

    modified = data.copy()

    n_capped = 0
    for i in range(len(time)):
        if time[i] < slip_time:
            continue

        fx = modified[i, fx_col]
        fy = modified[i, fy_col]
        fz = modified[i, fz_col]

        f_horiz = np.sqrt(fx**2 + fz**2)
        f_max = mu_slip * abs(fy)

        if f_horiz > f_max and f_horiz > 1e-10:
            scale = f_max / f_horiz
            modified[i, fx_col] = fx * scale
            modified[i, fz_col] = fz * scale
            n_capped += 1

    # Write the modified .mot file
    with open(output_file, "w") as f:
        f.write("Modified GRF with reduced friction\n")
        f.write("version=1\n")
        f.write(f"nRows={len(time)}\n")
        f.write(f"nColumns={len(col_names)}\n")
        f.write("inDegrees=yes\n")
        f.write("endheader\n")
        f.write("\t".join(col_names) + "\n")
        for row in modified:
            f.write("\t".join(f"{v:.10f}" for v in row) + "\n")

    return n_capped


def create_modified_grf_xml(original_xml, modified_mot, output_xml):
    """Create external loads XML pointing to the modified GRF .mot file."""
    with open(original_xml, "r") as f:
        content = f.read()

    # Replace the datafile reference
    original_mot = "subject01_walk1_grf.mot"
    content = content.replace(original_mot, modified_mot)

    with open(output_xml, "w") as f:
        f.write(content)


def run_forward_simulation(args, slip_time, use_correction=False):
    """Run forward dynamics with frozen CMC controls and modified GRF.

    If use_correction=True, adds a CorrectionController (PD feedback)
    that tracks the CMC states trajectory, preventing open-loop drift.
    """
    output_dir = args.output_dir
    t_final = slip_time + args.duration

    # Create the ForwardTool
    tool = osim.ForwardTool()
    tool.setName("slip_simulation")
    tool.setModelFilename(args.model_file)

    # Force set (append CMC actuators)
    tool.setReplaceForceSet(False)
    force_set = osim.ArrayStr()
    force_set.append(args.actuator_file)
    tool.setForceSetFiles(force_set)

    # Time range
    tool.setInitialTime(slip_time)
    tool.setFinalTime(t_final)

    # Results
    tool.setResultsDir(output_dir)
    tool.setOutputPrecision(20)

    # Initial states from CMC
    tool.setStatesFileName(args.cmc_states)

    # Integrator settings
    tool.setMaximumNumberOfSteps(30000)
    tool.setMaxDT(1.0)
    tool.setMinDT(1e-5)
    tool.setErrorTolerance(5e-5)

    # External loads (modified GRF)
    modified_grf_xml = os.path.join(output_dir, "slip_grf.xml")
    tool.setExternalLoadsFileName(modified_grf_xml)

    # Controllers: use CMC controls (frozen muscle activations)
    # We need to serialize and set up via XML since the Python API for
    # ControlSetController is limited. Write a setup XML file.
    setup_xml = os.path.join(output_dir, "slip_forward_setup.xml")
    tool.printToXML(setup_xml)

    # Now modify the XML to add the ControlSetController
    with open(setup_xml, "r") as f:
        content = f.read()

    # Replace the empty ControllerSet that ForwardTool serializes with our
    # populated one. The empty block looks like:
    #   <ControllerSet name="Controllers">
    #       <objects />
    #       <groups />
    #   </ControllerSet>
    import re

    correction_xml = ""
    if use_correction:
        correction_xml = """
				<CorrectionController name="">
					<actuator_list> </actuator_list>
					<isDisabled> false </isDisabled>
					<kp> 16.0 </kp>
					<kv> 8.0 </kv>
				</CorrectionController>"""

    controller_xml = f"""<ControllerSet name="Controllers">
			<objects>
				<ControlSetController name="">
					<actuator_list> </actuator_list>
					<isDisabled> false </isDisabled>
					<controls_file> {args.cmc_controls} </controls_file>
				</ControlSetController>{correction_xml}
			</objects>
			<groups />
		</ControllerSet>"""

    # Match the empty ControllerSet block (with self-closing or separate tags)
    empty_pattern = r'<ControllerSet name="Controllers">\s*<objects\s*/>\s*<groups\s*/>\s*</ControllerSet>'
    # Use a lambda to avoid re interpreting backslashes in the replacement
    content = re.sub(empty_pattern, lambda m: controller_xml, content)

    with open(setup_xml, "w") as f:
        f.write(content)

    # Re-load and run from the modified XML
    tool2 = osim.ForwardTool(setup_xml)
    tool2.run()

    return setup_xml


def compare_kinematics(normal_states_file, slip_states_file, slip_time, duration):
    """Compare normal vs slip kinematics using states files."""
    if not os.path.exists(normal_states_file) or not os.path.exists(slip_states_file):
        print("  Cannot compare - missing states files")
        print(f"    Normal: {normal_states_file} -> exists={os.path.exists(normal_states_file)}")
        print(f"    Slip:   {slip_states_file} -> exists={os.path.exists(slip_states_file)}")
        return

    normal = osim.Storage(normal_states_file)
    slip = osim.Storage(slip_states_file)

    # Key coordinates: short name -> column path in states file
    coords = {
        "pelvis_tx": "/jointset/ground_pelvis/pelvis_tx/value",
        "pelvis_ty": "/jointset/ground_pelvis/pelvis_ty/value",
        "pelvis_tz": "/jointset/ground_pelvis/pelvis_tz/value",
        "pelvis_tilt": "/jointset/ground_pelvis/pelvis_tilt/value",
        "pelvis_list": "/jointset/ground_pelvis/pelvis_list/value",
        "pelvis_rotation": "/jointset/ground_pelvis/pelvis_rotation/value",
        "hip_flexion_r": "/jointset/hip_r/hip_flexion_r/value",
        "knee_angle_r": "/jointset/knee_r/knee_angle_r/value",
        "ankle_angle_r": "/jointset/ankle_r/ankle_angle_r/value",
        "hip_flexion_l": "/jointset/hip_l/hip_flexion_l/value",
        "knee_angle_l": "/jointset/knee_l/knee_angle_l/value",
        "ankle_angle_l": "/jointset/ankle_l/ankle_angle_l/value",
    }

    print(f"\n  {'Coordinate':<22s} {'Normal(end)':>12s} {'Slip(end)':>12s} {'Diff':>10s} {'Units':>8s}")
    print("  " + "-" * 68)

    for short_name, col_path in coords.items():
        normal_col = osim.ArrayDouble()
        slip_col = osim.ArrayDouble()

        try:
            normal.getDataColumn(col_path, normal_col)
            slip.getDataColumn(col_path, slip_col)
        except Exception:
            continue

        normal_val = normal_col.get(normal_col.getSize() - 1)
        slip_val = slip_col.get(slip_col.getSize() - 1)
        diff = slip_val - normal_val

        units = "m" if "pelvis_t" in short_name else "deg"
        print(f"  {short_name:<22s} {normal_val:>12.4f} {slip_val:>12.4f} {diff:>10.4f} {units:>8s}")


def main():
    args = parse_args()
    os.chdir(BASE_DIR)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("SLIP SIMULATION")
    print("=" * 60)
    print(f"  Friction coefficient: {args.mu_slip}")
    print(f"  Slipping foot:       {args.foot}")
    print(f"  Simulation duration: {args.duration * 1000:.0f} ms")
    print(f"  Model:               {os.path.basename(args.model_file)}")
    print(f"  Output:              {args.output_dir}")

    # Step 1: Load GRF data
    print("\n--- Step 1: Loading GRF data ---")
    col_names, grf_data = load_grf(args.grf_file)
    print(f"  Loaded {len(grf_data)} time points")

    # Step 2: Determine slip onset time
    print("\n--- Step 2: Determining slip onset time ---")
    if args.slip_time is not None:
        slip_time = args.slip_time
        print(f"  Using user-specified slip time: {slip_time:.4f}s")
    else:
        # Auto-detect: use the CMC window for valid times
        slip_time = detect_heel_contact(col_names, grf_data, args.foot, 0.8, 1.15)
        if slip_time is None:
            # Foot is already on ground; pick midpoint of CMC window
            slip_time = 0.9
            print(f"  No heel contact in CMC window; using default: {slip_time:.4f}s")
        else:
            print(f"  Auto-detected heel contact: {slip_time:.4f}s")

    t_final = slip_time + args.duration
    print(f"  Simulation window: {slip_time:.4f}s to {t_final:.4f}s")

    # Step 3: Create modified GRF
    print("\n--- Step 3: Creating modified GRF ---")
    modified_mot = os.path.join(args.output_dir, "slip_grf.mot")
    n_capped = create_modified_grf(
        col_names, grf_data, args.foot, slip_time, args.mu_slip, modified_mot
    )
    print(f"  Capped {n_capped} time frames to mu={args.mu_slip}")
    print(f"  Written: {modified_mot}")

    # Create corresponding external loads XML
    modified_xml = os.path.join(args.output_dir, "slip_grf.xml")
    create_modified_grf_xml(args.grf_xml, modified_mot, modified_xml)
    print(f"  Written: {modified_xml}")

    # Show force comparison at slip onset
    time_col = grf_data[:, 0]
    slip_idx = np.argmin(np.abs(time_col - slip_time))
    if args.foot == "right":
        fx_col = col_names.index("ground_force_vx")
        fy_col = col_names.index("ground_force_vy")
        fz_col = col_names.index("ground_force_vz")
    else:
        fx_col = col_names.index("1_ground_force_vx")
        fy_col = col_names.index("1_ground_force_vy")
        fz_col = col_names.index("1_ground_force_vz")

    orig_fx = grf_data[slip_idx, fx_col]
    orig_fy = grf_data[slip_idx, fy_col]
    orig_fz = grf_data[slip_idx, fz_col]
    orig_horiz = np.sqrt(orig_fx**2 + orig_fz**2)
    max_horiz = args.mu_slip * abs(orig_fy)

    print(f"\n  At slip onset (t={time_col[slip_idx]:.4f}s):")
    print(f"    Original: Fx={orig_fx:.1f}N, Fy={orig_fy:.1f}N, Fz={orig_fz:.1f}N")
    print(f"    |F_horiz| = {orig_horiz:.1f}N")
    print(f"    mu*Fy cap  = {max_horiz:.1f}N")
    if orig_horiz > max_horiz:
        print(f"    -> Friction EXCEEDED, forces will be capped")
    else:
        print(f"    -> Friction NOT exceeded at this instant")

    # Step 4: Run forward simulation with slip
    print("\n--- Step 4: Running forward simulation (SLIP) ---")
    run_forward_simulation(args, slip_time)

    # Step 5: Run forward simulation without slip (normal, for comparison)
    print("\n--- Step 5: Running forward simulation (NORMAL, for comparison) ---")
    normal_output_dir = os.path.join(args.output_dir, "normal")
    os.makedirs(normal_output_dir, exist_ok=True)

    normal_args = argparse.Namespace(**vars(args))
    normal_args.output_dir = normal_output_dir

    # Create normal GRF XML (unchanged forces)
    normal_grf_xml = os.path.join(normal_output_dir, "slip_grf.xml")
    create_modified_grf_xml(args.grf_xml, args.grf_file, normal_grf_xml)

    normal_args_for_fwd = argparse.Namespace(**vars(args))
    normal_args_for_fwd.output_dir = normal_output_dir
    normal_args_for_fwd.grf_xml = normal_grf_xml

    # Point the external loads to the original GRF
    # Enable CorrectionController so normal sim tracks CMC kinematics
    run_forward_simulation(normal_args_for_fwd, slip_time, use_correction=True)

    # Step 6: Compare kinematics
    print("\n--- Step 6: Comparing kinematics ---")
    slip_states = os.path.join(
        args.output_dir, "slip_simulation_states_degrees.mot"
    )
    normal_states = os.path.join(
        normal_output_dir, "slip_simulation_states_degrees.mot"
    )
    compare_kinematics(normal_states, slip_states, slip_time, args.duration)

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print(f"  Slip results:   {args.output_dir}")
    print(f"  Normal results: {normal_output_dir}")


if __name__ == "__main__":
    main()
