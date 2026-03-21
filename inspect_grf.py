"""Inspect GRF data to identify heel contact timing."""
import numpy as np

data = np.loadtxt("subject01_walk1_grf.mot", skiprows=7)
time = data[:, 0]
fx_r = data[:, 1]   # right anterior-posterior
fy_r = data[:, 2]   # right vertical
fz_r = data[:, 3]   # right medial-lateral
fx_l = data[:, 7]   # left anterior-posterior
fy_l = data[:, 8]   # left vertical
fz_l = data[:, 9]   # left medial-lateral

threshold = 20.0

print("ALL gait events across the full trial:")
print(f"  Trial time range: {time[0]:.4f} - {time[-1]:.4f}s")
print()

for i in range(1, len(time)):
    # Right foot
    if fy_r[i - 1] < threshold and fy_r[i] >= threshold:
        print(f"  RIGHT heel contact at t={time[i]:.4f}s (Fy={fy_r[i]:.1f} N)")
    if fy_r[i - 1] >= threshold and fy_r[i] < threshold:
        print(f"  RIGHT toe-off     at t={time[i]:.4f}s (Fy={fy_r[i]:.1f} N)")
    # Left foot
    if fy_l[i - 1] < threshold and fy_l[i] >= threshold:
        print(f"  LEFT  heel contact at t={time[i]:.4f}s (Fy={fy_l[i]:.1f} N)")
    if fy_l[i - 1] >= threshold and fy_l[i] < threshold:
        print(f"  LEFT  toe-off     at t={time[i]:.4f}s (Fy={fy_l[i]:.1f} N)")

# Show left foot forces in CMC window
print("\nLeft foot GRF in CMC window (0.8-1.2s):")
print(f"{'time':>8s} {'Fy_L':>10s} {'Fx_L':>10s} {'Fz_L':>10s}")
print("-" * 42)
mask = (time >= 0.8) & (time <= 1.2)
for i in np.where(mask)[0][::30]:
    print(f"{time[i]:8.4f} {fy_l[i]:10.2f} {fx_l[i]:10.2f} {fz_l[i]:10.2f}")
