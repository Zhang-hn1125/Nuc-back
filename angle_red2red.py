import starfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree

# ===== Parameter Settings =====
STAR_FILE = 'red.star'      # Input .star file
OUTPUT_DIR = 'dihedral_angleresults'  # Output directory
PIXEL_SIZE = 0.272                    # nm/pixel
SEARCHrADIUS_NM = 7
SEARCHrADIUS_PX = SEARCHrADIUS_NM / PIXEL_SIZE

# ===== Load Data =====
print(f"Reading: {STAR_FILE}")
data = starfile.read(STAR_FILE)
data.columns = data.columns.str.strip()

# ===== Check Field Completeness =====
required_fields = ['rlnMicrographName', 'rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ',
                   'rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']
for f in required_fields:
    if f not in data.columns:
        raise ValueError(f"Missing required field: {f}")

# ===== Create Output Directory =====
os.makedirs(OUTPUT_DIR, exist_ok=True)
summary_stats = []
all_angles = []

# ===== Process by Tomogram Grouping =====
grouped = data.groupby('rlnMicrographName')
for name, group in grouped:
    print(f"\nProcessing: {name}  ({len(group)} particles)")

    coords = group[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].to_numpy()
    eulers = group[['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']].to_numpy()
    indices = group.index.to_numpy()

    # calculate vector
    normals = R.from_euler('zyz', eulers, degrees=True).apply(np.array([0, 0, 1]))
    tree = cKDTree(coords)

    records = []
    angle_list = []

    for i, (coord, n1) in enumerate(zip(coords, normals)):
        neighbors = tree.query_ball_point(coord, SEARCHrADIUS_PX)
        neighbors = [j for j in neighbors if j != i]
        if not neighbors:
            continue

        dists = [np.linalg.norm(coords[j] - coord) for j in neighbors]
        j_min = neighbors[np.argmin(dists)]
        n2 = normals[j_min]
        distance = dists[np.argmin(dists)]

        cos_theta = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
        angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        angle_deg = min(angle_deg, 180 - angle_deg)

        angle_list.append(angle_deg)
        all_angles.append(angle_deg)

        records.append({
            'Index1': indices[i],
            'Index2': indices[j_min],
            'Distance_nm': distance * PIXEL_SIZE,
            'Angle_deg': angle_deg
        })

    if records:
        basename = os.path.basename(name).replace('.tomostar', '').replace('.mrc', '')
        df_out = pd.DataFrame(records)
        csv_path = os.path.join(OUTPUT_DIR, f'{basename}_angles.csv')
        df_out.to_csv(csv_path, index=False)
        print(f" → Saved: {csv_path}")

        mean_angle = np.mean(angle_list)
        std_angle = np.std(angle_list)
        print(f" → Mean: {mean_angle:.2f}°, Std: {std_angle:.2f}°, N: {len(angle_list)}")

        summary_stats.append({
            'Tomogram': basename,
            'Count': len(angle_list),
            'Mean_angle': mean_angle,
            'Std_angle': std_angle
        })
    else:
        print(" → No valid neighbors found within 10 nm.")

# ===== Save summary_statistics.csv =====
if summary_stats:
    summary_df = pd.DataFrame(summary_stats)
    summary_path = os.path.join(OUTPUT_DIR, 'summary_statistics.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary written to: {summary_path}")

# ===== Plot Global Angle Distribution =====
if all_angles:
    angles_array = np.array(all_angles)
    mean_angle = np.mean(angles_array)
    std_angle = np.std(angles_array)

    plt.figure(figsize=(10, 6))
    n, bins, _ = plt.hist(angles_array, bins=np.linspace(0, 90, 46), color='darkorange',
                          edgecolor='black', alpha=0.75, label='Angle Distribution', density=True)

    plt.axvline(mean_angle, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_angle:.1f}°')
    plt.fill_betweenx([0, max(n)], mean_angle - std_angle, mean_angle + std_angle,
                      color='red', alpha=0.2, label=f'±1 Std = {std_angle:.1f}°')

    peak_bin_index = np.argmax(n)
    peak_center = 0.5 * (bins[peak_bin_index] + bins[peak_bin_index + 1])
    plt.annotate(f'Peak: {peak_center:.1f}°',
                 xy=(peak_center, max(n)),
                 xytext=(peak_center + 5, max(n) * 0.8),
                 arrowprops=dict(facecolor='black', arrowstyle='->'))

    plt.xlabel('Dihedral Angle (degrees)')
    plt.ylabel('Frequency')
    plt.title('Global Dihedral Angle Distribution')
    plt.legend()
    plt.xlim(0, 90)
    plt.tight_layout()
    plt.show()


    global_plot_path = os.path.join(OUTPUT_DIR, 'global_angle_histogram_enhanced.png')
    plt.savefig(global_plot_path)
    plt.close()

    print(f"\nGlobal histogram saved: {global_plot_path}")
all_angles_df = pd.DataFrame({'Angle_deg': all_angles})
all_angles_path = os.path.join(OUTPUT_DIR, 'all_dihedral_angles.csv')
all_angles_df.to_csv(all_angles_path, index=False)
print(f"✅ All angles saved to: {all_angles_path}")
# ===== Calculate the Proportion of Angles Less Than 30° =====
angles_array = np.array(all_angles)
threshold =30
num_below = np.sum(angles_array < threshold)
percent_below = (num_below / len(angles_array)) * 100

print(f"\n📊 Number of angles < {threshold}°: {num_below} / {len(angles_array)}")
print(f"✅ Proportion: {percent_below:.2f}%")
