import starfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree

# ========= Parameters =========
REF_STAR = 'red.star'
QUERY_STAR = 'gray.star'
OUTPUT_DIR = 'compare_starresults'
PIXEL_SIZE = 0.272  # nm/pixel
SEARCHrADIUS_NM = 10
SEARCHrADIUS_PX = SEARCHrADIUS_NM / PIXEL_SIZE

# ========= Load Files =========
ref = starfile.read(REF_STAR)
query = starfile.read(QUERY_STAR)
ref.columns = ref.columns.str.strip()
query.columns = query.columns.str.strip()

# ========= Check Required Fields =========
required_fields = ['rlnMicrographName', 'rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ',
                   'rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']
for f in required_fields:
    if f not in ref.columns or f not in query.columns:
        raise ValueError(f"Missing field: {f}")

# ========= Match by Tomogram Grouping =========
os.makedirs(OUTPUT_DIR, exist_ok=True)
groupedref = ref.groupby('rlnMicrographName')
grouped_query = query.groupby('rlnMicrographName')

common_tomos = sorted(set(groupedref.groups.keys()) & set(grouped_query.groups.keys()))
print(f"Found {len(common_tomos)} common tomograms in total")

summary_stats = []
all_angles = []

for tomo in common_tomos:
    ref_group = groupedref.get_group(tomo)
    query_group = grouped_query.get_group(tomo)
    print(f"\n🧩 Processing tomogram: {tomo}  →  Ref: {len(ref_group)}, Query: {len(query_group)}")

    coordsref = ref_group[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].to_numpy()
    eulersref = ref_group[['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']].to_numpy()
    normalsref = R.from_euler('zyz', eulersref, degrees=True).apply([0, 0, 1])
    indicesref = ref_group.index.to_numpy()

    coords_query = query_group[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].to_numpy()
    eulers_query = query_group[['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']].to_numpy()
    normals_query = R.from_euler('zyz', eulers_query, degrees=True).apply([0, 0, 1])
    indices_query = query_group.index.to_numpy()

    tree = cKDTree(coordsref)
    records = []
    angles_this = []

    for i, (coord_q, n_q) in enumerate(zip(coords_query, normals_query)):
        neighbors = tree.query_ball_point(coord_q, SEARCHrADIUS_PX)
        if not neighbors:
            continue

        dists = [np.linalg.norm(coordsref[j] - coord_q) for j in neighbors]
        j_min = neighbors[np.argmin(dists)]
        nr = normalsref[j_min]
        distance = dists[np.argmin(dists)]

        cos_theta = np.dot(n_q, nr) / (np.linalg.norm(n_q) * np.linalg.norm(nr))
        angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        angle_deg = min(angle_deg, 180 - angle_deg)

        records.append({
            'Query_Index': indices_query[i],
            'Ref_Index': indicesref[j_min],
            'Distance_nm': distance * PIXEL_SIZE,
            'Angle_deg': angle_deg
        })
        angles_this.append(angle_deg)
        all_angles.append(angle_deg)

    # Save matching results for each tomogram
    if records:
        basename = os.path.basename(tomo).replace('.tomostar', '').replace('.mrc', '')
        df_out = pd.DataFrame(records)
        csv_path = os.path.join(OUTPUT_DIR, f'{basename}_match.csv')
        df_out.to_csv(csv_path, index=False)

        mean_angle = np.mean(angles_this)
        std_angle = np.std(angles_this)
        summary_stats.append({
            'Tomogram': basename,
            'Pair_Count': len(angles_this),
            'Mean_angle': mean_angle,
            'Std_angle': std_angle
        })


if summary_stats:
    df_summary = pd.DataFrame(summary_stats)
    summary_csv = os.path.join(OUTPUT_DIR, 'summary_statistics.csv')
    df_summary.to_csv(summary_csv, index=False)
    print(f"\n✅ Summary table saved to: {summary_csv}")


if all_angles:
    angles_array = np.array(all_angles)
    mean_angle = np.mean(angles_array)
    std_angle = np.std(angles_array)

    plt.figure(figsize=(10, 6))
    n, bins, _ = plt.hist(angles_array, bins=np.linspace(0, 90, 46), color='seagreen',
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
    plt.title('Global Angle Distribution: Query vs Reference')
    plt.legend()
    plt.xlim(0, 90)
    plt.tight_layout()
    plt.show()


    global_plot_path = os.path.join(OUTPUT_DIR, 'global_query_vsreference_histogram.png')
    plt.savefig(global_plot_path)
    plt.close()
    print(f"📊 Global plot saved: {global_plot_path}")
else:
    print("⚠️ No matching angles found. Check if the query particles are within the reference set.")
all_angles_df = pd.DataFrame({'Angle_deg': all_angles})
all_angles_path = os.path.join(OUTPUT_DIR, 'all_dihedral_angles.csv')
all_angles_df.to_csv(all_angles_path, index=False)
print(f"✅ All angles saved to: {all_angles_path}")
# ===== Calculate the proportion of angles less than 30° =====
angles_array = np.array(all_angles)
threshold =30
num_below = np.sum(angles_array < threshold)
percent_below = (num_below / len(angles_array)) * 100

print(f"\n📊 Number of angles < {threshold}°: {num_below} / {len(angles_array)}")
print(f"✅ Proportion: {percent_below:.2f}%")