"""
Filters cut-in trajectories by velocity range.

Extracts trajectories with speeds within user-defined min/max thresholds
for scenario-specific analysis or outlier removal.
"""

import pandas as pd
import os

# 1. Define file path
data_dir = "./output/cutin_trajectories"                # Directory containing extracted cut-in trajectories
output_dir = "./output/cutin_trajectories_filtered"     # Directory for speed-filtered results
os.makedirs(output_dir, exist_ok=True)

# Configurable parameters
MIN_SPEED = 20  # Minimum speed threshold in m/s (adjustable)
MAX_SPEED = 35  # Maximum speed threshold in m/s (adjustable)

# 2. Get all files
cut_in_files = [f for f in os.listdir(data_dir) if 'cut_in_trajectories_' in f and f.endswith('.csv')]
print(f"Found {len(cut_in_files)} cut-in trajectory files")

# 3. Process each file
for i, file_name in enumerate(cut_in_files, 1):
    print(f"\n[{i}/{len(cut_in_files)}] Processing file: {file_name}")

    file_path = os.path.join(data_dir, file_name)
    df = pd.read_csv(file_path)
    print(f"Original data: {len(df)} rows, {df['id'].nunique()} vehicles")

    # 4. Filter vehicles by speed range
    # Adjust MIN_SPEED and MAX_SPEED as needed for your analysis
    df['speed_abs'] = df['xVelocity'].abs()
    slow_ids = df[df['speed_abs'] < MIN_SPEED]['id'].unique().tolist()
    fast_ids = df[df['speed_abs'] > MAX_SPEED]['id'].unique().tolist()

    # Merge IDs to remove
    ids_to_remove = list(set(slow_ids + fast_ids))

    # Remove all records for these IDs
    df_filtered = df[~df['id'].isin(ids_to_remove)].copy()

    print(f" Removed {len(slow_ids)} low-speed vehicles")
    print(f" Removed {len(fast_ids)} high-speed vehicles")
    print(f" Total removed: {len(ids_to_remove)} abnormal-speed vehicles")
    print(f" After filtering: {len(df_filtered)} rows, {df_filtered['id'].nunique()} vehicles")

    # 5. Validation
    if len(df_filtered) > 0:
        df_filtered['speed_abs'] = df_filtered['xVelocity'].abs()
        min_speed_abs = df_filtered['speed_abs'].min()
        max_speed_abs = df_filtered['speed_abs'].max()

        if min_speed_abs >= MIN_SPEED and max_speed_abs <= MAX_SPEED:
            print(f" Validation passed: speed range = [{min_speed_abs:.2f}, {max_speed_abs:.2f}] m/s")
        else:
            print(f" Validation failed: speed range = [{min_speed_abs:.2f}, {max_speed_abs:.2f}] m/s")
    else:
        print("Warning: Data empty after filtering!")

    # 6. Save filtered file
    output_file = file_name.replace('.csv', '_filtered.csv')
    output_path = os.path.join(output_dir, output_file)

    df_filtered.to_csv(output_path, index=False)
    print(f"Saved: {output_file}")

print(f"\nAll files processed!")
print(f"Filtered files saved in: {output_dir}")