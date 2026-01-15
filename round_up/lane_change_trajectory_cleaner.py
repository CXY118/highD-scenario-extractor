"""
Filters lane_change trajectories by velocity range.

Extracts trajectories with speeds within user-defined min/max thresholds
for scenario-specific analysis or outlier removal.
"""

import pandas as pd
import os

# 1. Define file path
data_dir = './output/lane_change_trajectories'                # Directory for saving lane_change vehicle trajectories
output_dir = './output/lane_change_trajectories_filtered'     # Directory for speed-filtered results

os.makedirs(output_dir, exist_ok=True)

# 2. Get all round_up trajectory files and corresponding info files
all_files = os.listdir(data_dir)
trajectory_files = [f for f in all_files if 'round_up_trajectories_' in f and f.endswith('.csv')]
info_files = [f for f in all_files if 'round_up_info_' in f and f.endswith('.csv')]

print(f"Found {len(trajectory_files)} round_up trajectory files")
print(f"Found {len(info_files)} round_up_info files")

# 3. Find matching file pairs
file_pairs = []
for traj_file in trajectory_files:
    scene_num = traj_file.replace('round_up_trajectories_', '').replace('.csv', '')

    expected_info_file = f"round_up_info_{scene_num}.csv"

    if expected_info_file in info_files:
        file_pairs.append({
            'scene_num': scene_num,
            'traj_file': traj_file,
            'info_file': expected_info_file,
            'traj_path': os.path.join(data_dir, traj_file),
            'info_path': os.path.join(data_dir, expected_info_file)
        })
    else:
        print(f"Warning: Trajectory file {traj_file} has no corresponding info file")

print(f"\nFound {len(file_pairs)} matching file pairs")

# 4. Process each file pair
all_summary = []  # Save processing summary for all files

for i, pair in enumerate(file_pairs, 1):
    scene_num = pair['scene_num']
    traj_file = pair['traj_file']
    info_file = pair['info_file']

    print(f"\n{'=' * 80}")
    print(f"[[{i}/{len(file_pairs)}]] Processing Scene {scene_num}:")
    print(f"Trajectory file: {traj_file}")
    print(f"Info file: {info_file}")

    # Read both files
    df_traj = pd.read_csv(pair['traj_path'])
    df_info = pd.read_csv(pair['info_path'])

    print(f"Original trajectory data: {len(df_traj)} rows, {df_traj['id'].nunique()} vehicles")
    print(f"Original info data: {len(df_info)} rows")

    # Check vehicle ID column name in info file
    vehicle_id_col_info = None
    for col in ['vehicle_id', 'id']:
        if col in df_info.columns:
            vehicle_id_col_info = col
            break

    if not vehicle_id_col_info:
        print(f"Error: Vehicle ID column not found in info file, skipping this file pair")
        continue

    # 5. Identify vehicle IDs in trajectory file that do not meet speed requirements
    print(f"\n  Detecting vehicles with abnormal speeds in trajectory file...")

    # Check speed column in trajectory file
    if 'speed' in df_traj.columns:
        speed_col = 'speed'
        print(f"Using speed column: {speed_col}")
    elif 'xVelocity' in df_traj.columns:
        speed_col = 'xVelocity'
        print(f"Using speed column: {speed_col} ")
        df_traj['speed_magnitude'] = abs(df_traj[speed_col])
        speed_col = 'speed_magnitude'
    else:
        print(f"Error: No speed column found in trajectory file, skipping this file pair")
        continue

    # Calculate speed range statistics per vehicle
    vehicle_stats = df_traj.groupby('id').agg({
        speed_col: ['min', 'max', 'count']
    })
    vehicle_stats.columns = ['min_speed', 'max_speed', 'frame_count']
    vehicle_stats = vehicle_stats.reset_index()

    # Identify vehicles with abnormal speeds
    abnormal_vehicles = []
    normal_vehicles = []

    for _, row in vehicle_stats.iterrows():
        vehicle_id = row['id']
        min_speed = row['min_speed']
        max_speed = row['max_speed']

        if min_speed >= 20 and max_speed <= 31:
            normal_vehicles.append(vehicle_id)
        else:
            abnormal_vehicles.append(vehicle_id)
            print(f"Vehicle {vehicle_id}: speed range [{min_speed:.2f}, {max_speed:.2f}] m/s - abnormal")

    print(f"\n Abnormal speed detection completed:")
    print(f" Normal vehicles: {len(normal_vehicles)}")
    print(f" Abnormal vehicles: {len(abnormal_vehicles)}")

    # 6. Clean trajectory data
    print(f"\n  Cleaning trajectory data...")

    # Keep data for normal vehicles only
    df_traj_clear = df_traj[df_traj['id'].isin(normal_vehicles)].copy()

    # Remove temporary speed_magnitude column if exists
    if 'speed_magnitude' in df_traj_clear.columns:
        df_traj_clear = df_traj_clear.drop(columns=['speed_magnitude'])

    # Statistics for cleaning results
    traj_original_count = len(df_traj)
    traj_clear_count = len(df_traj_clear)
    traj_deleted_count = traj_original_count - traj_clear_count

    print(f"Original trajectory rows: {traj_original_count}")
    print(f"Cleaned trajectory rows: {traj_clear_count}")
    print(f"Deleted rows: {traj_deleted_count}")

    # 7. Clean info data (delete records corresponding to abnormal vehicles)
    print(f"\n  Cleaning info data...")

    # Delete records in info file where vehicle ID is in abnormal vehicles list
    df_info_clear = df_info[~df_info[vehicle_id_col_info].isin(abnormal_vehicles)].copy()

    # Statistics for cleaning results
    info_original_count = len(df_info)
    info_clear_count = len(df_info_clear)
    info_deleted_count = info_original_count - info_clear_count

    print(f"Original info rows: {info_original_count}")
    print(f"Cleaned info rows: {info_clear_count}")
    print(f"Deleted rows: {info_deleted_count}")

    # 8. Verify cleaning results
    print(f"\n  Verifying cleaning results:")

    # Verify trajectory file
    if len(df_traj_clear) > 0:
        # Recalculate speed for verification
        if 'speed' in df_traj_clear.columns:
            check_speed_col = 'speed'
        elif 'xVelocity' in df_traj_clear.columns:
            df_traj_clear['check_speed'] = abs(df_traj_clear['xVelocity'])
            check_speed_col = 'check_speed'
        else:
            check_speed_col = speed_col

        # Check speed range after cleaning
        min_speed_after = df_traj_clear[check_speed_col].min()
        max_speed_after = df_traj_clear[check_speed_col].max()

        print(f"Trajectory speed range: [{min_speed_after:.2f}, {max_speed_after:.2f}] m/s")

        if min_speed_after >= 20 and max_speed_after <= 31:
            print(f"Trajectory verification passed")
        else:
            print(f"Trajectory verification warning: abnormal speeds still present")

        # Remove temporary column
        if 'check_speed' in df_traj_clear.columns:
            df_traj_clear = df_traj_clear.drop(columns=['check_speed'])
    else:
        print(f"Warning: Cleaned trajectory data is empty!")

    # Verify info file
    if len(df_info_clear) > 0:
        # Check if there are any abnormal vehicle records in info file
        remaining_abnormal = df_info_clear[df_info_clear[vehicle_id_col_info].isin(abnormal_vehicles)]
        if len(remaining_abnormal) == 0:
            print(f"Info file verification passed: no abnormal vehicle records")
        else:
            print(f"Info file verification warning: still has {len(remaining_abnormal)} abnormal vehicle records")
    else:
        print(f"Warning: Cleaned info data is empty!")

    # 9. Save cleaned files
    print(f"\nSaving cleaned files...")

    # Save trajectory file
    if len(df_traj_clear) > 0:
        traj_output_file = traj_file.replace('.csv', '_clear.csv')
        traj_output_path = os.path.join(output_dir, traj_output_file)
        df_traj_clear.to_csv(traj_output_path, index=False)
        print(f"Trajectory file saved: {traj_output_file}")
    else:
        print(f"Skipping trajectory file save: data is empty")

    if len(df_info_clear) > 0:
        info_output_file = info_file.replace('.csv', '_clear.csv')
        info_output_path = os.path.join(output_dir, info_output_file)
        df_info_clear.to_csv(info_output_path, index=False)
        print(f"Info file saved: {info_output_file}")
    else:
        print(f"Skipping info file save: data is empty")

    # 10. Record processing summary
    all_summary.append({
        'scene_num': scene_num,
        'traj_file': traj_file,
        'info_file': info_file,
        'traj_original': traj_original_count,
        'traj_clear': traj_clear_count,
        'traj_deleted': traj_deleted_count,
        'traj_vehicles_original': df_traj['id'].nunique(),
        'traj_vehicles_clear': len(normal_vehicles),
        'info_original': info_original_count,
        'info_clear': info_clear_count,
        'info_deleted': info_deleted_count,
        'abnormal_vehicles_count': len(abnormal_vehicles),
        'has_traj_data': len(df_traj_clear) > 0,
        'has_info_data': len(df_info_clear) > 0
    })

# 11. Generate summary report
print(f"\n{'=' * 80}")
print("All files processed successfully!")
print(f"Cleaned files saved in: {output_dir}")

# Count output files
output_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
clear_traj_files = [f for f in output_files if 'round_up_trajectories_' in f]
clear_info_files = [f for f in output_files if 'round_up_info_' in f]

print(f"Processing results summary:")
print(f"Processed file pairs: {len(file_pairs)}")
print(f"Generated trajectory files: {len(clear_traj_files)}")
print(f"Generated info files: {len(clear_info_files)}")

print(f"  Processing completed!")
print(f"{'=' * 80}")