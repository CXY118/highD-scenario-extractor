"""
Initializes cut-in scenarios for highway-env.

Integrates vehicle pair states and computes relative parameters
to align with highway-env observation space.
"""

import pandas as pd
import numpy as np
import os

# 1. Set base paths
base_data_dir = './data/highD-dataset-v1.0'                  # HighD dataset directory
cutin_initial_dir = './output/cutin_initial_states'          # Initial state information
victim_initial_dir = './output/victim_initial_states'        # Initial state information
message_dir = './output/aligned_states'                      # Aligned state information
merged_output = './output/merged_all_scenarios.csv'          # Merged all scenarios

os.makedirs(message_dir, exist_ok=True)

# Record processing statistics
summary_stats = []

# 2. Process all 60 recording files
for file_num in range(1, 61):
    file_num_str = f"{file_num:02d}"
    print(f"\nProcessing recording {file_num_str}...")

    # Build file paths
    file1_path = os.path.join(cutin_initial_dir, f'cut_in_trajectories_{file_num_str}_initial_state.csv')
    file2_path = os.path.join(victim_initial_dir, f'cutted_in_trajectories_{file_num_str}.csv')
    tracks_meta_path = os.path.join(base_data_dir, f'{file_num_str}_tracksMeta.csv')
    output_path = os.path.join(message_dir, f"message_cutin_{file_num_str}.csv")

    # Check files
    if not all(os.path.exists(f) for f in [file1_path, file2_path, tracks_meta_path]):
        print(f"Files missing, skipping recording {file_num_str}")
        continue

    # Read files
    try:
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)
        tracks_meta = pd.read_csv(tracks_meta_path)
    except Exception as e:
        print(f"Failed to read: {e}")
        continue

    # Find common frames
    common_frames = set(df1['frame']).intersection(set(df2['frame']))
    if len(common_frames) == 0:
        print(f"No common frames, skipping")
        continue

    # Extract data for common frames
    df1_common = df1[df1['frame'].isin(common_frames)].sort_values('frame').reset_index(drop=True)
    df2_common = df2[df2['frame'].isin(common_frames)].sort_values('frame').reset_index(drop=True)

    # Get driving direction
    unique_ids = df1_common['id'].unique()
    id_to_direction = {}
    for vehicle_id in unique_ids:
        meta_row = tracks_meta[tracks_meta['id'] == vehicle_id]
        id_to_direction[vehicle_id] = meta_row.iloc[0]['drivingDirection'] if len(meta_row) > 0 else None

    # Align and compute differences
    aligned_data = []
    min_length = min(len(df1_common), len(df2_common))

    for idx in range(min_length):
        row1, row2 = df1_common.iloc[idx], df2_common.iloc[idx]

        if row1['frame'] != row2['frame']:
            continue

        x_diff = abs(row1['x'] - row2['x'])
        y_diff = row1['y'] - row2['y']
        xVelocity_diff = row1['xVelocity'] - row2['xVelocity']
        laneId_diff = row1['laneId'] - row2['laneId']

        direction = id_to_direction.get(row1['id'])
        if direction == 1:
            adjusted_laneId_diff = -laneId_diff
            adjustment_note = "reverse direction (direction=1)"
        elif direction == 2:
            adjusted_laneId_diff = laneId_diff
            adjustment_note = "forward direction (direction=2)"
        else:
            adjusted_laneId_diff = laneId_diff
            adjustment_note = "unknown direction"

        aligned_data.append({
            'recording_id': file_num_str,
            'frame': row1['frame'],
            'cut_in_vehicle_id': row1['id'],
            'target_vehicle_id': row2['id'],
            'x_cut_in': row1['x'], 'y_cut_in': row1['y'],
            'x_target': row2['x'], 'y_target': row2['y'],
            'x_diff_abs': x_diff, 'y_diff': y_diff,
            'xVelocity_cut_in': row1['xVelocity'], 'xVelocity_target': row2['xVelocity'],
            'xVelocity_diff': xVelocity_diff,
            'laneId_cut_in': row1['laneId'], 'laneId_target': row2['laneId'],
            'laneId_diff': laneId_diff, 'adjusted_laneId_diff': adjusted_laneId_diff,
            'driving_direction': direction, 'adjustment_note': adjustment_note
        })

    # Save results
    if aligned_data:
        aligned_df = pd.DataFrame(aligned_data)
        aligned_df.to_csv(output_path, index=False)

        summary_stats.append({
            'recording': file_num_str,
            'status': 'success',
            'aligned_rows': len(aligned_df),
            'x_diff_mean': aligned_df['x_diff_abs'].mean()
        })
        print(f"Saved: {output_path} ({len(aligned_df)} rows)")
    else:
        print(f"No aligned data")

# 3. Merge all message files
print(f"\nMerging all CSV files in message directory...")
all_files = [f for f in os.listdir(message_dir) if f.endswith('.csv') and f.startswith('message_cutin_')]
all_files.sort()

if all_files:
    dataframes = []
    for filename in all_files:
        file_path = os.path.join(message_dir, filename)
        df = pd.read_csv(file_path)
        dataframes.append(df)
        print(f"Read: {filename} ({len(df)} rows)")

    merged = pd.concat(dataframes, ignore_index=True)
    merged.to_csv(merged_output, index=False)

    print(f"\nMerge complete! Total rows: {len(merged)}")
    print(f"Saved to: {merged_output}")
else:
    print(f"No CSV files found in message directory")

# 4. Generate processing summary
if summary_stats:
    summary_df = pd.DataFrame(summary_stats)
    summary_path = os.path.join(message_dir, "processing_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    success_count = len([s for s in summary_stats if s['status'] == 'success'])
    print(f"\nProcessing complete: {success_count} successful, {len(summary_stats) - success_count} failed")
    print(f"Summary saved: {summary_path}")