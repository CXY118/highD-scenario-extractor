"""
Extracts vehicle trajectories performing cut-in maneuvers from driving datasets.

Trajectories include the complete process: approaching, lane-changing alongside,
and overtaking the victim vehicle, with specified pre- and post-maneuver frames.
"""

import pandas as pd
import numpy as np
import os

# 1. Set base data directories
base_data_dir = './data/highD-dataset-v1.0'      # HighD dataset directory
save_dir = './output/cutin_trajectories'         # Directory for saving cut-in vehicle trajectories
os.makedirs(save_dir, exist_ok=True)

# Cut-in extraction parameters
PRE_FRAMES = 25      # Frames before cut-in
POST_FRAMES = 25     # Frames after cut-in

# 2. Process all recording files (01-60)
for file_num in range(1, 61):

    file_num_str = f"{file_num:02d}"

    # Build file paths
    tracksMeta_path = os.path.join(base_data_dir, f"{file_num_str}_tracksMeta.csv")
    tracks_path = os.path.join(base_data_dir, f"{file_num_str}_tracks.csv")

    # Create separate save path for each recording
    save_path = os.path.join(save_dir, f'cut_in_trajectories_{file_num_str}.csv')

    print(f"\nProcessing recording {file_num_str}...")

    # Check if files exist
    if not os.path.exists(tracksMeta_path) or not os.path.exists(tracks_path):
        print(f"Skipped: Files not found")
        continue

    # Read data
    try:
        meta = pd.read_csv(tracksMeta_path)
        tracks = pd.read_csv(tracks_path)
    except Exception as e:
        print(f"Failed to read: {e}")
        continue

    # 3. Select candidate vehicles from metadata
    # Vehicles must be cars and have exactly one recorded lane change
    target_ids = meta[(meta['class'] == 'Car') & (meta['numLaneChanges'] == 1)]['id'].tolist()

    cut_in_data = []

    # 4. Check each vehicle for cut-in behavior and extract precise trajectory segments
    for vid in target_ids:
        df = tracks[tracks['id'] == vid].sort_values('frame')
        df = df.reset_index(drop=True)

        if len(df) == 0:
            continue

        for i_idx in range(len(df)):
            current_row = df.iloc[i_idx]

            # Check if there is a preceding vehicle in adjacent lane
            if current_row['leftPrecedingId'] > 0 or current_row['rightPrecedingId'] > 0:
                # Identify the vehicle that is being cut-in (victim vehicle)
                victim_id = current_row['leftPrecedingId'] if current_row['leftPrecedingId'] > 0 else current_row[
                    'rightPrecedingId']

                # Check if victim vehicle appears behind target vehicle in future frames
                future_rows = df.iloc[i_idx + 1:]
                if not future_rows.empty and (future_rows['followingId'] == victim_id).any():
                    # Find frames where vehicles are side by side
                    if victim_id == current_row['leftPrecedingId']:
                        along_side_rows = future_rows[future_rows['leftAlongsideId'] == victim_id]
                    else:
                        along_side_rows = future_rows[future_rows['rightAlongsideId'] == victim_id]

                    if along_side_rows.empty:
                        continue

                    along_side_row = along_side_rows.iloc[0]
                    along_side_frame = along_side_row['frame']

                    # Find cut-in start frame
                    start_frame = along_side_frame - PRE_FRAMES

                    start_frames = df[df['frame'] >= start_frame]
                    if start_frames.empty:
                        continue
                    start_row = start_frames.iloc[0]
                    start_index = start_row.name

                    # Find cut-in completion frame
                    cut_completed_frames = future_rows[future_rows['followingId'] == victim_id]
                    if not cut_completed_frames.empty:
                        cut_complete_row = cut_completed_frames.iloc[0]
                        cut_complete_frame = cut_complete_row['frame']
                        cut_complete_index = cut_complete_row.name

                        planned_end_frame = cut_complete_frame + POST_FRAMES
                        end_frames = df[df['frame'] <= planned_end_frame]
                        if end_frames.empty:
                            continue
                        end_row = end_frames.iloc[-1]
                        actual_end_index = end_row.name
                        actual_end_frame = end_row['frame']

                        # Check if trajectory meets frame count requirements
                        if actual_end_frame < planned_end_frame:
                            continue

                        # Detect lane change within the frame window
                        lane_change_detected = False
                        for k in range(cut_complete_index + 1, actual_end_index + 1):
                            if k >= len(df):
                                break
                            row_k = df.iloc[k]
                            row_k_prev = df.iloc[k - 1]
                            if row_k['laneId'] != row_k_prev['laneId']:
                                lane_change_detected = True
                                break

                        if lane_change_detected:
                            continue

                        # Extract cut-in vehicle trajectory segment
                        cut_in_vehicle_segment = df[(df['frame'] >= start_frame) & (df['frame'] <= actual_end_frame)]

                        if not cut_in_vehicle_segment.empty:
                            cut_in_data.append(cut_in_vehicle_segment)

                        break

    # 5. Save results for current recording file
    if cut_in_data:
        result = pd.concat(cut_in_data)
        result = result.sort_values(['id', 'frame'])
        result.to_csv(save_path, index=False)

        # Statistics
        unique_vehicles = result['id'].nunique()
        unique_interactions = len(cut_in_data)
        print(f" Saved {unique_interactions} cut-in events")
        print(f" Involves {unique_vehicles} different vehicles")
        print(f" Total trajectory length: {len(result)} rows")
        print(f" Saved to: {save_path}")
    else:
        print(f"No cut-in behavior found")

print(f"\nAll recording files processed! Results saved in: {save_dir}")