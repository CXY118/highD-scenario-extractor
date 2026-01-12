"""
Extracts trajectories of victim vehicles (vehicles being cut in on).

Identifies and extracts the trajectory data of vehicles that are
being cut in on during cut-in maneuvers.
"""

import pandas as pd
import os

# 1. Set base data directories
base_data_dir = './data/highD-dataset-v1.0'                   # HighD dataset directory
cutin_data_dir = './output/cutin_trajectories'                # Directory for saving cut-in vehicle trajectories
# cutin_data_dir = "./output/cutin_trajectories_filtered"     # Directory for speed-filtered results (if speed filtered)
output_dir = './output/victim_trajectories'                   # Directory for saving victim vehicle trajectories

os.makedirs(output_dir, exist_ok=True)

# 2. Process all recording files (01-60)
for file_num in range(1, 61):
    # Build 2-digit file number
    file_num_str = f"{file_num:02d}"

    # Create file paths
    cut_in_path = os.path.join(cutin_data_dir, f'cut_in_trajectories_{file_num_str}.csv')
    #cut_in_path = os.path.join(cutin_data_dir, f'cut_in_trajectories_{file_num_str}_filtered.csv') (if speed filtered)
    tracks_path = os.path.join(base_data_dir, f'{file_num_str}_tracks.csv')
    save_path = os.path.join(output_dir, f'cutted_in_trajectories_{file_num_str}.csv')

    print(f"\n{'=' * 60}")
    print(f"Processing recording {file_num_str}...")
    print(f"Cut-in events file: {cut_in_path}")
    print(f"Trajectory file: {tracks_path}")
    print(f"Output file: {save_path}")
    print(f"{'=' * 60}")

    # Check if files exist
    if not os.path.exists(cut_in_path):
        print(f"Warning: Cut-in events file not found, skipping")
        continue
    if not os.path.exists(tracks_path):
        print(f"Warning: Trajectory file not found, skipping")
        continue

    # 3. Load data
    print("\nLoading cut-in event data...")
    try:
        cut_in_df = pd.read_csv(cut_in_path)
        tracks_df = pd.read_csv(tracks_path)
    except Exception as e:
        print(f"Failed to read: {e}")
        continue

    # Store cut-in event list
    cut_in_events = []

    # 4. Group by cut-in vehicle ID and detect each cut-in event
    cut_in_vehicle_ids = cut_in_df['id'].unique()

    for cut_in_vehicle_id in cut_in_vehicle_ids:
        # Get trajectory of this cut-in vehicle
        cut_in_vehicle_data = cut_in_df[cut_in_df['id'] == cut_in_vehicle_id].sort_values('frame')

        if len(cut_in_vehicle_data) < 2:
            continue

        # Determine start and end frames for this cut-in event
        start_frame = cut_in_vehicle_data['frame'].min()
        end_frame = cut_in_vehicle_data['frame'].max()

        # Search for victim vehicle in the entire trajectory
        for i in range(len(cut_in_vehicle_data)):
            current_row = cut_in_vehicle_data.iloc[i]

            # Check left and right preceding vehicles
            left_target = current_row['leftPrecedingId']
            right_target = current_row['rightPrecedingId']

            # Store possible cut-in targets
            possible_targets = []
            if left_target > 0:
                possible_targets.append(('left', left_target))
            if right_target > 0:
                possible_targets.append(('right', right_target))

            # Check each possible target
            for position, victim_id in possible_targets:
                # Check if this vehicle appears behind ego vehicle in subsequent frames
                found_cut_in = False

                for j in range(i + 1, len(cut_in_vehicle_data)):
                    if cut_in_vehicle_data.iloc[j]['followingId'] == victim_id:
                        cut_in_events.append({
                            'cut_in_vehicle_id': cut_in_vehicle_id,
                            'victim_vehicle_id': victim_id,
                            'start_frame': start_frame,
                            'end_frame': end_frame,
                            'cut_in_detected_frame': current_row['frame'],
                            'cut_in_direction': position
                        })
                        found_cut_in = True
                        break

                if found_cut_in:
                    break

            # If cut-in event found, break out of loop
            if possible_targets and found_cut_in:
                break

    print(f"Found {len(cut_in_events)} cut-in events")

    # 5. Extract victim vehicle trajectories
    all_trajectories = []

    for event_idx, event in enumerate(cut_in_events):
        victim_id = event['victim_vehicle_id']
        start_frame = event['start_frame']
        end_frame = event['end_frame']
        cut_in_id = event['cut_in_vehicle_id']
        cut_in_detected_frame = event['cut_in_detected_frame']

        # Extract complete trajectory of victim vehicle
        target_traj = tracks_df[tracks_df['id'] == victim_id].copy()

        if len(target_traj) == 0:
            print(f"Event {event_idx + 1}: Victim vehicle {victim_id} trajectory not found")
            continue

        # Extract trajectory segment within cut-in event time range
        target_traj_segment = target_traj[
            (target_traj['frame'] >= start_frame) &
            (target_traj['frame'] <= end_frame)
            ].copy()

        if len(target_traj_segment) == 0:
            print(f"Event {event_idx + 1}: Victim vehicle {victim_id} has no data in frame range {start_frame}-{end_frame}")
            continue

        target_traj_segment['is_cut_in_target'] = True
        target_traj_segment['cut_in_event_id'] = event_idx + 1
        target_traj_segment['cut_in_start_frame'] = start_frame
        target_traj_segment['cut_in_end_frame'] = end_frame
        target_traj_segment['cut_in_detected_frame'] = cut_in_detected_frame
        target_traj_segment['cut_in_vehicle_id'] = cut_in_id
        target_traj_segment['victim_vehicle_id'] = victim_id
        target_traj_segment['recording_id'] = file_num_str

        all_trajectories.append(target_traj_segment)
        print(f"   事件{event_idx + 1}: 车辆 {victim_id} 被 {cut_in_id} 切入")
        print(f"       时间范围: 帧{start_frame}到{end_frame} ({len(target_traj_segment)}帧)")
        print(f"       切入检测帧: {cut_in_detected_frame}")

    # 6. Merge and save results for current recording
    if all_trajectories:
        result_df = pd.concat(all_trajectories, ignore_index=True)

        # Add cut-in event ID to each segment
        result_df = result_df.sort_values(['cut_in_event_id', 'frame'])

        result_df.to_csv(save_path, index=False)
        print(f"\nSave completed! Total {len(result_df)} rows, {len(all_trajectories)} cut-in event trajectories")
        print(f"File saved to: {save_path}")

        # Output statistics
        print(f"\nStatistics:")
        print(f"- Total cut-in events: {len(cut_in_events)}")
        print(f"- Unique victim vehicles: {result_df['victim_vehicle_id'].nunique()}")

        # Check if any victim vehicle was cut in multiple times
        victim_counts = result_df.groupby('victim_vehicle_id')['cut_in_event_id'].nunique()
        multi_cut_victims = victim_counts[victim_counts > 1]
        if len(multi_cut_victims) > 0:
            print(f"- Vehicles cut in multiple times: {len(multi_cut_victims)}")
            for victim_id, count in multi_cut_victims.items():
                print(f"Vehicle {victim_id}: {count} cut-ins")
    else:
        print("No cut-in events found")

print(f"\n{'=' * 60}")
print(f"All recordings processed")
print(f"Results saved in: {output_dir}")
print(f"{'=' * 60}")