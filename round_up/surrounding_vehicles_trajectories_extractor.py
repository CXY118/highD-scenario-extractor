"""
Extracts trajectories of surrounding vehicles in lane change scenarios.

Extracts complete trajectories for three surrounding vehicles:
1. Adjacent lane preceding vehicle
2. Adjacent lane following vehicle
3. Original lane preceding vehicle

Also extracts the lane-changing vehicle's trajectory for reference.
All trajectories are extracted within the same frame range as the lane change event.
"""
import pandas as pd
import os
import glob
import re

# 1. Set base data directories
changing_dir = './output/lane_change_trajectories'               # Directory for saving lane_change vehicle trajectories
#changing_dir = './output/lane_change_trajectories_filtered'     # Directory for speed-filtered results (if speed filtered)
tracks_dir = './data/highD-dataset-v1.0'                         # HighD dataset directory

# 2. Create output directories
output_dirs = {
    'adj_preceding': './output/surround_data/adj_preceding_data',
    'adj_following': './output/surround_data/adj_following_data',
    'original_preceding': './output/surround_data/original_preceding_data',
    'changing_vehicle': './output/surround_data/changing_vehicle_data'
}

for role, dir_path in output_dirs.items():
    os.makedirs(dir_path, exist_ok=True)
    print(f"Created directory: {dir_path}")

# 3. Find all changing files
changing_files = glob.glob(os.path.join(changing_dir, 'changing_info_*.csv'))
#changing_files = glob.glob(os.path.join(changing_dir, 'changing_info_*_clear.csv'))      # if speed filtered
print(f"Found {len(changing_files)} changing files")

# 4. Find corresponding tracks files for each changing file
summary_data = []

# Initialize trajectory lists by role
role_trajectories = {role: [] for role in output_dirs.keys()}

for changing_file in changing_files:
    # Extract scene number from filename
    file_name = os.path.basename(changing_file)
    match = re.search(r'changing_info_(\d+)\.csv', file_name)
    #match = re.search(r'changing_info_(\d+)_clear\.csv', file_name)       # if speed filtered

    if not match:
        print(f"Warning: Unable to extract scene number from filename {file_name}, skipping")
        continue

    scene_num = match.group(1)
    scene_id = int(scene_num)

    # Construct corresponding tracks file path
    tracks_file = os.path.join(tracks_dir, f'{scene_num.zfill(2)}_tracks.csv')

    # Check if tracks file exists
    if not os.path.exists(tracks_file):
        print(f"Warning: Corresponding tracks file not found: {tracks_file}")
        # Try alternative naming format
        tracks_file_alt = os.path.join(tracks_dir, f'{scene_num}_tracks.csv')
        if os.path.exists(tracks_file_alt):
            tracks_file = tracks_file_alt
        else:
            print(f"Error: Unable to find trajectory file for scene {scene_num}, skipping")
            continue

    print(f"\n{'=' * 60}")
    print(f"Processing Scene {scene_num}:")
    print(f"Changing file: {file_name}")
    print(f"Tracks file: {os.path.basename(tracks_file)}")

    # 5. Load data
    try:
        changing_df = pd.read_csv(changing_file)
        tracks_df = pd.read_csv(tracks_file)
    except Exception as e:
        print(f"Error: Failed to read files - {e}")
        continue

    print(f"Read {len(changing_df)} lane change events")
    print(f"Trajectory data has {len(tracks_df)} rows")

    # Count extraction by role for current scene
    scene_role_counts = {role: 0 for role in output_dirs.keys()}

    # 6. Extract trajectories of target vehicles
    for event_idx, event in changing_df.iterrows():
        start_frame = event['start_frame']
        end_frame = event['end_frame']
        vehicle_id = event['vehicle_id']
        event_id = f"scene{scene_num}_event{event_idx + 1}"

        print(f"\nProcessing event {event_idx + 1}: vehicle {vehicle_id}")
        print(f"Time range: frames {start_frame} to {end_frame}")
        print(f"Direction: {event['direction']} ({event['old_lane']} -> {event['new_lane']})")

        extraction_targets = []

        # Add adj_preceding_id
        if not pd.isna(event['adj_preceding_id']):
            extraction_targets.append({
                'id': int(event['adj_preceding_id']),
                'role': 'adj_preceding',
                'description': 'Adjacent lane preceding vehicle'
            })

        # Add adj_following_id
        if not pd.isna(event['adj_following_id']):
            extraction_targets.append({
                'id': int(event['adj_following_id']),
                'role': 'adj_following',
                'description': 'Adjacent lane following vehicle'
            })

        # Add original_preceding_id
        if not pd.isna(event['original_preceding_id']):
            extraction_targets.append({
                'id': int(event['original_preceding_id']),
                'role': 'original_preceding',
                'description': 'Original lane preceding vehicle'
            })

        # Add lane-changing vehicle itself
        extraction_targets.append({
            'id': int(event['vehicle_id']),
            'role': 'changing_vehicle',
            'description': 'lane changing vehicle'
        })

        # Extract trajectories for each target vehicle
        for target in extraction_targets:
            target_id = target['id']
            target_role = target['role']
            target_desc = target['description']

            # Extract complete trajectory of this vehicle
            target_traj = tracks_df[tracks_df['id'] == target_id].copy()

            if len(target_traj) == 0:
                print(f"{target_desc} (ID: {target_id}): No trajectory found")
                continue

            # Extract trajectory segment within event time range
            target_traj_segment = target_traj[
                (target_traj['frame'] >= start_frame) &
                (target_traj['frame'] <= end_frame)
                ].copy()

            if len(target_traj_segment) == 0:
                print(f"{target_desc} (ID: {target_id}): No data in specified time range")
                continue

            # Add event information
            target_traj_segment['scene_id'] = scene_id
            target_traj_segment['event_id'] = event_id
            target_traj_segment['vehicle_role'] = target_role
            target_traj_segment['role_description'] = target_desc
            target_traj_segment['changing_vehicle_id'] = int(event['vehicle_id'])
            target_traj_segment['event_start_frame'] = start_frame
            target_traj_segment['event_end_frame'] = end_frame
            target_traj_segment['changing_frame'] = event['changing_frame']
            target_traj_segment['driving_direction'] = event['driving_direction']
            target_traj_segment['direction'] = event['direction']
            target_traj_segment['old_lane'] = event['old_lane']
            target_traj_segment['new_lane'] = event['new_lane']
            target_traj_segment['source_changing_file'] = file_name
            target_traj_segment['source_tracks_file'] = os.path.basename(tracks_file)

            role_trajectories[target_role].append(target_traj_segment)
            scene_role_counts[target_role] += 1

            print(f"{target_desc} (ID: {target_id}): Extracted {len(target_traj_segment)} frames")

    # Record scene summary information
    summary_data.append({
        'scene_id': scene_id,
        'changing_file': file_name,
        'tracks_file': os.path.basename(tracks_file),
        'total_events': len(changing_df),
        'adj_preceding_count': scene_role_counts['adj_preceding'],
        'adj_following_count': scene_role_counts['adj_following'],
        'original_preceding_count': scene_role_counts['original_preceding'],
        'changing_vehicle_count': scene_role_counts['changing_vehicle'],
        'total_extracted': sum(scene_role_counts.values())
    })

    print(f"\nScene {scene_num} extraction statistics:")

# 7. Save data by role separately
for role, trajectories in role_trajectories.items():
    if trajectories:
        # Merge all trajectories for this role
        result_df = pd.concat(trajectories, ignore_index=True)

        # Sort by scene_id, event_id and frame
        result_df = result_df.sort_values(['scene_id', 'event_id', 'frame'])

        # Save file path
        save_path = os.path.join(output_dirs[role], f'all_{role}_trajectories.csv')

        result_df.to_csv(save_path, index=False)

        print(f"Save path: {save_path}")
        print(f"Total data rows: {len(result_df)}")
        print(f"Unique vehicle count: {result_df['id'].nunique()}")
        print(f"Unique event count: {result_df['event_id'].nunique()}")
        print(f"Number of scenes involved: {result_df['scene_id'].nunique()}")

        # Save separate files by scene
        scenes = result_df['scene_id'].unique()
        for scene in scenes:
            scene_data = result_df[result_df['scene_id'] == scene]
            scene_save_path = os.path.join(output_dirs[role], f'scene{scene:02d}_{role}_trajectories.csv')
            scene_data.to_csv(scene_save_path, index=False)

        print(f"Saved {len(scenes)} separate files by scene")
    else:
        print(f"\n{role}: No trajectory data")

# 8. Save summary information
if summary_data:
    summary_df = pd.DataFrame(summary_data)

    totals = {
        'scene_id': 'Total',
        'changing_file': '-',
        'tracks_file': '-',
        'total_events': summary_df['total_events'].sum(),
        'adj_preceding_count': summary_df['adj_preceding_count'].sum(),
        'adj_following_count': summary_df['adj_following_count'].sum(),
        'original_preceding_count': summary_df['original_preceding_count'].sum(),
        'changing_vehicle_count': summary_df['changing_vehicle_count'].sum(),
        'total_extracted': summary_df['total_extracted'].sum()
    }

    summary_df = pd.concat([summary_df, pd.DataFrame([totals])], ignore_index=True)
    summary_path = os.path.join(output_dirs['changing_vehicle'], '..', 'extraction_summary.csv')
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'=' * 60}")
    print("Summary information:")
    print(f"Total scenes processed: {len(summary_data)}")
    print(f"Total lane change events: {totals['total_events']}")
    print(f"\n Extraction statistics by role:")
    print(f"Adjacent lane preceding vehicle: {totals['adj_preceding_count']} trajectories")
    print(f"Adjacent lane following vehicle: {totals['adj_following_count']} trajectories")
    print(f"Original lane preceding vehicle: {totals['original_preceding_count']} trajectories")
    print(f"Lane-changing vehicle: {totals['changing_vehicle_count']} trajectories")
    print(f"Total: {totals['total_extracted']} trajectories")
    print(f"\nSummary file saved to: {summary_path}")
    print(f"\nProcessing completed!")

print(f"\nProcessing completed!")
