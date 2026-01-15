"""
Extracts lane change scenarios from driving datasets.

Focuses on scenarios where a vehicle changes lanes between three surrounding vehicles.
Extracts complete trajectories: the lane-changing vehicle's movement and the relationship
transformation of the three surrounding vehicles (from adjacent lane to current lane).
"""

import pandas as pd
import os

# Parameter Settings
PRE_FRAMES = 50    # Number of frames before lane change starts
POST_FRAMES = 50   # Number of frames after lane change is completed


# Helper function: Check if ID is valid (non-zero and non-NaN)
def is_valid_id(id_value):
    return pd.notna(id_value) and int(id_value) > 0


# Helper function: Check if a vehicle has complete trajectory within specified frame range
def has_trajectory_in_range(tracks_df, vehicle_id, start_frame, end_frame):
    vehicle_traj = tracks_df[(tracks_df['id'] == vehicle_id) &
                             (tracks_df['frame'] >= start_frame) &
                             (tracks_df['frame'] <= end_frame)]

    # Get actual number of existing frames
    actual_frames = len(vehicle_traj)
    expected_frames = end_frame - start_frame + 1

    # Require 100% completeness
    return actual_frames == expected_frames

def process_file(tracksMeta_path, tracks_path, save_dir, file_id):
    """Process single file, preserving original output format"""

    # Build save paths
    save_path = os.path.join(save_dir, f'round_up_trajectories_{file_id:02d}.csv')
    info_path = os.path.join(save_dir, f'round_up_info_{file_id:02d}.csv')

    # Read data
    meta = pd.read_csv(tracksMeta_path)
    tracks = pd.read_csv(tracks_path)

    # Filter target vehicles from meta: Car class with lane changes, and get driving direction
    driving_direction_dict = {}
    for _, row in meta.iterrows():
        if row['class'] == 'Car' and row['numLaneChanges'] != 0:
            driving_direction_dict[row['id']] = row['drivingDirection']

    target_ids = list(driving_direction_dict.keys())

    round_up_data = []
    round_up_info = []

    print(f"\nStart processing file {file_id:02d}")
    print(f"Found {len(target_ids)} vehicles that may have performed lane changes")

    # Check each vehicle for lane change behavior and extract precise trajectory segment
    for vid in target_ids:
        df = tracks[tracks['id'] == vid].sort_values('frame')

        if len(df) == 0:
            continue

        df = df.reset_index(drop=True)

        # Get vehicle's driving direction from dictionary
        driving_direction = driving_direction_dict.get(vid, 1)

        # Find all lane change events
        for i in range(1, len(df)):
            current_row = df.iloc[i]
            prev_row = df.iloc[i - 1]

            # Detect lane change: laneId changes
            if current_row['laneId'] != prev_row['laneId']:
                round_up_frame = current_row['frame']

                # Determine lane change direction
                new_lane = current_row['laneId']
                old_lane = prev_row['laneId']

                if driving_direction == 1:
                    if new_lane > old_lane:
                        direction = 'left'
                        preceding_id_field = 'leftPrecedingId'
                        following_id_field = 'leftFollowingId'
                    else:
                        direction = 'right'
                        preceding_id_field = 'rightPrecedingId'
                        following_id_field = 'rightFollowingId'
                else:
                    if new_lane > old_lane:
                        direction = 'right'
                        preceding_id_field = 'rightPrecedingId'
                        following_id_field = 'rightFollowingId'
                    else:
                        direction = 'left'
                        preceding_id_field = 'leftPrecedingId'
                        following_id_field = 'leftFollowingId'

                print(f"\nVehicle {vid}: driving direction{driving_direction}, "
                      f"lane {old_lane}->{new_lane}, determined as {direction} lane change")

                # Check conditions in the frame before lane change
                # Requirement: Current lane preceding vehicle, adjacent lane preceding vehicle,
                # and adjacent lane following vehicle must all exist
                if not (is_valid_id(prev_row['precedingId']) and
                        is_valid_id(prev_row[preceding_id_field]) and
                        is_valid_id(prev_row[following_id_field])):
                    print(f"Does not meet pre-lane-change conditions: has ID 0")
                    continue

                # Get key IDs
                original_preceding_id = int(prev_row['precedingId'])
                adj_preceding_id = int(prev_row[preceding_id_field])
                adj_following_id = int(prev_row[following_id_field])

                print(f"Before lane change: current lane preceding={original_preceding_id}, "
                      f"{preceding_id_field}={adj_preceding_id}, "
                      f"{following_id_field}={adj_following_id}")

                # Find lane change completion frame (adjacent lane vehicles become current lane vehicles)
                change_complete_frame = None
                found_new_preceding = False
                found_new_following = False
                found_original_preceding_as_adj = False
                new_adj_preceding_field = None

                # Check up to 50 frames after lane change
                for j in range(i, min(i + 50, len(df))):
                    post_row = df.iloc[j]

                    # Check if current lane preceding and following vehicles are valid
                    if not (is_valid_id(post_row['precedingId']) and is_valid_id(post_row['followingId'])):
                        continue

                    # Check if adj_preceding_id becomes precedingId
                    if int(post_row['precedingId']) == adj_preceding_id:
                        found_new_preceding = True

                    # Check if adj_following_id becomes followingId
                    if int(post_row['followingId']) == adj_following_id:
                        found_new_following = True

                    # Check if original preceding vehicle becomes adjacent lane preceding vehicle
                    if not found_original_preceding_as_adj:
                        # Check left adjacent lane
                        if is_valid_id(post_row['leftPrecedingId']) and int(
                                post_row['leftPrecedingId']) == original_preceding_id:
                            found_original_preceding_as_adj = True
                            new_adj_preceding_field = 'leftPrecedingId'
                            print(f"Frame {post_row['frame']}: original preceding vehicle "
                                  f"{original_preceding_id} becomes left adjacent lane preceding")

                        # Check right adjacent lane
                        elif is_valid_id(post_row['rightPrecedingId']) and int(
                                post_row['rightPrecedingId']) == original_preceding_id:
                            found_original_preceding_as_adj = True
                            new_adj_preceding_field = 'rightPrecedingId'
                            print(f"  Frame {post_row['frame']}: original preceding vehicle "
                                  f"{original_preceding_id} becomes right adjacent lane preceding")
                    # When all three conditions are met, consider lane change completed
                    if found_new_preceding and found_new_following and found_original_preceding_as_adj:
                        change_complete_frame = post_row['frame']
                        break

                # Check if all conditions are met
                conditions = {
                    'found_new_preceding': found_new_preceding,
                    'found_new_following': found_new_following,
                    'found_original_preceding_as_adj': found_original_preceding_as_adj
                }

                if not all(conditions.values()):
                    failed_conditions = [k for k, v in conditions.items() if not v]
                    print(f"Does not meet post-lane-change conditions: {', '.join(failed_conditions)}")
                    continue

                print(f"Post-lane-change conditions satisfied, completion frame: {change_complete_frame}")
                print(f"After lane change, original preceding vehicle {original_preceding_id} becomes {new_adj_preceding_field}")

                # Extract trajectory segment: centered around lane change completion frame
                center_frame = change_complete_frame
                start_frame = max(df.iloc[0]['frame'], center_frame - PRE_FRAMES)
                end_frame = min(df.iloc[-1]['frame'], center_frame + POST_FRAMES)

                # Check if trajectory segment length is complete
                expected_frames = PRE_FRAMES + POST_FRAMES + 1
                segment_df = df[(df['frame'] >= start_frame) & (df['frame'] <= end_frame)].copy()

                if len(segment_df) != expected_frames:
                    print(f"Trajectory segment length incomplete: {len(segment_df)} frames, expected {expected_frames} frames")
                    continue

                print(f"Extract trajectory segment: {start_frame}-{end_frame} ({len(segment_df)} frames)")

                # Check ID validity and consistency throughout the entire trajectory segment
                all_ids_valid = True
                for idx, row in segment_df.iterrows():
                    # Determine which IDs to check based on time
                    if row['frame'] < change_complete_frame:
                        # Before lane change completion: check current lane preceding, adjacent lane preceding and following
                        # Need to check not only ID validity but also ID value consistency
                        if not (is_valid_id(row['precedingId']) and
                                int(row['precedingId']) == original_preceding_id and
                                is_valid_id(row[preceding_id_field]) and
                                int(row[preceding_id_field]) == adj_preceding_id and
                                is_valid_id(row[following_id_field]) and
                                int(row[following_id_field]) == adj_following_id):
                            all_ids_valid = False
                            print(f"Pre-lane-change frame {row['frame']}: ID mismatch or invalid")
                            print(f"Expected: precedingId={original_preceding_id}, "
                                  f"{preceding_id_field}={adj_preceding_id}, "
                                  f"{following_id_field}={adj_following_id}")
                            print(f"Actual: precedingId={row['precedingId']}, "
                                  f"{preceding_id_field}={row[preceding_id_field]}, "
                                  f"{following_id_field}={row[following_id_field]}")
                            break
                    else:
                        # After lane change completion: check current lane preceding and following
                        # Check ID value consistency
                        if not (is_valid_id(row['precedingId']) and
                                int(row['precedingId']) == adj_preceding_id and
                                is_valid_id(row['followingId']) and
                                int(row['followingId']) == adj_following_id):
                            all_ids_valid = False
                            print(f"Post-lane-change frame {row['frame']}: ID mismatch or invalid")
                            print(f"Expected: precedingId={adj_preceding_id}, followingId={adj_following_id}")
                            print(f"Actual: precedingId={row['precedingId']}, followingId={row['followingId']}")
                            break

                if not all_ids_valid:
                    print(f"Trajectory segment ID check failed")
                    continue

                print(f"Trajectory segment ID check passed")

                # Check if original preceding vehicle has complete trajectory data in the entire segment
                print(f"Checking trajectory of original preceding vehicle {original_preceding_id} in the entire segment...")
                if not has_trajectory_in_range(tracks, original_preceding_id, start_frame, end_frame):
                    print(f"Original preceding vehicle {original_preceding_id} has incomplete trajectory"
                          f"in frame range {start_frame}-{end_frame}")
                    continue

                # Check if adjacent lane preceding vehicle after lane change has complete trajectory data in the entire segment
                print(f"Checking trajectory of adjacent lane preceding vehicle {adj_preceding_id} in the entire segment...")
                if not has_trajectory_in_range(tracks, adj_preceding_id, start_frame, end_frame):
                    print(f"  Adjacent lane preceding vehicle {adj_preceding_id} after lane change has incomplete trajectory "
                          f"in frame range {start_frame}-{end_frame}")
                    continue

                # Check if adjacent lane following vehicle after lane change has complete trajectory data in the entire segment
                print(f"Checking trajectory of adjacent lane following vehicle {adj_following_id} in the entire segment...")
                if not has_trajectory_in_range(tracks, adj_following_id, start_frame, end_frame):
                    print(f"Adjacent lane following vehicle {adj_following_id} after lane change has incomplete trajectory "
                          f"in frame range {start_frame}-{end_frame}")
                    continue

                print(f"All related vehicles have complete trajectories within the segment")

                # All conditions satisfied, save trajectory segment
                segment_df['round_up_id'] = len(round_up_info) + 1
                segment_df['round_up_frame'] = round_up_frame
                segment_df['change_complete_frame'] = change_complete_frame
                segment_df['round_up_direction'] = direction
                segment_df['driving_direction'] = driving_direction
                segment_df['old_lane'] = old_lane
                segment_df['new_lane'] = new_lane
                segment_df['adj_preceding_id'] = adj_preceding_id
                segment_df['adj_following_id'] = adj_following_id
                segment_df['original_preceding_id'] = original_preceding_id

                round_up_data.append(segment_df)

                # Save scene information
                info = {
                    'scene_id': len(round_up_info) + 1,
                    'vehicle_id': vid,
                    'driving_direction': driving_direction,
                    'round_up_frame': round_up_frame,
                    'change_complete_frame': change_complete_frame,
                    'direction': direction,
                    'old_lane': old_lane,
                    'new_lane': new_lane,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'adj_preceding_id': adj_preceding_id,
                    'adj_following_id': adj_following_id,
                    'original_preceding_id': original_preceding_id,
                    'num_frames': len(segment_df),
                    'pre_frames': change_complete_frame - start_frame,
                    'post_frames': end_frame - change_complete_frame
                }
                round_up_info.append(info)

                print(f"Successfully extracted scene {len(round_up_info)}")

                break

    # Save results
    if round_up_data:
        # Merge all trajectory segments
        result = pd.concat(round_up_data)
        result = result.sort_values(['round_up_id', 'frame'])

        # Save trajectory data
        result.to_csv(save_path, index=False)

        # Save scene information
        info_df = pd.DataFrame(round_up_info)
        info_df.to_csv(info_path, index=False)

        unique_vehicles = result['id'].nunique()
        unique_scenes = len(round_up_data)

        print("\n" + "=" * 70)
        print(f"File {file_id:02d} extraction statistics:")
        print(f"Total found {unique_scenes} qualifying lane change scenarios")
        print(f"Involving {unique_vehicles} different vehicles")
        print(f"Total trajectory segment length: {len(result)} rows")
        print(f"Average length per scenario: {len(result) / unique_scenes:.1f} frames")

        print(f"Trajectory data saved to: {save_path}")
        print(f"Scene information saved to: {info_path}")
        print("=" * 70)

    else:
        print(f"File {file_id:02d}: No qualifying lane change behavior found")

    return len(round_up_data)

def main():
    """Automatically process 60 files, preserving original output format"""

    # Set base data directories
    base_data_dir = './data/highD-dataset-v1.0'         # HighD dataset directory
    save_dir = './output/lane_change_trajectories'      # Directory for saving lane_change vehicle trajectories
    os.makedirs(save_dir, exist_ok=True)

    total_scenes = 0

    # Process 60 files
    for file_id in range(1, 61):
        # Construct file paths
        tracksMeta_path = os.path.join(base_data_dir, f"{file_id:02d}_tracksMeta.csv")
        tracks_path = os.path.join(base_data_dir, f"{file_id:02d}_tracks.csv")

        # Check if files exist
        if not os.path.exists(tracksMeta_path) or not os.path.exists(tracks_path):
            print(f"\nFile {file_id:02d} does not exist, skipping")
            continue

        # Process current file
        scenes_count = process_file(tracksMeta_path, tracks_path, save_dir, file_id)
        total_scenes += scenes_count

        print(f"File {file_id:02d} processing completed, found {scenes_count} scenarios")

    print("\n" + "=" * 70)
    print(f"Total found {total_scenes} qualifying lane change scenarios across 60 files")
    print("=" * 70)

main()