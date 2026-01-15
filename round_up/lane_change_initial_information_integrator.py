"""
HighwayEnv vehicle initial state alignment
Extract and format the initial vehicle state from the lane change scene to match the HighwayEnv representation.
"""
import pandas as pd
import numpy as np
import os

# 1. Set base data directories
base_input_dir = './output/surround_data/surround_data_initial'
intermediate_output_dir = './output/round_up_message'
final_output_file = './output/round_up_scenes_integrated.csv'
meta_base_dir = './data/highD-dataset-v1.0'

os.makedirs(intermediate_output_dir, exist_ok=True)


# 2.Process Each Scene Data
def process_single_scene(scene_num):
    scene_str = f"{scene_num:02d}"

    print(f"\n{'=' * 60}")
    print(f"Processing scene {scene_str}")
    print(f"{'=' * 60}")

    # Construct file paths
    base_path = os.path.join(base_input_dir, 'merging_vehicle_data',
                             f'scene{scene_str}_merging_vehicle_trajectories_initial.csv')
    adj_following_path = os.path.join(base_input_dir, 'adj_following_data',
                                      f'scene{scene_str}_adj_following_trajectories_initial.csv')
    adj_preceding_path = os.path.join(base_input_dir, 'adj_preceding_data',
                                      f'scene{scene_str}_adj_preceding_trajectories_initial.csv')
    original_preceding_path = os.path.join(base_input_dir, 'original_preceding_data',
                                           f'scene{scene_str}_original_preceding_trajectories_initial.csv')

    output_path = os.path.join(intermediate_output_dir, f"scene{scene_str}_vehicles_initial_state_combined.csv")

    # Check if files exist
    required_files = [base_path, adj_following_path, adj_preceding_path, original_preceding_path]
    missing_files = []

    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(os.path.basename(file_path))

    if missing_files:
        print(f"Missing files: {missing_files}")
        return None, f"Scene {scene_str}: Files missing"

    try:
        df_base = pd.read_csv(base_path)
        df_adj_following = pd.read_csv(adj_following_path)
        df_adj_preceding = pd.read_csv(adj_preceding_path)
        df_original_preceding = pd.read_csv(original_preceding_path)

        print(f"Read successfully: base vehicle ({len(df_base)} rows), "
              f"following vehicle ({len(df_adj_following)} rows), "
              f"adjacent preceding vehicle ({len(df_adj_preceding)} rows), "
              f"original preceding vehicle ({len(df_original_preceding)} rows)")

        # Find common frames
        common_frames = set(df_base['frame']).intersection(
            set(df_adj_following['frame']),
            set(df_adj_preceding['frame']),
            set(df_original_preceding['frame'])
        )

        if len(common_frames) == 0:
            print("Warning: No common frames")
            return None, f"Scene {scene_str}: No common frames"

        print(f"Common frames count: {len(common_frames)}")

        # Extract data for common frames
        df_base_common = df_base[df_base['frame'].isin(common_frames)].sort_values('frame').reset_index(drop=True)
        df_adj_following_common = df_adj_following[df_adj_following['frame'].isin(common_frames)].sort_values(
            'frame').reset_index(drop=True)
        df_adj_preceding_common = df_adj_preceding[df_adj_preceding['frame'].isin(common_frames)].sort_values(
            'frame').reset_index(drop=True)
        df_original_preceding_common = df_original_preceding[
            df_original_preceding['frame'].isin(common_frames)].sort_values('frame').reset_index(drop=True)

        # Determine minimum length
        min_length = min(len(df_base_common), len(df_adj_following_common),
                         len(df_adj_preceding_common), len(df_original_preceding_common))

        print(f"Aligned data rows: {min_length}")

        # Process each row of data
        combined_data = []

        for idx in range(min_length):
            row_base = df_base_common.iloc[idx]
            row_following = df_adj_following_common.iloc[idx]
            row_preceding = df_adj_preceding_common.iloc[idx]
            row_original = df_original_preceding_common.iloc[idx]

            # Calculate relative information
            x_base = row_base['x']
            y_base = row_base['y']

            # Relative information for adjacent following vehicle
            x_rel_following = row_following['x'] - x_base
            y_rel_following = row_following['y'] - y_base
            distance_following = np.sqrt(x_rel_following ** 2 + y_rel_following ** 2)

            # Relative information for adjacent preceding vehicle
            x_rel_preceding = row_preceding['x'] - x_base
            y_rel_preceding = row_preceding['y'] - y_base
            distance_preceding = np.sqrt(x_rel_preceding ** 2 + y_rel_preceding ** 2)

            # Relative information for original preceding vehicle
            x_rel_original = row_original['x'] - x_base
            y_rel_original = row_original['y'] - y_base
            distance_original = np.sqrt(x_rel_original ** 2 + y_rel_original ** 2)

            # Get driving direction of base vehicle
            base_direction = row_base['driving_direction']

            # Verify other vehicles have same direction as base vehicle
            if row_following['driving_direction'] != base_direction:
                print(f"Warning: Following vehicle direction mismatch - base:{base_direction}, following:{row_following['driving_direction']}")
            if row_preceding['driving_direction'] != base_direction:
                print(f"Warning: Adjacent preceding vehicle direction mismatch - base:{base_direction}, preceding:{row_preceding['driving_direction']}")
            if row_original['driving_direction'] != base_direction:
                print(f"Warning: Original preceding vehicle direction mismatch - base:{base_direction}, original:{row_original['driving_direction']}")

            # Calculate adjusted lane difference
            if base_direction == 2:
                # Direction 2: Keep original
                adjusted_following_laneId_diff = row_following['laneId'] - row_base['laneId']
                adjusted_preceding_laneId_diff = row_preceding['laneId'] - row_base['laneId']
                adjusted_original_laneId_diff = row_original['laneId'] - row_base['laneId']
            else:
                # Direction 1: Take opposite sign
                adjusted_following_laneId_diff = -(row_following['laneId'] - row_base['laneId'])
                adjusted_preceding_laneId_diff = -(row_preceding['laneId'] - row_base['laneId'])
                adjusted_original_laneId_diff = -(row_original['laneId'] - row_base['laneId'])

            combined_row = {
                'scene': scene_str,
                'frame': row_base['frame'],

                # Base vehicle
                'base_id': row_base['id'],
                'base_x': x_base,
                'base_y': y_base,
                'base_xVelocity': row_base['xVelocity'],
                'base_laneId': row_base['laneId'],
                'base_driving_direction': base_direction,

                # Adjacent following vehicle
                'following_id': row_following['id'],
                'following_x_rel': x_rel_following,
                'following_y_rel': y_rel_following,
                'following_distance': distance_following,
                'following_xVelocity': row_following['xVelocity'],
                'following_laneId_diff': row_following['laneId'] - row_base['laneId'],
                'following_adjusted_laneId_diff': adjusted_following_laneId_diff,
                'following_driving_direction': row_following['driving_direction'],

                # Adjacent preceding vehicle
                'preceding_id': row_preceding['id'],
                'preceding_x_rel': x_rel_preceding,
                'preceding_y_rel': y_rel_preceding,
                'preceding_distance': distance_preceding,
                'preceding_xVelocity': row_preceding['xVelocity'],
                'preceding_laneId_diff': row_preceding['laneId'] - row_base['laneId'],
                'preceding_adjusted_laneId_diff': adjusted_preceding_laneId_diff,
                'preceding_driving_direction': row_preceding['driving_direction'],

                # Original preceding vehicle
                'original_id': row_original['id'],
                'original_x_rel': x_rel_original,
                'original_y_rel': y_rel_original,
                'original_distance': distance_original,
                'original_xVelocity': row_original['xVelocity'],
                'original_laneId_diff': row_original['laneId'] - row_base['laneId'],
                'original_adjusted_laneId_diff': adjusted_original_laneId_diff,
                'original_driving_direction': row_original['driving_direction'],
            }

            combined_data.append(combined_row)

        # Create DataFrame and save
        combined_df = pd.DataFrame(combined_data)
        combined_df.to_csv(output_path, index=False)

        # Collect statistics
        stats = {
            'scene': scene_str,
            'total_rows': len(combined_df),
            'base_vehicle_id': combined_df['base_id'].iloc[0] if len(combined_df) > 0 else 'N/A',
            'base_direction': combined_df['base_driving_direction'].iloc[0] if len(combined_df) > 0 else 'N/A',
            'frame_range': f"{combined_df['frame'].min()}-{combined_df['frame'].max()}" if len(
                combined_df) > 0 else 'N/A',
            'direction_consistency_check': all([
                combined_df['base_driving_direction'].iloc[0] == combined_df['following_driving_direction'].iloc[0],
                combined_df['base_driving_direction'].iloc[0] == combined_df['preceding_driving_direction'].iloc[0],
                combined_df['base_driving_direction'].iloc[0] == combined_df['original_driving_direction'].iloc[0]
            ]) if len(combined_df) > 0 else 'N/A'
        }

        print(f"Completed! Saved to: {os.path.basename(output_path)}")
        print(f"Data rows: {len(combined_df)}")
        print(f"Base vehicle direction: {stats['base_direction']}")
        print(f"Direction consistency check: {stats['direction_consistency_check']}")

        return combined_df, stats

    except Exception as e:
        print(f"Processing failed: {e}")
        return None, f"Scene {scene_str}: {str(e)}"

# 3. Merge All Scene Data
def merge_all_scenes():
    """Merge intermediate files from all scenes"""
    print("\n" + "=" * 60)
    print("Starting to merge all scene files...")
    print("=" * 60)

    # Get all scene files
    all_files = os.listdir(intermediate_output_dir)
    scene_files = [f for f in all_files if f.startswith('scene') and f.endswith('_vehicles_initial_state_combined.csv')]

    if not scene_files:
        scene_files = [f for f in all_files if f.startswith('scene') and f.endswith('_combined.csv')]

    print(f"Found {len(scene_files)} scene files")

    # Read and merge
    all_data = []
    scene_stats = []

    for filename in sorted(scene_files):
        file_path = os.path.join(intermediate_output_dir, filename)

        try:
            df = pd.read_csv(file_path)

            # Extract scene number
            scene_num = filename.split('_')[0].replace('scene', '')

            # Ensure scene column exists
            if 'scene' not in df.columns:
                df['scene'] = scene_num

            # Add filename
            df['source_file'] = filename
            all_data.append(df)

            # Collect statistics
            stats = {
                'scene': scene_num,
                'filename': filename,
                'rows': len(df),
                'frames': df['frame'].nunique() if 'frame' in df.columns else 0,
                'base_id': df['base_id'].iloc[0] if 'base_id' in df.columns and len(df) > 0 else 'N/A'
            }
            scene_stats.append(stats)

            print(f"Read: {filename} ({len(df)} rows)")

        except Exception as e:
            print(f"Failed to read file {filename}: {e}")

    # Merge all data
    if all_data:
        merged_df = pd.concat(all_data, ignore_index=True)

        # Save final result
        merged_df.to_csv(final_output_file, index=False)

        print(f"\nMerge completed!")
        print(f"Total rows: {len(merged_df)}")
        print(f"Total columns: {len(merged_df.columns)}")
        print(f"Saved to: {final_output_file}")

        # Detailed statistics
        print(f"\nDetailed statistics:")
        print(f"Number of scenes: {merged_df['scene'].nunique()}")
        print(f"Number of files: {merged_df['source_file'].nunique()}")

        if 'base_id' in merged_df.columns:
            print(f"Unique base vehicles: {merged_df['base_id'].nunique()}")

        if 'frame' in merged_df.columns:
            print(f"Total frame range: {merged_df['frame'].min()} - {merged_df['frame'].max()}")

        # Save scene statistics
        stats_df = pd.DataFrame(scene_stats)
        stats_file = final_output_file.replace('.csv', '_scene_stats.csv')
        stats_df.to_csv(stats_file, index=False)
        print(f"\nScene statistics saved to: {stats_file}")

        return merged_df
    else:
        print("No scene files found!")
        return None


# 4. main
def main():
    print("=" * 80)
    print("HIGH-D Dataset Vehicle Trajectory Merge Processor")
    print("=" * 80)

    # Store processing results
    successful_scenes = []
    failed_scenes = []
    all_stats = []

    # Process all scenes
    print(f"\nStarting batch processing of 60 scenes...")
    print(f"Intermediate output directory: {intermediate_output_dir}")

    for scene_num in range(1, 61):
        result, stats_or_error = process_single_scene(scene_num)

        if result is not None:
            successful_scenes.append(f"{scene_num:02d}")
            all_stats.append(stats_or_error)
        else:
            failed_scenes.append(stats_or_error)

    print(f"\n{'=' * 60}")
    print(f"Successfully processed: {len(successful_scenes)} scenes")
    print(f"Failed: {len(failed_scenes)} scenes")

    if failed_scenes:
        print("\nFailed scenes:")
        for fail in failed_scenes:
            print(f" - {fail}")

    # 合并所有场景
    print(f"\n{'=' * 60}")
    print(f"Final output file: {final_output_file}")

    merged_result = merge_all_scenes()

    print(f"\n{'=' * 80}")
    print("Processing completed!")
    print(f"Successful scenes: {len(successful_scenes)}")
    print(f"Failed scenes: {len(failed_scenes)}")

    if merged_result is not None:
        print(f"Final merged file: {final_output_file}")
        print(f"Total data volume: {len(merged_result)} rows")

    print("=" * 80)

main()