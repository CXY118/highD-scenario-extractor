"""
Extract the initial state information of the victim at the start of a cut-in.

For each detected cut-in event, extract the vehicle's state information at the moment the cut-in maneuver begins.
"""

import pandas as pd
import os

# 1. Set base data directories
input_folder = './output/victim_trajectories'             # Directory for saving victim vehicle trajectories
output_folder = './output/victim_initial_states'          # Initial state information

os.makedirs(output_folder, exist_ok=True)

# 2. Iterate through all CSV files in input folder
for file_name in os.listdir(input_folder):
    if file_name.endswith('.csv'):
        input_path = os.path.join(input_folder, file_name)

        try:
            df = pd.read_csv(input_path)
            print(f"Processing: {file_name}")
            print(f"Total rows: {len(df)}")

            # Check if cut_in_event_id column exists
            if 'cut_in_event_id' in df.columns:
                print(f"Cut-in events: {df['cut_in_event_id'].nunique()}")

                # Group by cut-in event ID, take first row for each event
                initial_states = []

                for event_id in sorted(df['cut_in_event_id'].unique()):
                    event_df = df[df['cut_in_event_id'] == event_id].sort_values('frame')

                    if not event_df.empty:
                        # Take first row of this event
                        first_row = event_df.iloc[0]
                        initial_states.append(first_row)

                # Create result DataFrame, preserve original column order
                result_df = pd.DataFrame(initial_states)
                result_df = result_df[df.columns]

            else:
                print(f"Warning: File missing 'cut_in_event_id' column, grouping by vehicle ID and frame continuity")

                # Group by vehicle ID
                df = df.sort_values(['id', 'frame'])
                initial_states = []

                for vehicle_id in df['id'].unique():
                    vehicle_df = df[df['id'] == vehicle_id]

                    if len(vehicle_df) == 1:
                        initial_states.append(vehicle_df.iloc[0])
                    else:
                        # Find frame discontinuity points (identify separate events based on frame continuity)
                        frames = vehicle_df['frame'].values
                        frame_diffs = frames[1:] - frames[:-1]

                        break_points = [0]
                        for i, diff in enumerate(frame_diffs, 1):
                            if diff > 1:
                                break_points.append(i)

                        # Take first row of each continuous segment
                        for i in range(len(break_points)):
                            start_idx = break_points[i]
                            end_idx = break_points[i + 1] if i + 1 < len(break_points) else len(vehicle_df)
                            segment = vehicle_df.iloc[start_idx:end_idx]
                            if not segment.empty:
                                initial_states.append(segment.iloc[0])

                # Create result DataFrame, preserve original column order
                result_df = pd.DataFrame(initial_states)
                result_df = result_df[df.columns]

            print(f"Extracted {len(result_df)} initial states")
            print(f"Involves {result_df['id'].nunique()} unique vehicle IDs")

            # Check if any vehicle has multiple events
            vehicle_counts = result_df['id'].value_counts()
            multi_event_vehicles = vehicle_counts[vehicle_counts > 1]
            if len(multi_event_vehicles) > 0:
                print(f"{len(multi_event_vehicles)} vehicles have multiple events")
                for vehicle_id, count in multi_event_vehicles.items():
                    # Get all event frames for this vehicle
                    frames = result_df[result_df['id'] == vehicle_id]['frame'].tolist()
                    print(f"Vehicle {vehicle_id}: {count} events, initial frames: {frames}")

            # Save to output folder
            output_path = os.path.join(output_folder, file_name)
            result_df.to_csv(output_path, index=False)
            print(f"Saved to: {output_path}")
            print("-" * 50)

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
            continue

print(f"Processing complete! All results saved to: {output_folder}")
