"""
Extract the initial state information of the vehicle at the start of a cut-in.

For each detected cut-in event, extract the vehicle's state information at the moment the cut-in maneuver begins.
"""

import pandas as pd
import os

# 1. Set base data directories
input_folder = './output/cutin_trajectories'                # Directory for saving cut-in vehicle trajectories
#input_folder = "./output/cutin_trajectories_filtered"     # Directory for speed-filtered results (if speed filtered)
output_folder = './output/cutin_initial_states'          # Initial state information

os.makedirs(output_folder, exist_ok=True)

# 2. Iterate through all CSV files in input folder
for file_name in os.listdir(input_folder):
    if file_name.endswith('.csv'):
        input_path = os.path.join(input_folder, file_name)

        try:
            df = pd.read_csv(input_path)
            print(f"Processing cut-in trajectory file: {file_name}")
            print(f"Rows: {len(df)}")
            print(f"Cut-in vehicles: {df['id'].nunique()}")

            # Group by cut-in vehicle ID
            initial_states = []

            for vehicle_id in sorted(df['id'].unique()):
                vehicle_df = df[df['id'] == vehicle_id].sort_values('frame')

                if len(vehicle_df) == 1:
                    # If only one row, use it as initial state
                    initial_states.append(vehicle_df.iloc[0])
                else:
                    # This trajectory contains a complete cut-in event
                    # Take the first row as initial state
                    initial_states.append(vehicle_df.iloc[0])

                    # Debug info: show time range of this cut-in event
                    start_frame = vehicle_df['frame'].min()
                    end_frame = vehicle_df['frame'].max()
                    print(f"Vehicle {vehicle_id}: frame range {start_frame}-{end_frame} ({len(vehicle_df)} frames)")

            # Create result DataFrame, preserve original column order
            result_df = pd.DataFrame(initial_states)
            result_df = result_df[df.columns]

            print(f"Extracted {len(result_df)} cut-in vehicle initial states")
            print(f"Involves {result_df['id'].nunique()} unique cut-in vehicles")

            # Save to output folder
            output_file_name = file_name.replace('filtered', 'initial_state')
            output_path = os.path.join(output_folder, output_file_name)
            result_df.to_csv(output_path, index=False)
            print(f"Saved to: {output_path}")
            print("-" * 50)

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
            continue

print(f"Processing complete! All results saved to: {output_folder}")
