"""
Extract the initial state information of each vehicle at the beginning of lane change.
For each detected lane change event, extract the status information of all vehicles involved at the beginning of the event.
"""

import pandas as pd
import os
import numpy as np
import re

input_folders = [
    './output/surround_data/adj_preceding_data',
    './output/surround_data/adj_following_data',
    './output/surround_data/original_preceding_data',
    './output/surround_data/changing_vehicle_data'
]

base_output_dir = './output/surround_data/surround_data_initial'
os.makedirs(base_output_dir, exist_ok=True)

# Create corresponding output subfolders for each input folder
output_folders = []
for folder in input_folders:
    folder_name = os.path.basename(folder)
    output_folder = os.path.join(base_output_dir, folder_name)
    output_folders.append(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    print(f"Created output directory: {output_folder}")

print(f"\nFound {len(input_folders)} input folders")
print(f"Created {len(output_folders)} output subfolders")


# Helper function: Extract numbers from strings for sorting
def extract_number_for_sorting(value):
    if pd.isna(value):
        return float('inf')

    # If numeric type, return directly
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    # If string, try to extract numbers
    if isinstance(value, str):
        # Try to match numeric parts
        numbers = re.findall(r'\d+', value)
        if numbers:
            # Take the last continuous number
            return float(numbers[-1])
        else:
            # If no numbers, sort by string
            return float('inf')

    # Other types, try to convert to number
    try:
        return float(value)
    except:
        return float('inf')


# Initialize statistics
total_stats = {
    'total_files': 0,
    'success_files': 0,
    'failed_files': 0,
    'total_original_rows': 0,
    'total_initial_rows': 0,
    'total_multi_event_vehicles': 0,
    'files_without_event_col': 0
}

for folder_idx, (input_folder, output_folder) in enumerate(zip(input_folders, output_folders), 1):
    print(f"\n{'=' * 80}")
    print(f"[{folder_idx}/{len(input_folders)}] Processing folder: {os.path.basename(input_folder)}")
    print(f"Input path: {input_folder}")
    print(f"Output path: {output_folder}")

    # Get all CSV files in this folder
    all_files = os.listdir(input_folder)
    csv_files = [f for f in all_files if f.endswith('.csv') and not f.startswith('processing_summary')]

    print(f"Found {len(csv_files)} CSV files")
    total_stats['total_files'] += len(csv_files)

    # Process each file
    folder_summary = []
    folder_stats = {
        'files_processed': 0,
        'files_success': 0,
        'files_failed': 0,
        'total_original_rows': 0,
        'total_initial_rows': 0,
        'multi_event_vehicles': 0
    }

    for file_idx, csv_file in enumerate(csv_files, 1):
        print(f"\n[{file_idx}/{len(csv_files)}] Processing file: {csv_file}")

        file_path = os.path.join(input_folder, csv_file)
        try:
            df = pd.read_csv(file_path)
            folder_stats['files_processed'] += 1
        except Exception as e:
            print(f"Error: Unable to read file {csv_file}: {e}")
            folder_stats['files_failed'] += 1
            total_stats['failed_files'] += 1
            continue

        original_rows = len(df)
        folder_stats['total_original_rows'] += original_rows
        total_stats['total_original_rows'] += original_rows

        print(f"Original data: {original_rows} rows, {df.shape[1]} columns")

        # Check necessary columns
        # 1. Determine vehicle ID column
        id_columns = ['id', 'vehicleId', 'vehicle_id', 'Vehicle_ID', 'vehicleID']
        id_col = None
        for col in id_columns:
            if col in df.columns:
                id_col = col
                break

        if id_col is None:
            print(f"Warning: Vehicle ID column not found, skipping this file")
            print(f"Available columns: {list(df.columns)}")
            folder_stats['files_failed'] += 1
            total_stats['failed_files'] += 1
            continue

        # 2. Determine frame column
        frame_columns = ['frame', 'frameId', 'frame_id', 'Frame']
        frame_col = None
        for col in frame_columns:
            if col in df.columns:
                frame_col = col
                break

        if frame_col is None:
            print(f"Warning: Frame column not found, skipping this file")
            folder_stats['files_failed'] += 1
            total_stats['failed_files'] += 1
            continue

        # 3. Determine event column - Critical modification: must find event column
        event_columns = ['event', 'Event', 'event_id', 'eventId', 'order', 'sequence', 'scenario', 'scenario_id']
        event_col = None
        for col in event_columns:
            if col in df.columns:
                event_col = col
                break

        if event_col is None:
            print(f"Warning: Event column not found! Skipping this file")
            print(f"Available columns: {list(df.columns)}")
            folder_stats['files_failed'] += 1
            total_stats['failed_files'] += 1
            total_stats['files_without_event_col'] += 1
            continue

        print(f"Using columns: ID={id_col}, Frame={frame_col}, Event={event_col}")

        # Statistics
        unique_vehicles = df[id_col].nunique()
        unique_events = df[event_col].nunique()

        # Check event column data type and sample values
        event_sample_values = df[event_col].dropna().unique()[:5]
        print(f"Unique vehicles: {unique_vehicles}, Unique events: {unique_events}")
        print(f"Event column sample values: {event_sample_values}")
        print(f"Event column data type: {df[event_col].dtype}")

        # Method: Group by event and vehicle ID, find minimum frame for each combination
        initial_states = []

        # Group by event and vehicle ID
        grouped = df.groupby([event_col, id_col])

        # For each (event, vehicle) combination, find minimum frame
        for (event_val, vehicle_id), group in grouped:
            min_frame = group[frame_col].min()

            # Get records corresponding to minimum frame
            min_frame_records = group[group[frame_col] == min_frame]

            if len(min_frame_records) > 0:
                # If multiple records (e.g., multiple lanes), take the first one
                initial_states.append(min_frame_records.iloc[0])

        # Create initial state DataFrame
        if initial_states:
            df_initial = pd.DataFrame(initial_states)
            print(f"Sorting by event column {event_col} numerically...")

            # Method 1: If event column is already numeric type, sort directly
            if pd.api.types.is_numeric_dtype(df_initial[event_col]):
                df_initial = df_initial.sort_values(by=event_col)
                print(f"Event column is numeric type, sorting directly")
            else:
                # Method 2: Create temporary column for numeric sorting
                df_initial['_event_numeric'] = df_initial[event_col].apply(extract_number_for_sorting)
                unique_numeric = df_initial['_event_numeric'].unique()
                print(f"Extracted event numeric values: {sorted(unique_numeric[:10])}")

                # Sort by numeric value
                df_initial = df_initial.sort_values(by='_event_numeric')

                # Remove temporary column
                df_initial = df_initial.drop(columns=['_event_numeric'])

            # Reset index
            df_initial = df_initial.reset_index(drop=True)

            # Result statistics
            initial_rows = len(df_initial)
            events_in_initial = df_initial[event_col].nunique()
            vehicles_in_initial = df_initial[id_col].nunique()

            folder_stats['total_initial_rows'] += initial_rows
            total_stats['total_initial_rows'] += initial_rows

            print(f"Extracted {initial_rows} initial state records")
            print(f"Involves {events_in_initial} events, {vehicles_in_initial} vehicles")

            # Check if vehicles appear in multiple events
            vehicle_event_counts = df_initial.groupby(id_col)[event_col].nunique()
            multi_event_vehicles = vehicle_event_counts[vehicle_event_counts > 1]
            multi_event_count = len(multi_event_vehicles)

            folder_stats['multi_event_vehicles'] += multi_event_count
            total_stats['total_multi_event_vehicles'] += multi_event_count

            if multi_event_count > 0:
                print(f"{multi_event_count} vehicles appear in multiple events")

            # 保存文件
            output_filename = csv_file.replace('.csv', '_initial.csv')
            output_path = os.path.join(output_folder, output_filename)

            try:
                df_initial.to_csv(output_path, index=False)
                print(f"Saved to: {output_filename}")

                # Record summary
                folder_summary.append({
                    'scene': csv_file.split('_')[0] if '_' in csv_file else csv_file,
                    'original_file': csv_file,
                    'output_file': output_filename,
                    'original_rows': original_rows,
                    'initial_rows': initial_rows,
                    'original_vehicles': unique_vehicles,
                    'initial_vehicles': vehicles_in_initial,
                    'original_events': unique_events,
                    'initial_events': events_in_initial,
                    'multi_event_vehicles': multi_event_count,
                    'event_column': event_col,
                    'id_column': id_col,
                    'frame_column': frame_col,
                    'status': 'Success'
                })

                folder_stats['files_success'] += 1
                total_stats['success_files'] += 1

            except Exception as e:
                print(f"Failed to save file: {e}")
                folder_stats['files_failed'] += 1
                total_stats['failed_files'] += 1

                folder_summary.append({
                    'scene': csv_file.split('_')[0] if '_' in csv_file else csv_file,
                    'original_file': csv_file,
                    'output_file': 'Save failed',
                    'original_rows': original_rows,
                    'initial_rows': initial_rows,
                    'original_vehicles': unique_vehicles,
                    'initial_vehicles': vehicles_in_initial,
                    'original_events': unique_events,
                    'initial_events': events_in_initial,
                    'multi_event_vehicles': multi_event_count,
                    'event_column': event_col,
                    'id_column': id_col,
                    'frame_column': frame_col,
                    'status': f'Failed - {str(e)[:50]}'
                })
        else:
            print(f"Warning: No initial state data extracted")
            folder_stats['files_failed'] += 1
            total_stats['failed_files'] += 1

            folder_summary.append({
                'scene': csv_file.split('_')[0] if '_' in csv_file else csv_file,
                'original_file': csv_file,
                'output_file': 'None',
                'original_rows': original_rows,
                'initial_rows': 0,
                'original_vehicles': unique_vehicles,
                'initial_vehicles': 0,
                'original_events': unique_events,
                'initial_events': 0,
                'multi_event_vehicles': 0,
                'event_column': event_col,
                'id_column': id_col,
                'frame_column': frame_col,
                'status': 'Failed - No data'
            })

    # Save processing summary for this folder
    if folder_summary:
        summary_df = pd.DataFrame(folder_summary)
        summary_file = os.path.join(output_folder, f"processing_summary.csv")
        summary_df.to_csv(summary_file, index=False)

        print(f"\n{os.path.basename(input_folder)} folder processing completed:")
        print(f"Files processed: {folder_stats['files_processed']}")
        print(f"Successfully processed: {folder_stats['files_success']} files")
        print(f"Failed processing: {folder_stats['files_failed']} files")
        print(f"Total original data rows: {folder_stats['total_original_rows']:,}")
        print(f"Total initial state rows: {folder_stats['total_initial_rows']:,}")
        print(f"Total multi-event vehicles: {folder_stats['multi_event_vehicles']}")

    if folder_stats['total_original_rows'] > 0:
            retention_rate = (folder_stats['total_initial_rows'] / folder_stats['total_original_rows']) * 100
            print(f"Data retention rate: {retention_rate:.2f}%")
    else:
        print(f"\n{os.path.basename(input_folder)} folder: No files processed")

# Generate final summary report
print(f"\nOverall statistics:")
print(f"Total files processed: {total_stats['total_files']}")
print(f"Successful files: {total_stats['success_files']}")
print(f"Failed files: {total_stats['failed_files']}")
print(f"Files without event column: {total_stats['files_without_event_col']}")
print(f"Total original data rows: {total_stats['total_original_rows']:,}")
print(f"Total initial state rows: {total_stats['total_initial_rows']:,}")
print(f"Total multi-event vehicles: {total_stats['total_multi_event_vehicles']}")

if total_stats['total_original_rows'] > 0:
    overall_retention = (total_stats['total_initial_rows'] / total_stats['total_original_rows']) * 100
    print(f"Overall data retention rate: {overall_retention:.2f}%")

print(f"\nProcessing completed!")