import pandas as pd
import numpy as np
from datetime import datetime
from geopy.distance import geodesic

# Load a chunk of the data
file_path = "your_file.csv"  # Update with your file path
chunksize = 100000  # Load in chunks to handle large files


def process_chunk(chunk):
    print("Processing chunk...")

    # Convert DatumZeit to datetime
    chunk['DatumZeit'] = pd.to_datetime(chunk['DatumZeit'], format='%Y-%m-%d %H:%M:%S.%f')
    print("Converted DatumZeit to datetime format.")

    # Sort by time (just in case)
    chunk = chunk.sort_values(by='DatumZeit')
    print("Sorted chunk by DatumZeit.")

    # Compute time difference in seconds
    chunk['time_diff'] = chunk['DatumZeit'].diff().dt.total_seconds()
    print("Computed time differences.")

    # Compute heading change (yaw angle)
    def compute_heading(lat1, lon1, lat2, lon2):
        delta_lon = np.radians(lon2 - lon1)
        lat1, lat2 = np.radians(lat1), np.radians(lat2)
        x = np.sin(delta_lon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
        heading = np.arctan2(x, y)
        return np.degrees(heading)

    chunk['heading'] = compute_heading(chunk['gps_lat'].shift(), chunk['gps_lon'].shift(), chunk['gps_lat'],
                                       chunk['gps_lon'])
    print("Computed headings.")

    # Compute yaw rate (change in heading over time)
    chunk['yaw_rate'] = chunk['heading'].diff() / chunk['time_diff']
    print("Computed yaw rate.")

    # Drop NaN values from diffs
    chunk = chunk.dropna()
    print("Dropped NaN values.")

    print("Chunk processing complete!")
    return chunk[['gps_lat', 'gps_lon', 'time_diff', 'heading', 'yaw_rate']]


# Process and save the first chunk to test
chunk = next(pd.read_csv(file_path, chunksize=chunksize))
processed_chunk = process_chunk(chunk)
processed_chunk.to_csv("processed_sample.csv", index=False)

print("Preprocessing complete! Saved sample to 'processed_sample.csv'")
