import pandas as pd
import numpy as np
from haversine import haversine

# 1️⃣ Load the data
df = pd.read_csv("2024-04-02.csv", delimiter=',', encoding='utf-8')

# 2️⃣ Convert `DatumZeit` to datetime format
df['DatumZeit'] = pd.to_datetime(df['DatumZeit'])

# 3️⃣ Sort data by time (just in case)
df = df.sort_values(by='DatumZeit')

# 4️⃣ Define spatial distance threshold (meters)
eps_space = 50.0  # If train moves more than this, it's considered "moving"
min_stop_points = 5  # Minimum number of consecutive points needed to form a stop

# 5️⃣ Initialize clustering
cluster_id = -1  # -1 means "not a stop"
df['cluster_label'] = -1  # Default: no cluster

# 6️⃣ Go through data sequentially and detect stops
previous_point = None
stop_start_index = None

for i, row in df.iterrows():
    current_point = (row['GPS_lat'], row['GPS_lon'])

    if previous_point is None:
        # First point, assume it's moving for now
        stop_start_index = i
    else:
        distance = haversine(previous_point, current_point) * 1000  # Convert km to meters

        if distance > eps_space:
            # Train moved -> check if we were in a stop
            if stop_start_index is not None and (i - stop_start_index) >= min_stop_points:
                # Assign a cluster ID to the stop period
                cluster_id += 1
                df.loc[stop_start_index:i-1, 'cluster_label'] = cluster_id

            # Reset stop detection
            stop_start_index = i
        else:
            # Train is still within stopping range
            if stop_start_index is None:
                stop_start_index = i  # Mark start of a new stop

    # Update previous point
    previous_point = current_point

# 7️⃣ Save output CSV
df.to_csv("2024-04-02_clustered.csv", index=False, encoding='utf-8')

print(f"Clustering done. Results saved in '2024-04-02_clustered.csv'.")
