import pandas as pd
import numpy as np
from filterpy.monte_carlo import systematic_resample

# Load the dataset
df = pd.read_csv("subsets_by_date/2024-01-01/2024-04-02.csv", delimiter=",")  # Adjust delimiter if needed

# Convert timestamp column to datetime
df["DatumZeit"] = pd.to_datetime(df["DatumZeit"], format="%Y-%m-%d %H:%M:%S.%f")

# Get time difference from the first timestamp in seconds
df["time_seconds"] = (df["DatumZeit"] - df["DatumZeit"].iloc[0]).dt.total_seconds()

# Extract latitude and longitude
gps_lat = df["GPS_lat"].values  # Adjust column names if needed
gps_lon = df["GPS_lon"].values
time = df["time_seconds"].values


# Particle Filter Function
def particle_filter(data, N=1000):
    particles = np.random.normal(data[0], 1, size=N)
    weights = np.ones(N) / N
    filtered = []

    for z in data:
        particles += np.random.normal(0, 0.5, N)  # Motion model
        weights *= np.exp(-0.5 * ((particles - z) / 1.0) ** 2)
        weights += 1.e-300
        weights /= np.sum(weights)
        indexes = systematic_resample(weights)
        particles = particles[indexes]
        filtered.append(np.mean(particles))

    return np.array(filtered)


# Apply particle filter to latitude and longitude separately
df["GPS_lat_filtered"] = particle_filter(gps_lat)
df["GPS_lon_filtered"] = particle_filter(gps_lon)

# Save the results to a new CSV file
df.to_csv("gps_data_filtered.csv", index=False, sep=",")

print("Filtered GPS data saved as 'gps_data_filtered.csv'.")
