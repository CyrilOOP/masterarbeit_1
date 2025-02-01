import pandas as pd
import numpy as np
from geopy.distance import geodesic


def filter_unrealistic_gps(df, threshold_multiplier=1.2):
    """
    Filters out GPS points that cannot be physically reached based on speed and acceleration using vectorized operations.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'GPS_lat', 'GPS_lon', 'dt', 'Geschwindigkeit in m/s ', and 'Beschleunigung in m/s2'.
    threshold_multiplier (float): Factor to allow some tolerance in filtering.

    Returns:
    pd.DataFrame: Filtered DataFrame with unrealistic GPS points removed.
    """
    # Compute time differences
    delta_t = df["dt"].diff().shift(-1)

    # Compute max possible distance
    max_possible_distance = (df["Geschwindigkeit in m/s"] * delta_t) + (
                0.5 * df["Beschleunigung in m/s2"] * (delta_t ** 2))
    max_possible_distance *= threshold_multiplier

    # Compute actual GPS distances using vectorized operations
    lat_lon_pairs = list(zip(df["GPS_lat"], df["GPS_lon"]))
    actual_distances = np.array(
        [geodesic(lat_lon_pairs[i], lat_lon_pairs[i + 1]).meters if i < len(lat_lon_pairs) - 1 else 0 for i in
         range(len(lat_lon_pairs))])

    # Identify valid points
    valid_points = actual_distances <= max_possible_distance.fillna(float("inf"))
    valid_points[0] = True  # Always keep the first point

    return df[valid_points].reset_index(drop=True)


if __name__ == "__main__":

    df = pd.read_csv("2024-04-02_time.csv")
    filtered_df = filter_unrealistic_gps(df, threshold_multiplier=1.2)
    filtered_df.to_csv("filtered_gps_data.csv", index=False)
