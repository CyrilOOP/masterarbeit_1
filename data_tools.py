import tkinter as tk
from tkinter import ttk
from typing import Dict, Any, Optional  # Add this line to fix the error

from geopy.distance import geodesic
from sklearn.cluster import DBSCAN

from csv_tools import  csv_select_gps_columns
import numpy as np
import pandas as pd
from pykalman import KalmanFilter
from pyproj import Transformer
from filterpy.monte_carlo import systematic_resample
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


# --------------------------------------------------------------------------
# Simplified Function: Convert to Planar Coordinates (UTM)
# --------------------------------------------------------------------------
def data_convert_to_planar(df: pd.DataFrame, config: Dict[str, str]) -> pd.DataFrame:
    """
    Convert latitude and longitude to planar coordinates (UTM).
    Uses the helper to select which GPS columns to use.

    Args:
        df: The input DataFrame containing GPS data.
        config: A configuration dictionary. It should at least contain the raw column names if no smoothed columns are available.

    Returns:
        The DataFrame with added planar coordinates (x, y) and a column 'selected_smoothing_method' indicating which method was used.
    """
    # Use the helper to select GPS columns.
    lat_input, lon_input = csv_select_gps_columns(
        df,
        title="Select GPS Data for Planar Conversion",
        prompt="Select the GPS data to use for planar conversion:"
    )
    print(f"Using input columns: {lat_input} and {lon_input}")

    # Here, you might decide to record which method was selected.
    # For example, if the key contains "smoothed", extract the method; otherwise, mark as "raw".
    if "smoothed" in lat_input:
        selected_method = lat_input.split("smoothed_")[-1]
    else:
        selected_method = "raw"

    # Create a transformer (example: WGS84 to UTM zone 33N; adjust as needed)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)

    # Perform coordinate transformation (note that the transformer expects lon first)
    x, y = transformer.transform(df[lon_input].values, df[lat_input].values)

    # Add planar coordinates and the method to the DataFrame.
    df["x"] = x
    df["y"] = y
    df["selected_smoothing_method"] = selected_method

    return df


def data_filter_points_by_distance(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Filter points by a minimum distance using columns and settings from config.

    Args:
        df: The input DataFrame containing coordinates.
        config: Dictionary containing configuration values:
            - "x_col": Column name for x-coordinates.
            - "y_col": Column name for y-coordinates.
            - "min_distance": Minimum distance to retain a point.

    Returns:
        Modified DataFrame with points spaced by at least the minimum distance
        and a new column 'min_distance' indicating the distance used for filtering.

    Raises:
        ValueError: If required columns are missing or the DataFrame is empty.
    """
    x_col = config["x_col"]
    y_col = config["y_col"]
    min_distance = config.get("min_distance")

    # Check if the DataFrame is empty
    if df.empty:
        return df

    # Validate required columns
    for col in [x_col, y_col]:
        if col not in df.columns:
            raise ValueError(
                f"Missing column '{col}'. Ensure planar coordinates exist before calling this function."
            )

    # Extract coordinates as a NumPy array
    coords = df[[x_col, y_col]].to_numpy()

    # Initialize list of retained indices
    retained_indices = [0]  # Always keep the first point
    last_retained_point = coords[0]  # Start with the first point

    # Iterate through the remaining points
    for i in range(1, len(coords)):
        distance = np.linalg.norm(coords[i] - last_retained_point)  # Distance to last retained point
        if distance >= min_distance:
            retained_indices.append(i)  # Retain the current point
            last_retained_point = coords[i]  # Update the last retained point

    # Filter the DataFrame
    df = df.iloc[retained_indices].reset_index(drop=True)

    # Add min_distance as a new column for all rows
    df['min_distance'] = min_distance

    return df


def parse_time_and_compute_dt(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Parse the given datetime column as pandas datetime and compute delta time (in seconds).

    Args:
        df: Input DataFrame.
        config: Configuration dictionary containing:
            - "datetime_col": Name of the column containing datetime information.

    Returns:
        A copy of the DataFrame with a new column 'dt' containing time differences in seconds.

    Raises:
        ValueError: If the datetime column cannot be parsed.
    """
    datetime_col = config.get("datetime_col", "DatumZeit")
    df = df.copy()

    # Convert the column to datetime
    try:
        df[datetime_col] = pd.to_datetime(df[datetime_col])
    except Exception as e:
        raise ValueError(f"Error parsing datetime column '{datetime_col}': {e}")

    # Compute the difference in timestamps
    df["dt"] = df[datetime_col].diff().dt.total_seconds()

    return df

def data_compute_heading_from_xy(df: pd.DataFrame, config: Dict[str, str]) -> pd.DataFrame:
    """
    Compute heading for each row based on consecutive (x, y) points.
    Heading is computed using arctan2(dy, dx) and returned in degrees within [0, 360).

    Args:
        df: The DataFrame containing at least the x_col and y_col specified in config.
        config: Configuration dictionary containing keys:
            - "x_col": Column name for the x-coordinate (default "x").
            - "y_col": Column name for the y-coordinate (default "y").
            - "heading_col": Name of the new column for heading (default "heading_deg").

    Returns:
        The modified DataFrame with the new column for heading.

    Raises:
        ValueError: If required columns are missing.
    """
    # Extract column names from config
    x_col = config.get("x_col", "x")
    y_col = config.get("y_col", "y")
    heading_col = config.get("heading_col", "heading_deg")

    # Check if required columns exist
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Required columns '{x_col}' and/or '{y_col}' not found in DataFrame.")

    # Calculate differences
    dx = df[x_col].diff()
    dy = df[y_col].diff()

    # Compute heading in radians
    heading_rad = np.arctan2(dy, dx)  # range: [-pi, pi]

    # Convert to degrees and shift to [0, 360)
    heading_deg = np.degrees(heading_rad)
    heading_deg = (heading_deg + 360) % 360

    # Assign to the specified column
    df[heading_col] = heading_deg

    return df


def data_compute_yaw_rate_from_heading(df: pd.DataFrame, config: Dict[str, str]) -> pd.DataFrame:
    """
    Calculate yaw rate in degrees/second given an existing heading column (in degrees)
    and a time-delta column (in seconds).

    Args:
        df: The DataFrame containing heading_col and dt_col.
        config: Configuration dictionary containing keys:
            - "heading_col": Column name containing heading in degrees (default "heading_deg").
            - "dt_col": Column name containing time deltas in seconds (default "dt").

    Returns:
        The modified DataFrame with an additional column 'yaw_rate_deg_s' for yaw rate.

    Raises:
        ValueError: If required columns are missing.
    """
    # Extract column names from config
    heading_col = config.get("heading_col_for_yaw_rate_function")
    dt_col = config.get("dt_col", "dt")

    # Ensure required columns exist
    if heading_col not in df.columns or dt_col not in df.columns:
        raise ValueError(f"Required columns '{heading_col}' and/or '{dt_col}' not found in DataFrame.")

    # 1. Heading difference
    heading_diff = df[heading_col].diff()

    # 2. Wrap to [-180, 180]
    heading_diff = (heading_diff + 180) % 360 - 180

    # 3. Divide by dt => degrees/second
    dt_vals = df[dt_col]
    yaw_rate_deg_s = heading_diff / dt_vals

    # 4. Assign to a fixed column name
    df["yaw_rate_deg_s"] = -yaw_rate_deg_s ###minus sign here!!!

    return df



def data_delete_the_one_percent(df: pd.DataFrame, config: Dict[str, str]):
    # Get required config values (with defaults, if needed).
    date_col = config.get("date_column", "DatumZeit")
    yaw_rate_col = config.get("yaw_rate_column", "yaw_rate_deg_s")
    input_lower_bound = config["delete_lower_bound_percentage"]/100
    print(f'input_lower_bound: {input_lower_bound}')
    input_upper_bound = config["delete_upper_bound_percentage"]/100
    print(f'input_upper_bound: {input_upper_bound}')
    # Check if required columns exist in df
    if yaw_rate_col not in df.columns or date_col not in df.columns:
        raise ValueError(f"The required columns '{yaw_rate_col}' or '{date_col}' "
                         f"are missing from the CSV file.")

    # Calculate 1% and 99% quantiles for yaw_rate
    lower_bound = df[yaw_rate_col].quantile(input_lower_bound)
    upper_bound = df[yaw_rate_col].quantile(input_upper_bound)

    # Filter rows within the quantile range
    df = df[(df[yaw_rate_col] >= lower_bound) & (df[yaw_rate_col] <= upper_bound)]

    return df



def data_compute_heading_from_ds(df: pd.DataFrame, config: Dict[str, str]) -> pd.DataFrame:
    """
    Computes the heading (yaw angle) of a train moving along a curved path
    based on arc-length differentiation.

    Args:
        df (pd.DataFrame): A DataFrame containing x and y coordinate columns.
        config (Dict[str, str]): A configuration dictionary containing column names.

    Returns:
        pd.DataFrame: The input DataFrame with an additional column 'heading_deg_ds'.
    """
    # Extract column names from config with defaults
    x_col = config.get("x_col", "x")
    y_col = config.get("y_col", "y")
    heading_col = config.get("heading_col_ds", "heading_deg_ds")

    # Check if required columns exist
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{x_col}' and '{y_col}' columns.")

    # Compute arc length s (cumulative distance along the path)
    dx = np.diff(df[x_col])
    dy = np.diff(df[y_col])
    ds = np.sqrt(dx**2 + dy**2)  # Distance between consecutive points
    s = np.concatenate(([0], np.cumsum(ds)))  # Cumulative sum of distances

    # Compute derivatives dx/ds and dy/ds using central differences
    dx_ds = np.gradient(df[x_col], s)
    dy_ds = np.gradient(df[y_col], s)

    # Compute heading (yaw angle) using atan2(dy/ds, dx/ds)
    heading = np.degrees(np.arctan2(dy_ds, dx_ds))

    # Convert negative angles to the [0, 360] range
    heading = (heading + 360) % 360

    # Store in DataFrame
    df[heading_col] = heading

    return df


def data_kalman_on_yaw_rate(df: pd.DataFrame, config: Dict[str, str]) -> pd.DataFrame:
    """
    Apply a Kalman filter to smooth yaw rate values computed from GPS heading.

    Args:
        df: The DataFrame containing GPS-derived yaw rate.
        config: Configuration dictionary with column names.

    Returns:
        The modified DataFrame with an additional column 'yaw_rate_smooth_gps'.
    """
    yaw_col = config.get("yaw_col_for_kalman")  # GPS-based yaw rate column

    if yaw_col not in df.columns:
        raise ValueError(f"Required column '{yaw_col}' not found in DataFrame.")

    # Extract noisy yaw rate from GPS
    measurements = df[yaw_col].values

    # Kalman Filter Setup (assuming constant yaw rate model)
    kf = KalmanFilter(
        initial_state_mean=[measurements[0]],  # Start with first yaw rate value
        initial_state_covariance=[1],  # Initial uncertainty
        transition_matrices=[1],  # Yaw rate follows smooth transitions
        observation_matrices=[1],  # Direct observation model
        transition_covariance=[0.1],  # Process noise (adjustable)
        observation_covariance=[2]  # Measurement noise (adjustable)
    )

    # Apply Kalman filter
    smoothed_state_means, _ = kf.filter(measurements.reshape(-1, 1))

    # Store smoothed yaw rate in DataFrame
    df["yaw_rate_kalman"] = smoothed_state_means[:, 0]

    return df


# ============================================================================
# Example Function 1: Savitzky–Golay Smoothing on GPS Data
# ============================================================================
def data_smooth_gps_savitzky(df: pd.DataFrame,
                             smoothing_params: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    """
    Apply a Savitzky–Golay filter to GPS data.

    It first uses the helper `select_gps_columns` to determine which GPS columns to use.
    The smoothed results are stored in new columns:
        'GPS_lat_smoothed_savitzky' and 'GPS_lon_smoothed_savitzky'.

    Args:
        df: DataFrame containing GPS data.
        smoothing_params: Optional dict to override default smoothing parameters
                          (default: {"window_length": 51, "polyorder": 2})

    Returns:
        The modified DataFrame.
    """
    # Choose GPS columns (raw or preprocessed) via the helper.
    lat_input, lon_input = csv_select_gps_columns(df,
                                              title="Select GPS Data for Savitzky–Golay Smoothing",
                                              prompt="Select the GPS data to use as input for Savitzky-Golay:")
    print(f"Using input columns: {lat_input} and {lon_input}")

    # Default smoothing parameters.
    params = {"window_length": 51, "polyorder": 2}
    if smoothing_params:
        params.update(smoothing_params)

    # Check that there is enough data for the chosen window_length.
    if len(df[lat_input]) < params["window_length"]:
        raise ValueError(f"Data length in {lat_input} is less than window_length ({params['window_length']}).")

    # Apply the Savitzky–Golay filter.
    df["GPS_lat_smoothed_savitzky"] = savgol_filter(df[lat_input], params["window_length"], params["polyorder"])
    df["GPS_lon_smoothed_savitzky"] = savgol_filter(df[lon_input], params["window_length"], params["polyorder"])

    print("Savitzky–Golay smoothing applied and saved as 'GPS_lat_smoothed_savitzky' and 'GPS_lon_smoothed_savitzky'.")
    return df


# ============================================================================
# Example Function 2: Gaussian Smoothing on GPS Data
# ============================================================================
def data_smooth_gps_gaussian(df: pd.DataFrame,
                             gaussian_params: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    Apply a Gaussian filter to GPS data.

    It first uses the helper `select_gps_columns` to determine which GPS columns to use.
    The smoothed results are stored in new columns:
        'GPS_lat_smooth_gaussian' and 'GPS_lon_smooth_gaussian'.

    Args:
        df: DataFrame containing GPS data.
        gaussian_params: Optional dict to override default Gaussian parameters
                         (default: {"sigma": 2})

    Returns:
        The modified DataFrame.
    """
    # Choose GPS columns.
    lat_input, lon_input = csv_select_gps_columns(df,
                                              title="Select GPS Data for Gaussian Smoothing",
                                              prompt="Select the GPS data to use as input for Gaussian:")
    print(f"Using input columns: {lat_input} and {lon_input}")

    # Default Gaussian parameters.
    params = {"sigma": 2}
    if gaussian_params:
        params.update(gaussian_params)

    # Apply the Gaussian filter.
    df["GPS_lat_smoothed_gaussian"] = gaussian_filter1d(df[lat_input], sigma=params["sigma"])
    df["GPS_lon_smoothed_gaussian"] = gaussian_filter1d(df[lon_input], sigma=params["sigma"])

    print("Gaussian smoothing applied and saved as 'GPS_lat_smooth_gaussian' and 'GPS_lon_smooth_gaussian'.")
    return df


# ============================================================================
# Example Function 3: Particle Filter using GPS, Speed, and Acceleration
# ============================================================================
def data_particle_filter(df: pd.DataFrame, config: Dict[str, str]) -> pd.DataFrame:
    """
    Apply a particle filter using GPS data (latitude and longitude) along with speed and acceleration.

    The function uses the helper `select_gps_columns` to choose the GPS columns.
    It also gets the names of the speed and acceleration columns from the config.

    The state vector at each time step is:
          [latitude, longitude, speed, acceleration]

    The filtered state estimates are added as new columns:
          "pf_lat", "pf_lon", "pf_speed", "pf_acc".

    The config dictionary may include:
       - "speed_column": Name of the speed column.
       - "acc_col_for_particule_filter": Name of the acceleration column.
       - "N_for_particule_filter": Number of particles.
       - Process and measurement noise parameters.

    Returns:
        The modified DataFrame.
    """
    # Choose GPS columns.
    gps_lat_col, gps_lon_col = csv_select_gps_columns(df,
                                                  title="Select GPS Data for Particle Filter",
                                                  prompt="Select the GPS data to use as input for particule filter:")
    print(f"Using GPS columns: {gps_lat_col} and {gps_lon_col}")

    # Get speed and acceleration column names from config.
    speed_col = config.get("speed_column")
    acc_col = config.get("acc_col_for_particule_filter")
    for col in [speed_col, acc_col]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in the DataFrame.")

    # Number of particles (ensure key name matches your config).
    N = int(config.get("N_for_particule_filter"))

    # Process noise standard deviations (for each state dimension).
    process_std = np.array([
        float(config.get("process_std_lat", 0.0001)),
        float(config.get("process_std_lon", 0.0001)),
        float(config.get("process_std_speed", 0.1)),
        float(config.get("process_std_acc", 0.1))
    ])

    # Measurement noise standard deviations.
    measurement_std = np.array([
        float(config.get("measurement_std_lat", 0.0001)),
        float(config.get("measurement_std_lon", 0.0001)),
        float(config.get("measurement_std_speed", 0.1)),
        float(config.get("measurement_std_acc", 0.1))
    ])

    # Build the observation matrix.
    # Each observation: [lat, lon, speed, acceleration]
    observations = df[[gps_lat_col, gps_lon_col, speed_col, acc_col]].values
    T = observations.shape[0]
    d = 4  # state dimension

    # Initialize particles around the first observation.
    init_obs = observations[0]
    init_cov = np.diag((measurement_std ** 2) * 10)
    try:
        particles = np.random.multivariate_normal(mean=init_obs, cov=init_cov, size=N)
    except np.linalg.LinAlgError:
        particles = np.random.multivariate_normal(mean=init_obs, cov=np.eye(d) * 1e-6, size=N)
    weights = np.ones(N) / N  # uniform initial weights

    # Helper: Systematic resampling.
    def systematic_resample(weights):
        Np = len(weights)
        positions = (np.arange(Np) + np.random.uniform()) / Np
        indexes = np.zeros(Np, dtype=int)
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < Np:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes

    # Particle filter loop.
    filtered_states = []
    for z in observations:
        # Prediction: propagate particles with Gaussian process noise.
        noise = np.random.normal(0, process_std, size=(N, d))
        particles = particles + noise

        # Update: compute likelihood for each particle given measurement z.
        diff = particles - z
        squared_error = np.sum((diff / measurement_std) ** 2, axis=1)
        likelihood = np.exp(-0.5 * squared_error)
        weights *= likelihood
        weights += 1.e-300  # avoid numerical underflow
        weights /= np.sum(weights)

        # Resample particles.
        indexes = systematic_resample(weights)
        particles = particles[indexes]
        weights = np.ones(N) / N  # reset weights

        # Estimate: mean state from particles.
        mean_state = np.mean(particles, axis=0)
        filtered_states.append(mean_state)

    filtered_states = np.array(filtered_states)  # shape (T, 4)

    # Append the filtered state estimates as new columns.
    df["GPS_lat_smoothed_particule"] = filtered_states[:, 0]
    df["GPS_lon_smoothed_particule"] = filtered_states[:, 1]
    df["pf_speed"] = filtered_states[:, 2]
    df["pf_acc"] = filtered_states[:, 3]

    return df

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

def data_remove_gps_outliers(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Cleans GPS data by removing outliers based on unrealistic speed values
    and DBSCAN clustering.

    Parameters:
        df (pd.DataFrame): DataFrame with columns containing GPS coordinates and timestamps.
        config (dict): Configuration dictionary with parameters:
            - "speed_threshold_outliers": Maximum allowed speed in m/s before considering a point an outlier.
            - "dbscan_eps": Clustering radius in meters for DBSCAN.
            - "min_samples": Minimum number of samples in a cluster for DBSCAN.
            - "date_column": The column name of the timestamp.

    Returns:
        pd.DataFrame: Cleaned GPS data with outliers removed.
    """

    # Retrieve threshold values from config
    speed_threshold = config["speed_threshold_outliers"]
    dbscan_eps = config["dbscan_eps"]
    min_samples = config["min_samples"]

    # -- 1. Ask user to select GPS columns
    lat_col, lon_col = csv_select_gps_columns(
        df,
        title="Select GPS Data for outliers",
        prompt="Select the GPS data to use as input for outliers:"
    )

    # -- 2. Convert datetime column to UNIX timestamps (seconds)
    date_col = config["date_column"]
    df[date_col] = pd.to_datetime(df[date_col], format="%Y-%m-%d %H:%M:%S.%f")
    df["timestamp_unix"] = df[date_col].astype(np.int64) / 1e9  # Convert to seconds

    # Sort DataFrame by time if not already sorted (important for speed calculations)
    df.sort_values(by=date_col, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # -- 3. Calculate Haversine distance between consecutive points
    lat1 = np.radians(df[lat_col].shift(0))
    lon1 = np.radians(df[lon_col].shift(0))
    lat2 = np.radians(df[lat_col].shift(1))
    lon2 = np.radians(df[lon_col].shift(1))

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distances = 6371000 * c  # Earth radius in meters

    # -- 4. Calculate time differences
    time_diffs = df["timestamp_unix"].shift(1) - df["timestamp_unix"]

    # -- 5. Calculate speed (m/s), avoid division by zero
    # Shift forward so each row's "speed" is how fast we got FROM the previous point TO current row
    df["speed"] = distances / np.where(time_diffs > 0, time_diffs, np.inf)

    # -- 6. Filter rows exceeding the speed threshold
    df = df[df["speed"] < speed_threshold].copy()

    if df.empty:
        print("Warning: No data left after speed filtering. Returning empty DataFrame.")
        return df

    # -- 7. Apply DBSCAN clustering
    # Convert lat/lon to radians for DBSCAN or keep degrees and
    # adjust eps to match degrees. We'll keep using meters-based conversion:
    coords = df[[lat_col, lon_col]].values
    # Note: 1 degree ~ 111 km, so we convert dbscan_eps in meters to "degrees":
    eps_in_degrees = dbscan_eps / 111000.0

    clustering = DBSCAN(
        eps=eps_in_degrees,
        min_samples=min_samples,
        metric="euclidean"
    ).fit(coords)

    # DBSCAN assigns outliers the label -1
    df["cluster"] = clustering.labels_

    # -- 8. Keep only points with cluster != -1
    df = df[df["cluster"] != -1].copy()

    # -- 9. Clean up columns and return
    return df.drop(columns=["speed", "cluster", "timestamp_unix"])


def data_rolling_windows_gps_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Reduce the DataFrame row count by grouping:
      1) All consecutive 'stopped' points (speed < threshold) into ONE row
      2) Moving points in a speed-dependent time window
         (time_window = distance_window / speed, clamped between optional min/max)

    For each group/window, we compute the average lat/lon (and optionally speed),
    then store them in new columns:
       - "GPS_lat_smoothed_rolling_windows"
       - "GPS_lon_smoothed_rolling_windows"

    We keep the OTHER columns from the FIRST row in that group,
    so you still have all original columns but fewer total rows in the final output.

    Parameters
    ----------
    df : pd.DataFrame
        Must include:
            - A datetime column (config["date_column"], default "DatumZeit").
            - A speed column (named config["speed_column"] or default "speed").
        We also select lat/lon columns via csv_select_gps_columns (stubbed here).
    config : dict
        Expects keys:
          "speed_threshold_stopped_rolling_windows": float
          "distance_window_meters": float
          "time_window_min": float (optional)
          "time_window_max": float (optional)
          "speed_column": str
          "date_column": str (optional, default "DatumZeit")
          ... plus any other keys for controlling the behavior.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with:
          - FEWER rows: 1 row per group/window
          - All original columns from the FIRST row of each group
          - 2 extra columns:
              "GPS_lat_smoothed_rolling_windows",
              "GPS_lon_smoothed_rolling_windows"
            containing the group-average lat/lon.
        The 'time_numeric' in each row is set to the midpoint of that window.
    """

    lat_col_rol_win, lon_col_rol_win = csv_select_gps_columns(
        df,
        title="Select GPS Data for rolling windows",
        prompt="Select the GPS data to use as input for rolling windows:"
    )
    print(f"Using GPS columns: {lat_col_rol_win} and {lon_col_rol_win}")

    # 1. Parse datetime column -> numeric timestamps (in seconds)
    date_col = config.get("date_column", "DatumZeit")
    df[date_col] = pd.to_datetime(df[date_col], format="%Y-%m-%d %H:%M:%S.%f")
    # Convert to UNIX timestamp (float seconds)
    t_arr = df[date_col].astype(np.int64) / 1e9

    # -------------------------------------------------------------------------
    # 2. Extract relevant arrays (lat, lon, speed)
    # -------------------------------------------------------------------------
    lat_arr = df[lat_col_rol_win].to_numpy(dtype=float)
    lon_arr = df[lon_col_rol_win].to_numpy(dtype=float)

    speed_col = config["speed_column"]
    spd_arr = df[speed_col].to_numpy(dtype=float)

    n = len(df)

    # -------------------------------------------------------------------------
    # 3. Config parameters
    # -------------------------------------------------------------------------
    speed_threshold_stopped = config["speed_threshold_stopped_rolling_windows"]
    distance_window_meters = config["distance_window_meters"]
    time_window_min = config["time_window_min"] # optional
    time_window_max = config["time_window_max"]  # optional

    # Convert speed from e.g. km/h to m/s if needed:
    # Here we assume speed is ALREADY in m/s.
    # If it's in km/h, you'd do: spd_arr = spd_arr / 3.6

    def get_window_length(speed_value):
        """
        Return a time window (in seconds) based on a continuous function of speed:
          time_window = distance_window / speed_value (clamped).

        If speed < threshold_stopped, return None (we treat 'stopped' separately).
        """
        if speed_value < speed_threshold_stopped:
            return None  # We'll handle stopped logic in main loop

        speed_m_s = speed_value  # If needed: speed_value / 3.6 for km/h -> m/s

        raw_window = distance_window_meters / (speed_m_s + 1e-6)
        # Clamp extremes
        wlen = max(time_window_min, min(time_window_max, raw_window))
        return wlen

    # -------------------------------------------------------------------------
    # 4. Main loop: group rows => store only ONE row per group
    # -------------------------------------------------------------------------
    grouped_rows = []
    i = 0

    while i < n:
        current_speed = spd_arr[i]

        # CASE A: STOPPED => group all consecutive rows with speed < threshold
        if current_speed < speed_threshold_stopped:
            sum_lat = 0.0
            sum_lon = 0.0
            sum_spd = 0.0
            count = 0

            j = i
            while j < n and spd_arr[j] < speed_threshold_stopped:
                sum_lat += lat_arr[j]
                sum_lon += lon_arr[j]
                sum_spd += spd_arr[j]
                count += 1
                j += 1

            mean_lat = sum_lat / count
            mean_lon = sum_lon / count
            mean_spd = sum_spd / count

            # Create a NEW row from the FIRST row's data in this group
            row_dict = df.iloc[i].to_dict()  # copy original columns
            midpoint_time = 0.5 * (t_arr[i] + t_arr[j - 1])
            row_dict["time_numeric"] = midpoint_time
            row_dict[speed_col] = mean_spd

            # Add new columns for the smoothed lat/lon
            row_dict["GPS_lat_smoothed_rolling_windows"] = mean_lat
            row_dict["GPS_lon_smoothed_rolling_windows"] = mean_lon

            grouped_rows.append(row_dict)
            i = j

        # CASE B: MOVING => define a time window based on speed
        else:
            wlen = get_window_length(current_speed)
            # If speed < threshold_stopped, wlen would be None,
            # but we already handled that case above.

            window_end = t_arr[i] + wlen

            sum_lat = 0.0
            sum_lon = 0.0
            sum_spd = 0.0
            count = 0

            j = i
            while j < n and t_arr[j] <= window_end:
                sum_lat += lat_arr[j]
                sum_lon += lon_arr[j]
                sum_spd += spd_arr[j]
                count += 1
                j += 1

            mean_lat = sum_lat / count
            mean_lon = sum_lon / count
            mean_spd = sum_spd / count

            # Make a row from the FIRST row's data in [i..j-1]
            row_dict = df.iloc[i].to_dict()
            midpoint_time = 0.5 * (t_arr[i] + t_arr[j - 1])
            row_dict["time_numeric"] = midpoint_time
            row_dict[speed_col] = mean_spd

            row_dict["GPS_lat_smoothed_rolling_windows"] = mean_lat
            row_dict["GPS_lon_smoothed_rolling_windows"] = mean_lon

            grouped_rows.append(row_dict)
            i = j

    # -------------------------------------------------------------------------
    # 5. Build a NEW DataFrame with fewer rows
    # -------------------------------------------------------------------------
    df_grouped = pd.DataFrame(grouped_rows)

    return df_grouped

