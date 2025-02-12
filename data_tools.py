import os
import tkinter as tk
from tkinter import ttk
from typing import Dict, Any, Optional  # Add this line to fix the error

import requests
from scipy.interpolate import splprep, splev
from sklearn.neighbors import BallTree

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
      1) A speed-class break: if the speed class changes (e.g., from stopped to a different bin),
         we end the current group.
      2) For "stopped" speed class: group consecutive rows until speed moves above threshold
         or until a maximum stop time is reached.
      3) For "moving" speed classes: define a time window based on speed
         (time_window = distance_window / speed),
         but break early if speed crosses into a new class.

    The final row of each group contains averaged lat/lon/speed and a midpoint time.
    """

    # -------------------------------------------------------------------------
    # 1. Parse config and identify columns
    # -------------------------------------------------------------------------
    date_col = config.get("date_column", "DatumZeit")
    speed_col = config["speed_column"]
    speed_threshold_stopped = config["speed_threshold_stopped_rolling_windows"]
    distance_window_meters = config["distance_window_meters"]
    time_window_min = config["time_window_min"]
    time_window_max = config["time_window_max"]

    # New: maximum stop window in seconds (3 minutes = 180 seconds)
    max_stop_window = config.get("max_stop_window", 1800)

    # If config doesn't explicitly provide speed_bins,
    # we assume a simple 2-bin scenario: [0, threshold_stopped, ∞).
    speed_bins = config["speed_bins"]

    # 2. Convert date_col to numeric timestamps (float seconds)
    df[date_col] = pd.to_datetime(df[date_col], format="%Y-%m-%d %H:%M:%S.%f")
    t_arr = df[date_col].astype(np.int64) / 1e9  # convert ns -> s (float)

    # 3. Identify lat/lon columns (stubbed function you'd have in your code)
    lat_col_rol_win, lon_col_rol_win = csv_select_gps_columns(
        df,
        title="Select GPS Data for rolling windows",
        prompt="Select the GPS data to use as input for rolling windows:"
    )
    print(f"Using GPS columns: {lat_col_rol_win} and {lon_col_rol_win}")

    # 4. Extract arrays from DataFrame
    lat_arr = df[lat_col_rol_win].to_numpy(dtype=float)
    lon_arr = df[lon_col_rol_win].to_numpy(dtype=float)
    spd_arr = df[speed_col].to_numpy(dtype=float)

    n = len(df)

    # -------------------------------------------------------------------------
    # Helper: get_speed_class
    # -------------------------------------------------------------------------
    def get_speed_class(spd, bins):
        """
        Returns an integer index indicating which bin 'spd' falls into.
        E.g. if bins = [0.0, 0.3, 5.0, inf]:
           spd < 0.3 => class 0
           0.3 <= spd < 5.0 => class 1
           5.0 <= spd => class 2
        """
        for i in range(len(bins) - 1):
            if bins[i] <= spd < bins[i + 1]:
                return i
        # Fallback (should not happen if bins end at inf)
        return len(bins) - 1

    # -------------------------------------------------------------------------
    # Helper: get_window_length (time window for "moving" classes)
    # -------------------------------------------------------------------------
    def get_window_length(speed_value):
        """
        Return a time window (in seconds) based on a continuous function of speed:
          time_window = distance_window / speed_value (clamped between min/max).
        """
        # If speed is below threshold_stopped, we won't use a window;
        # that logic is handled separately for 'stopped'.
        if speed_value < speed_threshold_stopped:
            return None

        # if speed is in m/s already, just use it:
        raw_window = distance_window_meters / (speed_value + 1e-6)
        wlen = max(time_window_min, min(time_window_max, raw_window))
        return wlen

    # -------------------------------------------------------------------------
    # 5. Main grouping logic
    # -------------------------------------------------------------------------
    grouped_rows = []
    i = 0

    while i < n:
        # Determine the speed class of the current row
        current_class = get_speed_class(spd_arr[i], speed_bins)

        # CASE A: Stopped class (i.e., current_class == 0 if bins=[0, thr, inf])
        #         or more generally, you can treat class 0 as "stopped"
        if current_class == 0:
            sum_lat = 0.0
            sum_lon = 0.0
            sum_spd = 0.0
            count = 0
            j = i

            # Accumulate all consecutive rows that remain in class 0 (stopped)
            while j < n:
                # Break if the speed class has changed.
                if get_speed_class(spd_arr[j], speed_bins) != 0:
                    break
                # Break if the time elapsed exceeds the maximum stop window.
                if (t_arr[j] - t_arr[i]) > max_stop_window:
                    break

                sum_lat += lat_arr[j]
                sum_lon += lon_arr[j]
                sum_spd += spd_arr[j]
                count += 1
                j += 1

            mean_lat = sum_lat / count
            mean_lon = sum_lon / count
            mean_spd = sum_spd / count

            # Build the grouped row from the FIRST row's data
            row_dict = df.iloc[i].to_dict()
            midpoint_time = 0.5 * (t_arr[i] + t_arr[j - 1])
            row_dict["time_numeric"] = midpoint_time
            row_dict[speed_col] = mean_spd
            row_dict["GPS_lat_smoothed_rolling_windows"] = mean_lat
            row_dict["GPS_lon_smoothed_rolling_windows"] = mean_lon

            grouped_rows.append(row_dict)
            i = j

        # CASE B: "Moving" classes
        else:
            # Use the speed at the start of the group to define a window length
            initial_speed = spd_arr[i]
            wlen = get_window_length(initial_speed)
            if wlen is None:
                # If for some reason it's None, that means speed < threshold (contradiction).
                # We'll treat it as "stopped" or use a fallback.
                wlen = time_window_min

            window_end = t_arr[i] + wlen

            sum_lat = 0.0
            sum_lon = 0.0
            sum_spd = 0.0
            count = 0
            j = i

            # Accumulate points in the SAME speed class, as long as we're within window_end
            while j < n:
                # If speed class changed, break
                if get_speed_class(spd_arr[j], speed_bins) != current_class:
                    break
                # If time passes the window, break
                if t_arr[j] > window_end:
                    break

                sum_lat += lat_arr[j]
                sum_lon += lon_arr[j]
                sum_spd += spd_arr[j]
                count += 1
                j += 1

            mean_lat = sum_lat / count
            mean_lon = sum_lon / count
            mean_spd = sum_spd / count

            # Build the grouped row from the FIRST row's data
            row_dict = df.iloc[i].to_dict()
            midpoint_time = 0.5 * (t_arr[i] + t_arr[j - 1])
            row_dict["time_numeric"] = midpoint_time
            row_dict[speed_col] = mean_spd
            row_dict["GPS_lat_smoothed_rolling_windows"] = mean_lat
            row_dict["GPS_lon_smoothed_rolling_windows"] = mean_lon

            grouped_rows.append(row_dict)
            i = j

    # -------------------------------------------------------------------------
    # 6. Build a new DataFrame with fewer rows
    # -------------------------------------------------------------------------
    df_grouped = pd.DataFrame(grouped_rows)
    return df_grouped

def compute_spline_segment_distances(lat, lon, num_points=100):
    """
    Compute the traveled distance between consecutive GPS points
    using cubic splines.

    :param lat: Latitude array
    :param lon: Longitude array
    :param num_points: Number of points for spline evaluation (higher = more accurate)
    :return: Array of distances for each consecutive point
    """
    if len(lat) < 2:
        return np.array([])

    # Convert lat/lon to Cartesian coordinates (approximate Mercator projection)
    R = 6371000  # Earth radius in meters
    x = np.radians(lon) * R * np.cos(np.radians(lat))  # Approximate x (longitude)
    y = np.radians(lat) * R  # Approximate y (latitude)

    # Fit a cubic spline through the x, y coordinates
    tck, u = splprep([x, y], s=0)  # s=0 ensures strict interpolation

    # Compute segment-wise distances
    segment_distances = np.zeros(len(lat) - 1)

    for i in range(len(lat) - 1):
        # Generate fine-grained points **only for this segment**
        unew = np.linspace(u[i], u[i + 1], num_points)
        x_smooth, y_smooth = splev(unew, tck)

        # Compute the traveled distance along the spline for this segment
        dx = np.diff(x_smooth)
        dy = np.diff(y_smooth)
        segment_distances[i] = np.sum(np.sqrt(dx ** 2 + dy ** 2))

    return segment_distances


def data_compute_curvature(df, config):
    """
    Compute curvature from GPS data in two ways:
    1) Using consecutive lat/lon and yaw differences (distance-based).
    2) Using yaw_rate / speed directly.

    :param df: Pandas DataFrame containing GPS data
    :param config: Dict with column names for 'yaw', 'yaw_rate', 'speed'
    :return: Modified df with new columns:
             - 'distance_spline_segment'
             - 'curvature_yaw_distance'
             - 'curvature_yaw_rate'
    """

    # Select GPS columns
    lat_col, lon_col = csv_select_gps_columns(
        df,
        title="Select GPS Data",
        prompt="Select GPS columns for curvature calculation"
    )

    yaw_col = config["yaw"]  # e.g., 'yaw' in degrees
    yaw_rate_col = config["yaw_rate"]  # e.g., 'yaw_rate' in deg/s
    speed_col = config['speed']  # e.g., 'speed' in m/s

    # 1) Compute the traveled distance along the curve for each segment
    lat_vals = df[lat_col].values
    lon_vals = df[lon_col].values

    # Compute distances between consecutive points along the spline
    ds = compute_spline_segment_distances(lat_vals, lon_vals)  # shape: (n-1,)

    # Add the computed distances to the DataFrame
    df['distance_spline_segment'] = np.append(ds, np.nan)

    # Convert yaw from degrees -> radians and unwrap differences
    yaw_deg = df[yaw_col].values
    yaw_rad = np.radians(yaw_deg)
    dtheta = np.diff(yaw_rad)  # shape: (n-1,)
    dtheta = np.unwrap(dtheta)  # Avoid ±π jumps

    # Compute curvature (dtheta / ds)
    curvature_dist = np.zeros_like(ds)
    mask = (ds != 0)
    curvature_dist[mask] = dtheta[mask] / ds[mask]

    # Append NaN to align with DataFrame row count
    df['curvature_yaw_distance'] = np.append(curvature_dist, np.nan)

    # 2) Curvature from yaw_rate and speed
    yaw_rate_deg_s = df[yaw_rate_col].values  # deg/s
    speed_m_s = df[speed_col].values  # m/s

    # Convert deg/s -> rad/s
    yaw_rate_rad_s = yaw_rate_deg_s * (np.pi / 180.0)

    # Avoid div-by-zero and invalid values
    with np.errstate(divide='ignore', invalid='ignore'):
        curvature_rate = yaw_rate_rad_s / speed_m_s
        curvature_rate[~np.isfinite(curvature_rate)] = np.nan

    df['curvature_yaw_rate'] = curvature_rate

    return df


# ------------------------------------------------------------------------------
# data_fetch_tunnel_data: Fetches railway tunnel data from OSM using Overpass.
# ------------------------------------------------------------------------------
def data_fetch_tunnel_data(config: dict) -> pd.DataFrame:
    bbox = config["bbox"]
    overpass_url = config["overpass_url"]
    query = f"""
    [out:json];
    (
      way["railway"="rail"]["tunnel"="yes"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
    );
    out body;
    >;
    out skel qt;
    """
    print("Fetching railway tunnels from OSM...")
    response = requests.get(overpass_url, params={'data': query})
    data = response.json()
    print("Tunnel data received!")

    ways = {el["id"]: el["nodes"] for el in data["elements"] if el["type"] == "way"}
    nodes = {el["id"]: (el["lat"], el["lon"]) for el in data["elements"] if el["type"] == "node"}

    tunnel_list = []
    for way_id, node_ids in ways.items():
        coords = [nodes[nid] for nid in node_ids if nid in nodes]
        if len(coords) > 1:
            tunnel_list.append({"structure_id": way_id, "structure_type": "tunnel", "coordinates": coords})

    tunnel_df = pd.DataFrame(tunnel_list)
    return tunnel_df


# ------------------------------------------------------------------------------
# data_fetch_bridge_data: Fetches railway bridge data from OSM using Overpass.
# ------------------------------------------------------------------------------
def data_fetch_bridge_data(config: dict) -> pd.DataFrame:
    bbox = config["bbox"]
    overpass_url = config["overpass_url"]
    query = f"""
    [out:json];
    (
      way["railway"="rail"]["bridge"="yes"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
    );
    out body;
    >;
    out skel qt;
    """
    print("Fetching railway bridges from OSM...")
    response = requests.get(overpass_url, params={'data': query})
    data = response.json()
    print("Bridge data received!")

    ways = {el["id"]: el["nodes"] for el in data["elements"] if el["type"] == "way"}
    nodes = {el["id"]: (el["lat"], el["lon"]) for el in data["elements"] if el["type"] == "node"}

    bridge_list = []
    for way_id, node_ids in ways.items():
        coords = [nodes[nid] for nid in node_ids if nid in nodes]
        if len(coords) > 1:
            bridge_list.append({"structure_id": way_id, "structure_type": "bridge", "coordinates": coords})

    bridge_df = pd.DataFrame(bridge_list)
    return bridge_df


# ------------------------------------------------------------------------------
# data_add_structure_status: Annotates an input DataFrame with structure status,
# using the "GPS Qualität" column.
# ------------------------------------------------------------------------------
def data_add_infrastructure_status(input_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Annotates the input DataFrame with a new column "structure_status" based on the train's GPS data.

    The config dict should include:
      - "overpass_url": Overpass API URL.
      - "bbox": Bounding box [min_lat, min_lon, max_lat, max_lon].
      - "structure_threshold": Distance in km within which a train point is considered near a structure.
      - "tunnel_file": Local file name for tunnel data (e.g., "tunnels.csv").
      - "bridge_file": Local file name for bridge data (e.g., "bridges.csv").
      - "gps_lat_col": Name of the latitude column in the train DataFrame.
      - "gps_lon_col": Name of the longitude column in the train DataFrame.
      - "gps_quality_col": Name of the column with GPS quality (e.g., "GPS Qualität").

    For each train point:
      - If the value in "GPS Qualität" is not 4, the function checks for nearby structure nodes (tunnels or bridges).
          - If one is found, the status is set to "Encountered {structure_type} {structure_id}".
          - If none is found, it is set to "No structure encountered".
      - If "GPS Qualität" equals 4, the status is set to "GPS quality OK".

    Returns the updated DataFrame.
    """
    # Load or fetch tunnel data.
    tunnel_file = config.get("tunnel_file", "tunnels.csv")
    if os.path.exists(tunnel_file):
        print(f"Loading tunnel data from {tunnel_file} ...")
        tunnels_df = pd.read_csv(tunnel_file, converters={"coordinates": eval})
        print("Tunnel data loaded.")
    else:
        print("Tunnel file not found. Fetching tunnel data from OSM...")
        tunnels_df = data_fetch_tunnel_data(config)
        tunnels_df.to_csv(tunnel_file, index=False)
        print(f"Tunnel data saved to {tunnel_file}.")

    # Load or fetch bridge data.
    bridge_file = config.get("bridge_file", "bridges.csv")
    if os.path.exists(bridge_file):
        print(f"Loading bridge data from {bridge_file} ...")
        bridges_df = pd.read_csv(bridge_file, converters={"coordinates": eval})
        print("Bridge data loaded.")
    else:
        print("Bridge file not found. Fetching bridge data from OSM...")
        bridges_df = data_fetch_bridge_data(config)
        bridges_df.to_csv(bridge_file, index=False)
        print(f"Bridge data saved to {bridge_file}.")

    # Combine tunnel and bridge data.
    structures_df = pd.concat([tunnels_df, bridges_df], ignore_index=True)

    # Prepare structure data for spatial querying.
    all_structure_points = []
    all_structure_ids = []
    all_structure_types = []
    for _, row in structures_df.iterrows():
        for point in row["coordinates"]:
            all_structure_points.append(point)
            all_structure_ids.append(row["structure_id"])
            all_structure_types.append(row["structure_type"])
    all_structure_points = np.array(all_structure_points)
    structure_points_rad = np.radians(all_structure_points)
    tree = BallTree(structure_points_rad, metric='haversine')

    # Prepare train data.
    lat_col, lon_col = csv_select_gps_columns(
        input_df,
        title="Select GPS Data",
        prompt="Select GPS columns for tunnels"
    )
    train_points = input_df[[lat_col, lon_col]].to_numpy()
    train_points_rad = np.radians(train_points)

    structure_threshold = config.get("structure_threshold", 0.01)  # km
    earth_radius = 6371.0  # km
    threshold_rad = structure_threshold / earth_radius

    # Query the BallTree for nearby structure points.
    indices = tree.query_radius(train_points_rad, r=threshold_rad)

    quality_col = config.get("gps_quality_col", "GPS Qualität")
    statuses = []
    for i, idx_list in enumerate(indices):
        try:
            gps_quality = int(input_df.iloc[i][quality_col])
        except (ValueError, TypeError):
            gps_quality = None

        if gps_quality is not None and gps_quality != 4:
            if len(idx_list) > 0:
                structure_id = all_structure_ids[idx_list[0]]
                structure_type = all_structure_types[idx_list[0]]
                status = f"Encountered {structure_type} {structure_id}"
            else:
                status = "No structure encountered"
        else:
            status = "GPS quality OK"
        statuses.append(status)

    input_df["structure_status"] = statuses
    return input_df


