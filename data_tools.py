import tkinter as tk
from tkinter import ttk
from typing import Dict, Any  # Add this line to fix the error

import numpy as np
import pandas as pd
from pykalman import KalmanFilter
from pyproj import Transformer
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter



def data_convert_to_planar(df: pd.DataFrame, config: Dict[str, str]) -> pd.DataFrame:
    """
    Convert latitude and longitude coordinates to planar (UTM) coordinates using vectorized operations.
    Prioritize smoothed columns if available, otherwise use the original columns.

    Args:
        df: The input DataFrame with GPS data.
        config: Configuration dictionary with column names and transformation settings.

    Returns:
        DataFrame with added planar coordinates (x, y) and a column for the selected smoothing method.

    Raises:
        ValueError: If invalid columns or configurations are provided.
    """
    # Identify smoothed latitude and longitude columns
    smoothed_lat_columns = [col for col in df.columns if col.startswith("GPS_lat_smooth_")]
    smoothed_lon_columns = [col.replace("GPS_lat", "GPS_lon") for col in smoothed_lat_columns]

    # Initialize selected method
    selected_method = "none"  # Default to raw columns if no smoothing is applied

    # Determine which columns to use
    if len(smoothed_lat_columns) > 1:
        # Show a GUI to let the user choose
        root = tk.Tk()
        root.title("Select Smoothing Algorithm")

        selected_method_var = tk.StringVar(value=smoothed_lat_columns[0].split("_")[-1])  # Default to the first method

        def submit():
            root.destroy()

        # Add a label and dropdown menu
        tk.Label(root, text="Choose a smoothing method:").pack(pady=10)
        dropdown = ttk.Combobox(
            root,
            textvariable=selected_method_var,
            values=[col.split("_")[-1] for col in smoothed_lat_columns],
            state="readonly",
        )
        dropdown.pack(pady=10)

        # Add a submit button
        tk.Button(root, text="Submit", command=submit).pack(pady=10)

        # Run the GUI
        root.mainloop()

        selected_method = selected_method_var.get()

        # Validate the user input
        lat_col = f"GPS_lat_smooth_{selected_method}"
        lon_col = f"GPS_lon_smooth_{selected_method}"
        if lat_col not in df.columns or lon_col not in df.columns:
            raise ValueError(
                f"Invalid selection: {selected_method}. "
                f"Columns {lat_col} and {lon_col} not found."
            )
        print(f"Using smoothed columns: {lat_col}, {lon_col}")
    elif len(smoothed_lat_columns) == 1:
        # Use the single available smoothed column
        lat_col = smoothed_lat_columns[0]
        lon_col = smoothed_lon_columns[0]
        selected_method = lat_col.split("_")[-1]
        print(f"Automatically using smoothed columns: {lat_col}, {lon_col}")
    else:
        # Fall back to raw data columns specified in config
        lat_col = config["lat_col"]
        lon_col = config["lon_col"]
        print(f"No smoothed GPS columns found. Using raw data columns: {lat_col}, {lon_col}")

    # Transformer: WGS84 (EPSG:4326) to UTM zone 33N (EPSG:32633)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
    x, y = transformer.transform(df[lon_col].values, df[lat_col].values)

    # Add planar coordinates to the DataFrame
    df["x"] = x
    df["y"] = y

    # Add the selected smoothing method to the DataFrame
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


def data_smooth_gps_savitzky(df: pd.DataFrame, config: Dict[str, str]) -> pd.DataFrame:
    """
    Smooth the GPS latitude and longitude data using a Savitzky-Golay filter.

    Args:
        df: The input DataFrame with GPS data.
        config: Configuration dictionary containing column names and settings.

    Returns:
        Updated DataFrame with smoothed GPS latitude and longitude columns.

    Raises:
        KeyError: If required columns are missing in the config.
    """
    # Ensure the required keys are in the config
    if "lat_col" not in config or "lon_col" not in config:
        raise KeyError("Configuration must include 'lat_col' and 'lon_col'.")

    lat_col = config["lat_col"]
    lon_col = config["lon_col"]

    # S-G filter parameters (adjust as needed)
    window_length = 51  # must be odd
    polyorder = 2

    df[f"{lat_col}_smooth_savitzky"] = savgol_filter(df[lat_col], window_length, polyorder)
    df[f"{lon_col}_smooth_savitzky"] = savgol_filter(df[lon_col], window_length, polyorder)

    return df


def data_smooth_gps_gaussian(df: pd.DataFrame, config: Dict[str, str]) -> pd.DataFrame:
    """
    Smooth the GPS latitude and longitude data using a Gaussian filter.

    Args:
        df: The input DataFrame with GPS data.
        config: Configuration dictionary containing column names and settings.

    Returns:
        Updated DataFrame with smoothed GPS latitude and longitude columns.

    Raises:
        KeyError: If required columns are missing in the config.
    """
    # Ensure the required keys are in the config
    if "lat_col" not in config or "lon_col" not in config:
        raise KeyError("Configuration must include 'lat_col' and 'lon_col'.")

    lat_col = config["lat_col"]
    lon_col = config["lon_col"]

    # Gaussian filter parameter (standard deviation)
    sigma = 2

    df[f"{lat_col}_smooth_gaussian"] = gaussian_filter1d(df[lat_col], sigma)
    df[f"{lon_col}_smooth_gaussian"] = gaussian_filter1d(df[lon_col], sigma)

    return df

def data_delete_the_one_percent(df: pd.DataFrame, config: Dict[str, str]):
    # Get required config values (with defaults, if needed).
    date_col = config.get("date_column", "DatumZeit")
    yaw_rate_col = config.get("yaw_rate_column", "yaw_rate_deg_s")

    # Check if required columns exist in df
    if yaw_rate_col not in df.columns or date_col not in df.columns:
        raise ValueError(f"The required columns '{yaw_rate_col}' or '{date_col}' "
                         f"are missing from the CSV file.")

    # Calculate 1% and 99% quantiles for yaw_rate
    lower_bound = df[yaw_rate_col].quantile(0.01)
    upper_bound = df[yaw_rate_col].quantile(0.99)

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
