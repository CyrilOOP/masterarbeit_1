import os
from typing import Tuple, List, Dict, Any, Optional  # Add this line to fix the error

import requests
from geopy.distance import geodesic
from scipy.interpolate import splprep, splev, CubicSpline
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree

from csv_tools import  csv_select_gps_columns
import numpy as np

from pykalman import KalmanFilter
from pyproj import Transformer
from filterpy.monte_carlo import systematic_resample
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from filterpy.kalman import KalmanFilter


import logging

import concurrent.futures
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import utm
import pandas as pd
import pyproj
from typing import Dict


def data_convert_to_planar(df: pd.DataFrame, config: Dict[str, str]) -> pd.DataFrame:
    """
    Convert latitude and longitude to planar coordinates (UTM).
    Uses the helper to select which GPS columns to use.

    Args:
        df: The input DataFrame containing GPS data.
        config: A configuration dictionary.

    Returns:
        The DataFrame with added planar coordinates (x, y) and a column 'selected_smoothing_method'.
    """
    # Use helper to select GPS columns.
    lat_input, lon_input = csv_select_gps_columns(
        df,
        title="Select GPS Data for Planar Conversion",
        prompt="Select the GPS data to use for planar conversion:"
    )
    print(f"Using input columns: {lat_input} and {lon_input}")

    # Determine the selected smoothing method
    selected_method = lat_input.split("smoothed_")[-1] if "smoothed" in lat_input else "raw"

    # **Dynamically determine UTM zone based on longitude**
    utm_zone = int((df[lon_input].mean() + 180) / 6) + 1
    is_northern = df[lat_input].mean() >= 0  # True if in northern hemisphere
    epsg_code = f"EPSG:{32600 + utm_zone if is_northern else 32700 + utm_zone}"

    # **Use pyproj Transformer**
    transformer = pyproj.Transformer.from_crs("EPSG:4326", epsg_code, always_xy=True)

    # Convert coordinates
    df["x"], df["y"] = transformer.transform(df[lon_input].values, df[lat_input].values)

    # Add selected method
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


def data_parse_time_and_compute_dt(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Parse the given datetime column as pandas datetime and compute two time metrics:
      - 'time_seconds': Time in seconds relative to the first timestamp.
      - 'dt': Time difference in seconds between consecutive rows.

    Args:
        df: Input DataFrame.
        config: Configuration dictionary containing:
            - "datetime_col": Name of the column containing datetime information.
              Defaults to "DatumZeit" if not provided.

    Returns:
        A copy of the DataFrame with new columns 'time_seconds' and 'dt'.

    Raises:
        ValueError: If the datetime column cannot be parsed.
    """
    datetime_col = config.get("datetime_col", "DatumZeit")
    df = df.copy()

    # Convert the column to datetime using explicit format (including milliseconds)
    try:
        df[datetime_col] = pd.to_datetime(df[datetime_col], format="%Y-%m-%d %H:%M:%S.%f")
    except Exception as e:
        raise ValueError(f"Error parsing datetime column '{datetime_col}': {e}")

    # Create a column with time in seconds relative to the first timestamp
    df["elapsed_time_s"] = (df[datetime_col] - df[datetime_col].iloc[0]).dt.total_seconds()

    # Compute the difference in timestamps between consecutive rows
    df["delta_time_s"] = df[datetime_col].diff().dt.total_seconds()

    return df



def data_compute_heading_dx_dy(df: pd.DataFrame, config: Dict[str, str]) -> pd.DataFrame:
    """
    Compute the heading based on the train's path using the differences in coordinates.
    This heading represents the direction the train is pointing based on its (x, y) path.

    Args:
        df: DataFrame containing the coordinate data.
        config: Configuration dictionary with keys:
            - "x_col": Name of the x-coordinate column (default: "x").
            - "y_col": Name of the y-coordinate column (default: "y").
            - "heading_col": Name of the output heading column (default: "heading_deg").

    Returns:
        The DataFrame with a new column for the path-based heading.
    """
    # Retrieve column names from config with defaults
    x_col = "x"
    y_col = "y"
    heading_col = "heading_dx_dy"

    # Validate that required columns exist
    for col in [x_col, y_col]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # Prepare an array for the computed heading
    n = len(df)
    heading = np.full(n, np.nan)  # Fill with NaN by default

    # Extract coordinate arrays
    x = df[x_col].values
    y = df[y_col].values

    # Loop over the data where a full window exists
    for i in range(3, n - 3):
        # Calculate mean of 3 points before
        x_before = np.mean(x[i - 3:i])
        y_before = np.mean(y[i - 3:i])

        # Calculate mean of 3 points after
        x_after = np.mean(x[i + 1:i + 4])
        y_after = np.mean(y[i + 1:i + 4])

        # Compute the difference between the means
        dx = x_after - x_before
        dy = y_after - y_before

        # Calculate heading in degrees relative to north (0° = north)
        heading[i] = np.degrees(np.arctan2(dx, dy)) % 360

    # Assign the computed heading to the DataFrame
    df[heading_col] = heading

    return df



def data_compute_heading_ds(df: pd.DataFrame, config: Dict[str, str]) -> pd.DataFrame:
    """
    Compute the heading (yaw angle in degrees) using vectorized operations,
    based on x and y coordinates and a precomputed cumulative distance.

    The heading is computed as:
        heading = arctan2(dx/ds, dy/ds)
    and adjusted to the [0, 360) degree range, so that 0° corresponds to north.

    Args:
        df (pd.DataFrame): DataFrame containing coordinate columns and a cumulative distance column.
        config (Dict[str, str]): Configuration dictionary with:
            - "x_col": Name of the x-coordinate column (default "x").
            - "y_col": Name of the y-coordinate column (default "y").
            - "cum_dist_col": Name of the cumulative distance column (default "cumulative_distance").
            - "heading_col_ds": Name for the output heading column (default "heading_deg_ds").

    Returns:
        pd.DataFrame: The input DataFrame with a new vectorized column containing the heading in degrees.
    """
    # Retrieve column names from config with defaults
    x_col = "x"
    y_col = "y"
    cum_dist_col = config.get("cum_dist_col", "cumulative_distance")
    heading_col = "heading_deg_ds"

    # Validate required columns exist
    for col in [x_col, y_col, cum_dist_col]:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain column '{col}'.")

    # Convert required columns to numpy arrays
    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()
    s = df[cum_dist_col].to_numpy()

    # Ensure s is strictly increasing by adding a small epsilon where needed
    eps = 1e-6
    s_fixed = s.copy()
    for i in range(1, len(s_fixed)):
        if s_fixed[i] <= s_fixed[i - 1]:
            s_fixed[i] = s_fixed[i - 1] + eps

    # Compute derivatives with respect to the fixed cumulative distance
    dx_ds = np.gradient(x, s_fixed)
    dy_ds = np.gradient(y, s_fixed)

    # Compute heading in radians then convert to degrees.
    # Swapping the arguments in arctan2 gives 0° = north.
    heading = np.degrees(np.arctan2(dx_ds, dy_ds))

    # Normalize to [0, 360)
    heading = (heading + 360) % 360

    # Store the computed heading in the DataFrame
    df[heading_col] = heading

    return df

def select_heading_columns_gui(heading_candidates):
    """
    Opens a dynamic Tkinter GUI dialog with a multi-select Listbox,
    allowing the user to select one or more heading columns.

    Returns a list of selected column names.
    """
    selected = []

    def on_ok():
        nonlocal selected
        indices = listbox.curselection()
        selected = [heading_candidates[i] for i in indices]
        root.destroy()

    root = tk.Tk()
    root.title("Select Heading Columns")
    root.geometry("400x300")
    root.minsize(300, 200)

    # Configure grid weights to allow dynamic resizing
    root.columnconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)

    # Label at the top
    label = ttk.Label(root, text="Select heading column(s):")
    label.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")

    # Frame for the listbox and scrollbar
    list_frame = ttk.Frame(root)
    list_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
    list_frame.columnconfigure(0, weight=1)
    list_frame.rowconfigure(0, weight=1)

    # Create the listbox and scrollbar
    scrollbar = ttk.Scrollbar(list_frame, orient="vertical")
    scrollbar.grid(row=0, column=1, sticky="ns")

    listbox = tk.Listbox(list_frame, selectmode="multiple", yscrollcommand=scrollbar.set)
    listbox.grid(row=0, column=0, sticky="nsew")
    scrollbar.config(command=listbox.yview)

    # Insert heading candidates into the listbox
    for col in heading_candidates:
        listbox.insert(tk.END, col)

    # Frame for the OK button at the bottom
    button_frame = ttk.Frame(root)
    button_frame.grid(row=2, column=0, padx=10, pady=(5, 10), sticky="ew")
    button_frame.columnconfigure(0, weight=1)

    ok_button = ttk.Button(button_frame, text="OK", command=on_ok)
    ok_button.grid(row=0, column=0, sticky="ew")

    root.mainloop()
    return selected


import pandas as pd
from tkinter import messagebox


def data_compute_yaw_rate_from_heading(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Compute yaw rate (in degrees/second) from heading column(s) in the DataFrame.
    Instead of using a pre-computed delta_time_s, this function derives the time delta
    by diffing the 'elapsed_time_s' column.
    """
    yaw_rate_sign = config.get("yaw_rate_sign", 1)

    # --- 1) Check if we have elapsed_time_s in df ---
    if "elapsed_time_s" not in df.columns:
        raise ValueError(
            "Column 'elapsed_time_s' is missing from DataFrame; "
            "cannot compute time deltas."
        )

    # Create (or overwrite) a column for time delta by diffing elapsed_time_s
    df["time_delta_s"] = df["elapsed_time_s"].diff()
    # Replace NaN or 0 with a tiny value to avoid division-by-zero
    df["time_delta_s"] = df["time_delta_s"].fillna(0).replace(0, 1e-6)

    # --- 2) Find candidate heading columns (case-insensitive) ---
    heading_candidates = [col for col in df.columns if "heading" in col.lower()]
    if not heading_candidates:
        messagebox.showinfo("Yaw Rate Computation", "No heading columns found in DataFrame.")
        return df

    # If there's only one heading column, use it; otherwise, let user pick via GUI
    if len(heading_candidates) == 1:
        selected_heading = heading_candidates
        print(f"Only one heading column found: '{heading_candidates[0]}'. Using it for yaw rate computation.")
    else:
        selected_heading = select_heading_columns_gui(heading_candidates)
        if not selected_heading:
            messagebox.showwarning("Yaw Rate Computation", "No heading columns were selected. No yaw rate computed.")
            return df

    # --- 3) Compute yaw rate for each selected heading column ---
    for heading_col in selected_heading:
        # Compute the difference between consecutive heading values
        heading_diff = df[heading_col].diff()

        # Wrap heading differences to the range [-180, 180]
        heading_diff_wrapped = (heading_diff + 180) % 360 - 180

        # Compute yaw rate = (wrapped difference) / (time delta) * yaw_rate_sign
        yaw_rate = yaw_rate_sign * (heading_diff_wrapped / df["time_delta_s"])

        # Create a new column for the computed yaw rate
        new_col_name = f"yaw_rate_from_{heading_col}"
        df[new_col_name] = yaw_rate

        # Optionally set yaw rate to 0 for rows marked as STOP
        if "segment_marker" in df.columns:
            stop_indices = df["segment_marker"].isin(["STOP_START", "STOP_END"])
            df.loc[stop_indices, new_col_name] = 0

    # --- 4) After computing yaw rates, optionally remove the first row after each STOP_END ---
    if "segment_marker" in df.columns:
        indices_to_drop = []
        for idx in df[df["segment_marker"] == "STOP_END"].index:
            next_idx = idx + 1
            # Drop next row unless it's also STOP_END
            if next_idx in df.index and df.loc[next_idx, "segment_marker"] != "STOP_END":
                indices_to_drop.append(next_idx)
        if indices_to_drop:
            df = df.drop(indices_to_drop).reset_index(drop=True)

    return df


def data_delete_the_one_percent(df: pd.DataFrame, config: Dict[str, str]) -> pd.DataFrame:
    """
    Delete rows with extreme yaw rate values, defined by lower and upper quantile bounds.

    If more than one yaw rate column is present and no specific column is provided via config,
    a nested GUI function will prompt you to select which yaw rate column to use.

    Config keys:
      - "date_column": Name of the date/time column (default "DatumZeit").
      - "yaw_rate_column": (Optional) Name of the yaw rate column to filter.
      - "delete_lower_bound_percentage": Lower bound percentage (e.g., 1 for 1%).
      - "delete_upper_bound_percentage": Upper bound percentage (e.g., 99 for 99%).

    Returns:
      The filtered DataFrame.
    """

    # Nested GUI function to select a yaw rate column
    def select_yaw_rate_column_gui(candidates):
        # Create the Tk root window first
        root = tk.Tk()
        root.title("Select Yaw Rate Column")
        root.geometry("300x200")

        # Now create a StringVar associated with this root
        selected = tk.StringVar(root, value=candidates[0])  # default to first candidate

        label = ttk.Label(root, text="Select a yaw rate column to use for filtering:")
        label.pack(padx=10, pady=10)

        # Create radio buttons for each candidate column
        for candidate in candidates:
            rb = ttk.Radiobutton(root, text=candidate, variable=selected, value=candidate)
            rb.pack(anchor="w", padx=10)

        def on_submit():
            root.destroy()

        submit_button = ttk.Button(root, text="OK", command=on_submit)
        submit_button.pack(pady=10)

        root.mainloop()
        return selected.get()

    # Get required config values (with defaults, if needed)
    date_col = config.get("date_column", "DatumZeit")
    yaw_rate_col = config.get("yaw_rate_column", None)
    input_lower_bound = config["delete_lower_bound_percentage"] / 100.0
    print(f'input_lower_bound: {input_lower_bound}')
    input_upper_bound = config["delete_upper_bound_percentage"] / 100.0
    print(f'input_upper_bound: {input_upper_bound}')

    # Check if required date column exists
    if date_col not in df.columns:
        raise ValueError(f"The required column '{date_col}' is missing from the CSV file.")

    # Determine which yaw rate column to use.
    if not yaw_rate_col or yaw_rate_col not in df.columns:
        # Look for candidate columns (case-insensitive search)
        candidates = [col for col in df.columns if "yaw_rate" in col.lower()]
        if len(candidates) == 0:
            raise ValueError("No yaw rate columns found in the CSV file.")
        elif len(candidates) == 1:
            yaw_rate_col = candidates[0]
            print(f"Only one yaw rate column found: '{yaw_rate_col}'. Using it for filtering.")
        else:
            # Multiple candidates found: use the nested GUI to let the user choose.
            yaw_rate_col = select_yaw_rate_column_gui(candidates)
            print(f"You selected '{yaw_rate_col}' for filtering.")

    # Calculate lower and upper quantiles for the selected yaw rate column.
    lower_bound = df[yaw_rate_col].quantile(input_lower_bound)
    upper_bound = df[yaw_rate_col].quantile(input_upper_bound)
    print(f"Filtering '{yaw_rate_col}' between {lower_bound} and {upper_bound}.")

    # Filter rows within the quantile range.
    df = df[(df[yaw_rate_col] >= lower_bound) & (df[yaw_rate_col] <= upper_bound)]
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
    Process GPS data into rolling windows based on speed classes, with additional
    filtering and hysteresis to reduce noise. Two types of segments are handled:

      1) Stopped segments (speed in class 0): Group consecutive rows that remain stopped,
         but break if the speed rises for several consecutive points (hysteresis) or if
         the maximum stop window is exceeded.
      2) Moving segments (non-zero speed class): Define a time window based on speed
         (time_window = distance_window / speed) and break early if the speed class changes,
         using a hysteresis check.

    Each segment is represented by a final (or pair of) row(s) with averaged lat, lon, speed,
    and a representative timestamp.
    """

    # 1. Parse config and identify columns
    date_col = config.get("date_column", "DatumZeit")
    speed_col = config["speed_column"]
    speed_threshold_stopped = config["speed_threshold_stopped_rolling_windows"]
    distance_window_meters = config["distance_window_meters"]
    time_window_min = config["time_window_min"]
    time_window_max = config["time_window_max"]
    max_stop_window = config["max_stop_window"]
    speed_bins = config["speed_bins"]

    # 2. Convert date_col to datetime and then to numeric timestamps (in seconds)
    df[date_col] = pd.to_datetime(df[date_col], format="%Y-%m-%d %H:%M:%S.%f")
    t_arr = df[date_col].astype(np.int64) / 1e9  # nanoseconds -> seconds

    # 3. Identify lat/lon columns (using your helper; assume it returns valid column names)
    lat_col_rol_win, lon_col_rol_win = csv_select_gps_columns(
        df,
        title="Select GPS Data for rolling windows",
        prompt="Select the GPS data to use as input for rolling windows:"
    )
    print(f"Using GPS columns: {lat_col_rol_win} and {lon_col_rol_win}")

    # 4. Extract arrays from DataFrame
    lat_arr = df[lat_col_rol_win].to_numpy(dtype=float)
    lon_arr = df[lon_col_rol_win].to_numpy(dtype=float)

    # Apply a rolling (smoothed) speed filter to reduce noise.
    df['speed_filtered'] = df[speed_col].rolling(window=5, center=True, min_periods=1).mean()
    spd_arr = df['speed_filtered'].to_numpy(dtype=float)

    n = len(df)

    # -------------------------------------------------------------------------
    # Helper: get_speed_class
    # -------------------------------------------------------------------------
    def get_speed_class(spd, bins):
        """
        Returns an integer index indicating which bin 'spd' falls into.
        For example, if bins = [0.0, 0.3, 5.0, inf]:
          spd < 0.3      => class 0 (stopped)
          0.3 <= spd < 5.0 => class 1 (moving slow)
          5.0 <= spd      => class 2 (moving fast)
        """
        for i in range(len(bins) - 1):
            if bins[i] <= spd < bins[i + 1]:
                return i
        return len(bins) - 1  # fallback

    # -------------------------------------------------------------------------
    # Helper: get_window_length (for "moving" classes)
    # -------------------------------------------------------------------------
    def get_window_length(speed_value):
        """
        Return a time window (in seconds) based on a continuous function of speed:
          time_window = distance_window_meters / speed_value (clamped between time_window_min and time_window_max)
        """
        if speed_value < speed_threshold_stopped:
            return None
        raw_window = distance_window_meters / (speed_value + 1e-6)
        return max(time_window_min, min(time_window_max, raw_window))

    # Set a hysteresis window (number of samples) to ensure the state change is stable.
    hysteresis_window = 10

    # -------------------------------------------------------------------------
    # 5. Main grouping logic with filtering and hysteresis
    # -------------------------------------------------------------------------
    grouped_rows = []
    i = 0

    while i < n:
        # Use the filtered speed for state decisions.
        current_speed = spd_arr[i]
        current_class = get_speed_class(current_speed, speed_bins)

        # CASE A: Stopped class (class 0)
        if current_class == 0:
            sum_lat = 0.0
            sum_lon = 0.0
            sum_spd = 0.0
            count = 0
            j = i

            # Accumulate rows while the segment remains stopped.
            while j < n:
                # If current row's speed appears non-stopped, check the next few samples
                if get_speed_class(spd_arr[j], speed_bins) != 0:
                    # Look ahead over a small window to confirm the change.
                    end_idx = min(n, j + hysteresis_window)
                    if np.mean(spd_arr[j:end_idx]) >= speed_threshold_stopped:
                        break
                # Break if the elapsed time exceeds the maximum stop window.
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

            # Instead of one row, we create two: one for the start and one for the end of the stop segment.
            start_row_dict = df.iloc[i].copy().to_dict()
            start_row_dict["time_numeric"] = t_arr[i]
            start_row_dict[speed_col] = mean_spd
            start_row_dict["GPS_lat_smoothed_rolling_windows"] = mean_lat
            start_row_dict["GPS_lon_smoothed_rolling_windows"] = mean_lon
            start_row_dict["segment_marker"] = "STOP_START"
            grouped_rows.append(start_row_dict)

            end_row_dict = df.iloc[j - 1].copy().to_dict()
            end_row_dict["time_numeric"] = t_arr[j - 1]
            end_row_dict[speed_col] = mean_spd
            end_row_dict["GPS_lat_smoothed_rolling_windows"] = mean_lat
            end_row_dict["GPS_lon_smoothed_rolling_windows"] = mean_lon
            end_row_dict["segment_marker"] = "STOP_END"
            grouped_rows.append(end_row_dict)

            i = j

        # CASE B: Moving classes (non-zero speed class)
        else:
            initial_speed = spd_arr[i]
            wlen = get_window_length(initial_speed)
            if wlen is None:
                wlen = time_window_min
            window_end = t_arr[i] + wlen

            sum_lat = 0.0
            sum_lon = 0.0
            sum_spd = 0.0
            count = 0
            j = i

            # Accumulate rows while within the same speed class and within the time window.
            while j < n:
                if get_speed_class(spd_arr[j], speed_bins) != current_class:
                    # Hysteresis: check a few points ahead to see if the change is sustained.
                    end_idx = min(n, j + hysteresis_window)
                    if get_speed_class(np.mean(spd_arr[j:end_idx]), speed_bins) != current_class:
                        break
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

            # For moving segments, create a single row at the midpoint.
            row_dict = df.iloc[i].copy().to_dict()
            midpoint_time = 0.5 * (t_arr[i] + t_arr[j - 1])
            row_dict["time_numeric"] = midpoint_time
            row_dict[speed_col] = mean_spd
            row_dict["GPS_lat_smoothed_rolling_windows"] = mean_lat
            row_dict["GPS_lon_smoothed_rolling_windows"] = mean_lon
            row_dict["segment_marker"] = "MOVING"
            grouped_rows.append(row_dict)
            i = j

    # 6. Build a new DataFrame with the grouped (filtered & smoothed) rows
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



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def data_get_elevation(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Fetch elevation data for GPS coordinates in a DataFrame using an external API
    and append the elevation data as a new column.

    Parameters:
        df (pd.DataFrame): DataFrame containing GPS coordinate data.
        config (Dict[str, Any]): Configuration dictionary with the following keys:
            - 'api_key': API key for the elevation service.
            - 'elevation_column': Name of the column to store elevation data.
            - 'api_url': Base URL of the elevation API.
            - 'batch_size': Number of coordinates to process in each batch.
            - 'threads': Number of parallel threads to use.

    Returns:
        pd.DataFrame: The DataFrame with an added column for elevation data.

    Raises:
        KeyError: If required keys are missing from the configuration.
        ValueError: If the DataFrame is empty or the selected GPS columns are missing.
    """
    # Show the API Key warning pop-up
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    response = messagebox.askyesno(
        "Warning: API Key Usage",
        "⚠️ WARNING: This script uses **my** private API key.\n\n"
        "If you run this code, be aware that:\n"
        "- API calls may cost money depending on my quota.\n"
        "- Misuse of my key can lead to restrictions or bans.\n"
        "- Make sure you understand what you're doing before proceeding.\n\n"
        "Do you want to proceed?"
    )

    root.destroy()

    if not response:
        exit("Execution aborted by user.")  # Stop execution if user cancels

    print("Continuing with the script execution...")

    # Validate configuration keys (lowercase expected)
    required_keys = ["api_key", "elevation_column", "api_url", "batch_size", "threads"]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(f"Missing configuration keys: {missing_keys}")

    # Load configuration values
    api_key: str = config["api_key"]
    elevation_column: str = config["elevation_column"]
    api_url: str = config["api_url"]
    batch_size: int = config["batch_size"]
    threads: int = config["threads"]

    # Validate DataFrame
    if df.empty:
        raise ValueError("DataFrame is empty.")

    # Select GPS columns (Assuming csv_select_gps_columns is defined elsewhere)
    lat_col, lon_col = csv_select_gps_columns(
        df,
        title="Select GPS Data for Elevation",
        prompt="Select the GPS data to use as input for elevation:"
    )
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError("Selected GPS columns not found in DataFrame.")

    logger.info(f"Using GPS columns: {lat_col} and {lon_col}")

    # Prepare list of coordinates
    coords: List[Tuple[Any, Any]] = list(zip(df[lat_col], df[lon_col]))
    total_rows: int = len(coords)

    # Create batches for multithreading
    elevations: List[Any] = [None] * total_rows  # Placeholder for elevation data
    batch_indices: List[List[int]] = [
        list(range(i, min(i + batch_size, total_rows)))
        for i in range(0, total_rows, batch_size)
    ]
    batches: List[List[Tuple[Any, Any]]] = [
        coords[indices[0]: indices[-1] + 1] for indices in batch_indices
    ]

    logger.info(f"🚀 Processing {len(batches)} batches with {threads} parallel threads...")

    def get_elevation_batch(coords_batch: List[Tuple[Any, Any]]) -> List[Any]:
        """
        Retrieve elevation data for a batch of coordinates from the API.

        Parameters:
            coords_batch (List[Tuple[Any, Any]]): A list of (latitude, longitude) tuples.

        Returns:
            List[Any]: A list of elevation values or None for each coordinate.
        """
        locations_str = "|".join(f"{lat},{lon}" for lat, lon in coords_batch)
        params = {
            "locations": locations_str,
            "key": api_key,
        }

        try:
            response = requests.get(api_url, params=params)
            response.raise_for_status()  # Raises HTTPError for bad status codes
            data = response.json()

            if data.get("status") == "OK":
                results = data.get("results", [])
                return [result.get("elevation") for result in results]
            else:
                logger.error(f"API Error: {data.get('status')}")
                return [None] * len(coords_batch)
        except Exception as e:
            logger.error(f"Request failed for batch starting with {coords_batch[0]}: {e}")
            return [None] * len(coords_batch)

    # Execute API calls in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        results: List[List[Any]] = list(executor.map(get_elevation_batch, batches))

    # Merge the results back into the elevations list
    for batch_idx, batch_elevations in enumerate(results):
        for idx, elevation in zip(batch_indices[batch_idx], batch_elevations):
            elevations[idx] = elevation

    # Add the elevation data as a new column in the DataFrame
    df[elevation_column] = elevations

    return df

def data_compute_traveled_distance(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute the traveled distance between consecutive GPS points using a spline-based approach
    (only if enough unique points), plus geodesic approximation for all rows.
    """
    print("=== data_compute_traveled_distance: START ===")

    # Select GPS columns
    lat_col, lon_col = csv_select_gps_columns(
        df,
        title="Select GPS Data for Distances",
        prompt="Select the GPS data to use as input to calculate the distance traveled:"
    )
    print(f"[INFO] Selected lat_col='{lat_col}', lon_col='{lon_col}'")

    # Verify columns exist
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"Selected GPS columns '{lat_col}' or '{lon_col}' not found in DataFrame.")

    logger.info(f"Using GPS columns: {lat_col} and {lon_col}")
    print(f"[INFO] Number of rows in df: {len(df)}")

    # Extract numpy arrays from the DataFrame
    latitudes = df[lat_col].values
    longitudes = df[lon_col].values

    print("[DEBUG] latitudes[:5]:", latitudes[:5])
    print("[DEBUG] longitudes[:5]:", longitudes[:5])

    # 1) Gather lat/lon coordinates and filter out consecutive duplicates
    coords = np.column_stack((latitudes, longitudes))
    mask = np.ones(len(coords), dtype=bool)
    # Mark rows that are identical to the previous row as False
    mask[1:] = (coords[1:] != coords[:-1]).any(axis=1)
    filtered_coords = coords[mask]
    print(f"[INFO] After filtering, data has {len(filtered_coords)} points.")

    # Compute unique coordinates from the filtered data
    unique_coords = np.unique(filtered_coords, axis=0)
    print(f"[INFO] Found {len(unique_coords)} unique lat/lon coordinate pairs in filtered data.")

    # 2) If fewer than 3 unique points, skip the spline
    if len(unique_coords) < 3:
        warning_msg = (
            "Not enough unique GPS points to build a valid spline. "
            "Skipping spline-based derivative calculations."
        )
        logger.warning(warning_msg)
        print(f"[WARNING] {warning_msg}")
        tck, u, derivatives = None, None, None
    else:
        # 3) Use the filtered data to build the spline with a small smoothing factor
        lat_filtered = filtered_coords[:, 0]
        lon_filtered = filtered_coords[:, 1]
        print("[INFO] Building spline on filtered data with s=1e-6 (small smoothing)...")
        try:
            tck, u = splprep([lat_filtered, lon_filtered], s=1e-6, full_output=False)
            print("[INFO] tck and u computed successfully.")
            derivatives = np.array(splev(u, tck, der=1)).T
            print(f"[INFO] Spline built successfully. Derivatives shape: {derivatives.shape}")
        except Exception as e:
            print(f"[ERROR] Spline computation failed: {e}")
            tck, u, derivatives = None, None, None

    # 4) Compute geodesic distances for every consecutive pair (using the original data)
    print("[INFO] Computing geodesic distances between consecutive points...")
    geodesic_distances = []
    for i in range(len(df) - 1):
        dist = geodesic((latitudes[i], longitudes[i]), (latitudes[i + 1], longitudes[i + 1])).meters
        geodesic_distances.append(dist)
    # Insert a zero distance for the first point (since there's no previous point)
    geodesic_distances.insert(0, 0)
    print("[DEBUG] Sample of geodesic distances:", geodesic_distances[:10])

    # 5) Add the computed distances to the DataFrame
    df["distance"] = geodesic_distances
    df["cumulative_distance"] = np.cumsum(geodesic_distances)

    print("=== data_compute_traveled_distance: END ===\n")
    return df


def data_compute_gradient(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute the gradient (slope) and gradient in per mille between consecutive points using elevation
    and horizontal distance data.

    The gradient is calculated as the difference in elevation divided by the difference in cumulative
    horizontal distance between consecutive points. The per mille value is the gradient multiplied by 1000.
    These values are stored in new columns in the DataFrame.

    Note:
        For accurate results, the horizontal distance should be the cumulative distance traveled along
        the track. If your DataFrame only contains segment distances, compute the cumulative sum first.

    Parameters:
        df (pd.DataFrame): DataFrame containing the elevation and horizontal distance data.
        config (Dict[str, Any]): Configuration dictionary with the following keys:
            - "elevation_column": (str) Name of the elevation column.
            - "horizontal_distance_column": (str) Name of the cumulative horizontal distance column
              (default "cumulative_distance").
            - "gradient_column": (str, optional) Name for the new gradient column (default "gradient").
            - "gradient_promille_column": (str, optional) Name for the new per mille column (default "gradient_promille").

    Returns:
        pd.DataFrame: The DataFrame with two additional columns:
            - 'gradient': The gradient (slope) values.
            - 'gradient_promille': The gradient values multiplied by 1000 (per mille).
    """
    # Retrieve column names from config with defaults
    elevation_col = config.get("elevation_column", "elevation")
    horizontal_distance_col = config.get("horizontal_distance_column", "cumulative_distance")
    gradient_col = config.get("gradient_column", "gradient")
    gradient_promille_col = config.get("gradient_promille_column", "gradient_promille")

    # Validate that the required columns exist in the DataFrame
    for col in [elevation_col, horizontal_distance_col]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame.")

    gradients = [0.0]           # First point has no previous point so gradient is 0
    gradients_promille = [0.0]  # Same for the per mille value

    # Compute gradient using differences in elevation and cumulative horizontal distance
    for i in range(1, len(df)):
        elev_diff = df.iloc[i][elevation_col] - df.iloc[i - 1][elevation_col]
        horiz_diff = df.iloc[i][horizontal_distance_col] - df.iloc[i - 1][horizontal_distance_col]

        # Avoid division by zero
        if horiz_diff == 0:
            grad = 0.0
        else:
            grad = elev_diff / horiz_diff

        gradients.append(grad)
        gradients_promille.append(grad * 1000)

    # Add new columns to DataFrame
    df[gradient_col] = gradients
    df[gradient_promille_col] = gradients_promille

    return df




def data_smooth_gradient(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    gradient_promille_col = config.get("gradient_promille_column", "gradient_promille")
    smoothed_col = config.get("smoothed_gradient_promille_column", "smoothed_gradient_promille")
    window_size = config.get("smoothing_window", 501)

    # Savitzky-Golay filter requires an odd window length
    if window_size % 2 == 0:
        window_size += 1

    # Apply the filter with a polynomial order of 2 (you can experiment with this)
    df[smoothed_col] = savgol_filter(df[gradient_promille_col], window_length=window_size, polyorder=2)
    return df


import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter

import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter


def moving_average(data, window_size):
    """ Compute moving average for 2D position data (x, y)."""
    if data.shape[0] < window_size:
        return data[-1]  # Return last valid entry if not enough data
    smoothed_x = np.convolve(data[:, 0], np.ones(window_size) / window_size, mode='valid')
    smoothed_y = np.convolve(data[:, 1], np.ones(window_size) / window_size, mode='valid')
    return np.array([smoothed_x[-1], smoothed_y[-1]])


def downsample_gps(df, time_col, time_step):
    """ Downsample GPS data to a lower frequency by averaging over time intervals."""
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce', infer_datetime_format=True)
    df = df.set_index(time_col).resample(f'{time_step}s').mean().dropna().reset_index()
    return df


def initialize_kalman(dt, process_noise, measurement_noise):
    """ Initialize the Kalman filter."""
    kf = KalmanFilter(dim_x=4, dim_z=2)  # 4 state variables (x, y, vx, vy), 2 measurement variables (x, y)
    kf.F = np.array([[1, 0, dt, 0],  # x position update
                     [0, 1, 0, dt],  # y position update
                     [0, 0, 1, 0],  # x velocity
                     [0, 0, 0, 1]])  # y velocity
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])  # We only measure position
    kf.Q = np.eye(4) * process_noise  # Process noise covariance
    kf.R = np.eye(2) * measurement_noise  # Measurement noise covariance
    kf.P = np.eye(4) * 100  # Large initial uncertainty
    kf.x = np.zeros((4, 1))  # Initial state vector (position and velocity)
    return kf


def data_rollWin_Kalman_on_gps(df, config):
    """ Process GPS data to remove stops and smooth with Kalman filtering using hysteresis."""
    x_col = config['x_col']
    y_col = config['y_col']
    speed_col = config['speed_column']
    time_col = config['time_column']

    REQUIRED_COLUMNS = {x_col, y_col, speed_col, time_col}

    # Validate input DataFrame
    if not REQUIRED_COLUMNS.issubset(df.columns):
        raise ValueError(f"Input DataFrame must contain the following columns: {REQUIRED_COLUMNS}")
    if df.empty:
        raise ValueError("Input DataFrame is empty. Please provide valid GPS data.")

    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

    # Downsample GPS data
    df = downsample_gps(df, time_col, time_step=1)

    # Handle missing values
    df = df.dropna(subset=[x_col, y_col, speed_col])
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.interpolate(method='linear', limit_direction='both')
    df = df.dropna().reset_index(drop=True)

    SPEED_MOVE = config['speed_move']
    SPEED_STOP = config['speed_stop']
    MOVE_DURATION = config['move_duration']
    STOP_DURATION = config['stop_duration']
    DT = config['time_step']
    PROCESS_NOISE = config['process_noise']
    MEASUREMENT_NOISE = config['measurement_noise']

    kf = initialize_kalman(DT, PROCESS_NOISE, MEASUREMENT_NOISE)
    is_moving = None
    move_timer, stop_timer = 0, 0
    filtered_positions = []
    valid_rows = []
    prev_x, prev_y = None, None

    for i, row in df.iterrows():
        if not np.isfinite(row[x_col]) or not np.isfinite(row[y_col]):
            continue  # Skip invalid rows

        current_pos = np.array([[row[x_col]], [row[y_col]]])

        if i == 0:
            kf.x = np.array([[row[x_col]], [row[y_col]], [0], [0]])
            filtered_positions.append([row[x_col], row[y_col]])
            prev_x, prev_y = row[x_col], row[y_col]
            valid_rows.append(i)
            continue

        dx, dy = row[x_col] - prev_x, row[y_col] - prev_y
        norm = np.sqrt(dx ** 2 + dy ** 2)

        if norm > 0:
            vx, vy = row[speed_col] * (dx / norm), row[speed_col] * (dy / norm)
        else:
            vx, vy = 0, 0

        prev_x, prev_y = row[x_col], row[y_col]

        if row[speed_col] > SPEED_MOVE:
            move_timer += 1
            stop_timer = 0
        elif row[speed_col] < SPEED_STOP:
            stop_timer += 1
            move_timer = 0

        if is_moving is None:
            if move_timer >= MOVE_DURATION:
                is_moving = True
            elif stop_timer >= STOP_DURATION:
                is_moving = False
        else:
            if move_timer >= MOVE_DURATION:
                is_moving = True
            elif stop_timer >= STOP_DURATION:
                is_moving = False

        if is_moving:
            kf.predict()
            kf.update(current_pos)
            filtered_pos = kf.x[:2].flatten().tolist()
            valid_rows.append(i)
        else:
            filtered_pos = moving_average(df[[x_col, y_col]].values[:i + 1], window_size=3).tolist()

        filtered_positions.append(filtered_pos)

    filtered_df = pd.DataFrame(filtered_positions, columns=['filtered_x', 'filtered_y'], index=df.index[valid_rows])
    df = df.loc[valid_rows].reset_index(drop=True)
    df = pd.concat([df, filtered_df], axis=1)

    return df
