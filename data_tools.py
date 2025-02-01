import tkinter as tk
from tkinter import ttk
from typing import Dict, Any  # Add this line to fix the error

import numpy as np
import pandas as pd
from pykalman import KalmanFilter
from pyproj import Transformer
from filterpy.monte_carlo import systematic_resample
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


import pandas as pd
from typing import Dict, Optional, Tuple
from scipy.signal import savgol_filter
x


def data_smooth_gps_savitzky(df: pd.DataFrame, smoothing_params: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    """
    Check for available GPS data (raw or preprocessed) and ask the user which to use
    for applying a Savitzky–Golay filter. The chosen data is then smoothed, and the results
    are stored in new columns:
      'GPS_lat_smoothed_savitzky' and 'GPS_lon_smoothed_savitzky'.

    Args:
        df: DataFrame containing GPS data. Expected to have either:
              - Raw data: 'GPS_lat' and 'GPS_lon'
              - Or preprocessed data with names like 'GPS_lat_smoothed_<method>' and 'GPS_lon_smoothed_<method>'
        smoothing_params: Optional dict to override default smoothing parameters
                          (default: {"window_length": 51, "polyorder": 2})

    Returns:
        The DataFrame with new smoothed columns.
    """

    # Inner helper function for GUI selection.
    def choose_from_options(title: str, prompt: str, options: list) -> str:
        selected_value = {"value": None}

        def on_ok():
            selected = combobox.get()
            if selected not in options:
                messagebox.showerror("Invalid Selection", "Please select a valid option.")
                return
            selected_value["value"] = selected
            dialog.destroy()

        dialog = tk.Tk()
        dialog.title(title)
        dialog.resizable(False, False)

        # Center the window on the screen.
        dialog.update_idletasks()
        width = 350
        height = 150
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f"{width}x{height}+{x}+{y}")

        # Prompt label.
        label = tk.Label(dialog, text=prompt)
        label.pack(pady=(20, 10))

        # Combobox for options.
        combobox = ttk.Combobox(dialog, values=options, state="readonly", width=30)
        combobox.pack(pady=5)
        combobox.current(0)  # default selection

        # OK button.
        ok_button = tk.Button(dialog, text="OK", command=on_ok)
        ok_button.pack(pady=(10, 20))

        dialog.mainloop()

        if selected_value["value"] is None:
            raise ValueError("No selection was made.")
        return selected_value["value"]

    # Build a dictionary of candidate input pairs.
    # Key: label for display; Value: tuple (lat_column, lon_column)
    candidates: Dict[str, Tuple[str, str]] = {}

    # Check for raw data.
    if "GPS_lat" in df.columns and "GPS_lon" in df.columns:
        candidates["raw (GPS_lat, GPS_lon)"] = ("GPS_lat", "GPS_lon")

    # Check for preprocessed columns.
    for col in df.columns:
        prefix = "GPS_lat_smoothed_"
        if col.startswith(prefix):
            method = col[len(prefix):]
            lon_col = f"GPS_lon_smoothed_{method}"
            if lon_col in df.columns:
                label = f"preprocessed ({method})"
                candidates[label] = (col, lon_col)

    if not candidates:
        raise KeyError("No valid GPS data found. Expected raw columns ('GPS_lat', 'GPS_lon') or preprocessed columns "
                       "with the pattern 'GPS_lat_smoothed_<method>' and 'GPS_lon_smoothed_<method>'.")

    # Create a list of options for the user.
    options = list(candidates.keys())

    # Ask the user to choose the input data.
    chosen_label = choose_from_options("Select GPS Data for Smoothing with Savitzky-Golay",
                                       "Select the GPS data to use as input for Savitzky-Golay:",
                                       options)
    lat_input, lon_input = candidates[chosen_label]
    print(f"Using input columns: {lat_input} and {lon_input}")

    # Use default smoothing parameters if not provided.
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


import pandas as pd
from typing import Dict, Optional, Tuple
from scipy.ndimage import gaussian_filter1d
import tkinter as tk
from tkinter import ttk, messagebox


def data_smooth_gps_gaussian(df: pd.DataFrame, gaussian_params: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    Check for available GPS data (raw or preprocessed) and ask the user which to use
    for applying a Gaussian filter. The chosen data is then smoothed using the Gaussian filter,
    and the results are stored in new columns:
      'GPS_lat_smooth_gaussian' and 'GPS_lon_smooth_gaussian'.

    Args:
        df: DataFrame containing GPS data. Expected to have either:
              - Raw data: 'GPS_lat' and 'GPS_lon'
              - Or preprocessed data with names like 'GPS_lat_smoothed_<method>' and 'GPS_lon_smoothed_<method>'
        gaussian_params: Optional dict to override default Gaussian parameters
                         (default: {"sigma": 2})

    Returns:
        The DataFrame with new smoothed columns.
    """

    # Inner helper function to display a GUI for option selection.
    def choose_from_options(title: str, prompt: str, options: list) -> str:
        selected_value = {"value": None}

        def on_ok():
            selected = combobox.get()
            if selected not in options:
                messagebox.showerror("Invalid Selection", "Please select a valid option.")
                return
            selected_value["value"] = selected
            dialog.destroy()

        dialog = tk.Tk()
        dialog.title(title)
        dialog.resizable(False, False)

        # Center the window on the screen.
        dialog.update_idletasks()
        width = 350
        height = 150
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f"{width}x{height}+{x}+{y}")

        # Prompt label.
        label = tk.Label(dialog, text=prompt)
        label.pack(pady=(20, 10))

        # Combobox for options.
        combobox = ttk.Combobox(dialog, values=options, state="readonly", width=30)
        combobox.pack(pady=5)
        combobox.current(0)  # default selection

        # OK button.
        ok_button = tk.Button(dialog, text="OK", command=on_ok)
        ok_button.pack(pady=(10, 20))

        dialog.mainloop()

        if selected_value["value"] is None:
            raise ValueError("No selection was made.")
        return selected_value["value"]

    # Build a dictionary of candidate input pairs.
    # Key: label for display; Value: tuple (latitude_column, longitude_column)
    candidates: Dict[str, Tuple[str, str]] = {}

    # Check for raw data.
    if "GPS_lat" in df.columns and "GPS_lon" in df.columns:
        candidates["raw (GPS_lat, GPS_lon)"] = ("GPS_lat", "GPS_lon")

    # Check for preprocessed columns.
    for col in df.columns:
        prefix = "GPS_lat_smoothed_"
        if col.startswith(prefix):
            method = col[len(prefix):]
            lon_col = f"GPS_lon_smoothed_{method}"
            if lon_col in df.columns:
                label = f"preprocessed ({method})"
                candidates[label] = (col, lon_col)

    if not candidates:
        raise KeyError("No valid GPS data found. Expected raw columns ('GPS_lat', 'GPS_lon') or preprocessed columns "
                       "with the pattern 'GPS_lat_smoothed_<method>' and 'GPS_lon_smoothed_<method>'.")

    # Create a list of options for the user.
    options = list(candidates.keys())

    # Ask the user to choose the input data.
    chosen_label = choose_from_options("Select GPS Data for Gaussian Smoothing",
                                       "Select the GPS data to use as input for Gaussian Smoothing:",
                                       options)
    lat_input, lon_input = candidates[chosen_label]
    print(f"Using input columns: {lat_input} and {lon_input}")

    # Use default Gaussian parameters if not provided.
    params = {"sigma": 2}
    if gaussian_params:
        params.update(gaussian_params)

    # Apply the Gaussian filter.
    df["GPS_lat_smoothed_gaussian"] = gaussian_filter1d(df[lat_input], sigma=params["sigma"])
    df["GPS_lon_smoothed_gaussian"] = gaussian_filter1d(df[lon_input], sigma=params["sigma"])

    print("Gaussian smoothing applied and saved as 'GPS_lat_smooth_gaussian' and 'GPS_lon_smooth_gaussian'.")
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



def data_particle_filter(df: pd.DataFrame, config: Dict[str, str]) -> pd.DataFrame:
    """
    Apply a particle filter using GPS data (latitude and longitude) along with speed and acceleration.
    This function searches for candidate GPS columns in the DataFrame. It considers:
      - Raw data: 'GPS_lat' and 'GPS_lon'
      - Preprocessed data: any pair of columns matching
            'GPS_lat_smoothed_<method>' and 'GPS_lon_smoothed_<method>'
    A simple GUI is presented (if multiple candidates exist) to let the user choose which GPS data to use.

    The state vector at each time step is:
         [latitude, longitude, speed, acceleration]
    Speed and acceleration column names are taken from config (defaults are "speed" and "acceleration").

    The function returns the original DataFrame with four additional columns:
         "pf_lat", "pf_lon", "pf_speed", "pf_acc"
    containing the filtered state estimates.

    The config dictionary may include:
       - For GPS:
         * (No explicit keys needed; the function searches for raw or preprocessed GPS columns.)
       - For speed and acceleration:
         * "speed_col": Name of the speed column (default "speed")
         * "acc_col": Name of the acceleration column (default "acceleration")
       - "N_for_particule_filter": Number of particles (default 100)
       - Process noise standard deviations:
         * "process_std_lat" (default 0.0001)
         * "process_std_lon" (default 0.0001)
         * "process_std_speed" (default 0.1)
         * "process_std_acc" (default 0.1)
       - Measurement noise standard deviations:
         * "measurement_std_lat" (default 0.0001)
         * "measurement_std_lon" (default 0.0001)
         * "measurement_std_speed" (default 0.1)
         * "measurement_std_acc" (default 0.1)
    """

    # --- Helper: GUI for selecting among candidate options ---
    def choose_from_options(title: str, prompt: str, options: list) -> str:
        selected_value = {"value": None}

        def on_ok():
            selected = combobox.get()
            if selected not in options:
                messagebox.showerror("Invalid Selection", "Please select a valid option.")
                return
            selected_value["value"] = selected
            dialog.destroy()

        dialog = tk.Tk()
        dialog.title(title)
        dialog.resizable(False, False)

        # Center the window on the screen.
        dialog.update_idletasks()
        width = 350
        height = 150
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f"{width}x{height}+{x}+{y}")

        # Prompt label.
        label = tk.Label(dialog, text=prompt)
        label.pack(pady=(20, 10))

        # Combobox for options.
        combobox = ttk.Combobox(dialog, values=options, state="readonly", width=30)
        combobox.pack(pady=5)
        combobox.current(0)  # default selection

        # OK button.
        ok_button = tk.Button(dialog, text="OK", command=on_ok)
        ok_button.pack(pady=(10, 20))

        dialog.mainloop()

        if selected_value["value"] is None:
            raise ValueError("No selection was made.")
        return selected_value["value"]

    # --- End Helper ---

    # --- Determine which GPS columns to use ---
    # Build a dictionary of candidate input pairs:
    # Key: Display label; Value: (lat_column, lon_column)
    gps_candidates = {}

    # Raw GPS columns:
    if "GPS_lat" in df.columns and "GPS_lon" in df.columns:
        gps_candidates["raw (GPS_lat, GPS_lon)"] = ("GPS_lat", "GPS_lon")

    # Look for preprocessed versions (columns with suffixes):
    for col in df.columns:
        prefix = "GPS_lat_smoothed_"
        if col.startswith(prefix):
            method = col[len(prefix):]
            lon_candidate = f"GPS_lon_smoothed_{method}"
            if lon_candidate in df.columns:
                gps_candidates[f"preprocessed ({method})"] = (col, lon_candidate)

    if not gps_candidates:
        raise KeyError("No valid GPS data columns found. Expect raw 'GPS_lat'/'GPS_lon' or preprocessed "
                       "versions with pattern 'GPS_lat_smoothed_<method>' and 'GPS_lon_smoothed_<method>'.")

    # If more than one candidate exists, ask the user which one to use.
    options = list(gps_candidates.keys())
    if len(options) > 1:
        chosen_label = choose_from_options("Select GPS Data for Particle Filter",
                                           "Select the GPS data to use as input (latitude and longitude):",
                                           options)
    else:
        chosen_label = options[0]
    gps_lat_col, gps_lon_col = gps_candidates[chosen_label]
    print(f"Using GPS columns: {gps_lat_col} and {gps_lon_col}")

    # --- Get speed and acceleration column names from config (with defaults) ---
    speed_col = config.get("speed_column")
    acc_col = config.get("acc_col_for_particule_filter")
    for col in [speed_col, acc_col]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in the DataFrame.")

    # --- Particle filter parameters ---
    N = int(config.get("N_for_particule_filter"))

    # Process noise standard deviations (for each state dimension)
    process_std = np.array([
        float(config.get("process_std_lat", 0.0001)),
        float(config.get("process_std_lon", 0.0001)),
        float(config.get("process_std_speed", 0.1)),
        float(config.get("process_std_acc", 0.1))
    ])

    # Measurement noise standard deviations
    measurement_std = np.array([
        float(config.get("measurement_std_lat", 0.0001)),
        float(config.get("measurement_std_lon", 0.0001)),
        float(config.get("measurement_std_speed", 0.1)),
        float(config.get("measurement_std_acc", 0.1))
    ])

    # --- Build the observation matrix ---
    # Each observation row: [lat, lon, speed, acceleration]
    observations = df[[gps_lat_col, gps_lon_col, speed_col, acc_col]].values
    T = observations.shape[0]
    d = 4  # state dimension

    # --- Initialize particles ---
    init_obs = observations[0]
    # Set an initial covariance (scaled from measurement noise)
    init_cov = np.diag((measurement_std ** 2) * 10)
    try:
        particles = np.random.multivariate_normal(mean=init_obs, cov=init_cov, size=N)
    except np.linalg.LinAlgError:
        # Fallback in case the covariance is not positive definite.
        particles = np.random.multivariate_normal(mean=init_obs, cov=np.eye(d) * 1e-6, size=N)

    weights = np.ones(N) / N  # uniform initial weights

    # --- Helper: Systematic resampling ---
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

    # --- End Helper ---

    # --- Particle Filter Loop ---
    filtered_states = []
    for z in observations:
        # Prediction: propagate particles with Gaussian process noise.
        noise = np.random.normal(0, process_std, size=(N, d))
        particles = particles + noise

        # Update: compute likelihood for each particle given measurement z.
        diff = particles - z  # shape (N, d)
        # Calculate squared error normalized by measurement noise.
        squared_error = np.sum((diff / measurement_std) ** 2, axis=1)
        likelihood = np.exp(-0.5 * squared_error)

        # Update weights.
        weights *= likelihood
        weights += 1.e-300  # avoid numerical underflow
        weights /= np.sum(weights)

        # Resample particles.
        indexes = systematic_resample(weights)
        particles = particles[indexes]
        weights = np.ones(N) / N  # reset weights after resampling

        # Estimate: compute the mean state from the particles.
        mean_state = np.mean(particles, axis=0)
        filtered_states.append(mean_state)

    filtered_states = np.array(filtered_states)  # shape (T, 4)

    # --- Append filtered state estimates as new columns in the DataFrame ---
    df["pf_lat"] = filtered_states[:, 0]
    df["pf_lon"] = filtered_states[:, 1]
    df["pf_speed"] = filtered_states[:, 2]
    df["pf_acc"] = filtered_states[:, 3]

    return df

