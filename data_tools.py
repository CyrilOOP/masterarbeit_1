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


import logging
import sys

# Create and configure the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Adjust the level as needed (DEBUG, INFO, WARNING, ERROR, CRITICAL)

# Optional: create a formatter with a certain format
formatter = logging.Formatter(
    fmt="%(asctime)s %(levelname)8s [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Optional: log to console (stdout)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Optional: log to file
# file_handler = logging.FileHandler("app.log")
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)

# Now anywhere in your code, you can do:
#   from <your_module> import logger
# and then
#   logger.debug("Debug message")
#   logger.info("Info message")
#   logger.warning("Warning message")
# etc.




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

def data_compute_yaw_rate_from_heading(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Compute yaw rate (in degrees/second) from heading column(s) in the DataFrame.
    Uses a local variable for time deltas (instead of writing them to the DataFrame).
    Yaw rate sign is fixed at 1 (positive).
    """
    # --- 1) Check if we have elapsed_time_s in df ---
    if "elapsed_time_s" not in df.columns:
        raise ValueError(
            "Column 'elapsed_time_s' is missing from DataFrame; "
            "cannot compute time deltas."
        )

    # Calculate time deltas locally (no column added to df)
    time_delta_s = df["elapsed_time_s"].diff()
    # Replace NaN or 0 with a tiny value to avoid division-by-zero
    time_delta_s = time_delta_s.fillna(0).replace(0, 1e-6)

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

        # Compute yaw rate = (wrapped difference) / (time delta)
        yaw_rate = heading_diff_wrapped / time_delta_s

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
    #df["distance"] = geodesic_distances
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


import numpy as np
import pandas as pd

def data_compute_curvature_radius_and_detect_steady_curves(
    df: pd.DataFrame, config: dict = None
) -> pd.DataFrame:
    """
    1) Compute signed curvature & radius (with sign).
    2) Apply 'straight line' threshold if radius > e.g. 11000.
    3) Segment the data using a variable-window approach for 'steady curves'.
    4) Optionally compute mean radius/curvature for each segment.
    """
    if config is None:
        config = {}

    # --- Step 1) & 2) Just like before ---
    def compute_radius_and_curvature_with_threshold(df_inner: pd.DataFrame, cfg: dict) -> None:
        distance_col = cfg.get("distance_col", "cumulative_distance")
        heading_col = cfg.get("heading_col", "heading_deg_ds")
        radius_col = cfg.get("radius_col", "radius_m")
        curvature_col = cfg.get("curvature_col", "curvature")
        straight_threshold = cfg.get("straight_threshold", 11000.0)

        heading_diff_deg = df_inner[heading_col].diff()
        heading_diff_deg = (heading_diff_deg + 180) % 360 - 180
        heading_diff_deg = heading_diff_deg.fillna(0)

        heading_diff_rad = heading_diff_deg * (np.pi / 180.0)
        heading_diff_rad_no_zero = heading_diff_rad.replace(0, np.nan)

        radius = df_inner[distance_col] / heading_diff_rad_no_zero
        radius = radius.fillna(np.inf)
        curvature = heading_diff_rad_no_zero / df_inner[distance_col]
        curvature = curvature.fillna(0)

        # apply threshold for "straight line"
        too_large = radius.abs() > straight_threshold
        radius[too_large] = np.inf
        curvature[too_large] = 0.0

        df_inner[radius_col] = radius
        df_inner[curvature_col] = curvature

    # --- Step 3) Dynamic "Steady" segmentation ---
    def detect_steady_curves_variable_window(df_inner: pd.DataFrame, cfg: dict) -> None:
        curvature_col = cfg.get("curvature_col", "curvature")
        std_threshold = cfg.get("curvature_std_thresh", 0.0001)
        min_segment_size = cfg.get("min_segment_size", 3)  # e.g. 5 points

        c = df_inner[curvature_col].to_numpy()
        n = len(c)

        steady_group_ids = np.zeros(n, dtype=int)  # store group IDs
        group_id = 0

        start_idx = 0
        for i in range(n):
            seg_curv = c[start_idx : i + 1]
            curv_std = np.nanstd(seg_curv)  # or np.std(...) if no NaNs

            if curv_std <= std_threshold:
                # keep extending
                pass
            else:
                # finalize up to i-1
                seg_length = i - start_idx
                if seg_length >= min_segment_size:
                    group_id += 1
                    steady_group_ids[start_idx : i] = group_id
                # start new group from i
                start_idx = i

        # finalize last segment
        seg_length = n - start_idx
        if seg_length >= min_segment_size:
            group_id += 1
            steady_group_ids[start_idx : n] = group_id

        df_inner["steady_group"] = steady_group_ids

    # --- Step 4) Optionally compute mean radius/curvature for each segment
    def compute_segment_means(df_inner: pd.DataFrame, cfg: dict) -> None:
        radius_col = cfg.get("radius_col", "radius_m")
        curvature_col = cfg.get("curvature_col", "curvature")

        df_inner["steady_mean_radius"] = (
            df_inner.groupby("steady_group")[radius_col].transform("mean")
        )
        df_inner["steady_mean_curvature"] = (
            df_inner.groupby("steady_group")[curvature_col].transform("mean")
        )

    # -------------------------------------------
    # 1) & 2) radius/curvature with threshold
    compute_radius_and_curvature_with_threshold(df, config)

    # 3) dynamic "steady" detection
    detect_steady_curves_variable_window(df, config)

    # 4) segment means
    compute_segment_means(df, config)

    return df
