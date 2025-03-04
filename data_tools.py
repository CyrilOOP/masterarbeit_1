"""
Example Script: Data Preprocessing & Analysis with Logging and Tkinter GUIs
---------------------------------------------------------------------------
This script illustrates a collection of data processing functions for:
 - GPS coordinate conversions
 - Time parsing and computation
 - Heading & yaw rate calculations
 - Data filtering with quantiles
 - Rolling-window segmenting
 - Elevation fetching (with concurrency)
 - Distance and gradient calculations
 - Curvature & radius detection

Author: You
Last Update: 2025-02-22
"""

# =============================================================================
# 1) IMPORTS
# =============================================================================
# --- Standard Library ---
import os
import sys
import logging
import concurrent.futures
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

# --- Third-party Libraries ---
import requests
import pandas as pd
import pyproj
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from geopy.distance import geodesic
from scipy.signal import savgol_filter

# --- Local Imports ---
from csv_tools import csv_select_gps_columns

# =============================================================================
# 2) LOGGER CONFIGURATION
# =============================================================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Adjust level as needed (DEBUG, INFO, WARNING, ERROR, CRITICAL)

# Optional: log to console (stdout)
formatter = logging.Formatter(
    fmt="%(asctime)s %(levelname)8s [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Optional: log to file
# file_handler = logging.FileHandler("app.log")
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)


# =============================================================================
# 3) FUNCTION DEFINITIONS
# =============================================================================

def data_convert_to_planar(df: pd.DataFrame, config: Dict[str, str]) -> pd.DataFrame:
    """
    Convert latitude and longitude to planar coordinates (UTM).
    Uses the helper to select which GPS columns to use.

    Adds a new column 'projection_epsg' to store the projection used.

    Args:
        df: The input DataFrame containing GPS data.
        config: A configuration dictionary.

    Returns:
        The DataFrame with added planar coordinates (x, y),
        projection information, and a column 'selected_smoothing_method'.
    """
    print("\n[FUNCTION] data_convert_to_planar: START")


    x_col = config["x"]
    y_col = config["y"]


    # Use helper to select GPS columns.
    lat_input, lon_input = csv_select_gps_columns(
        df,
        title="Select GPS Data for Planar Conversion",
        prompt="Select the GPS data to use for planar conversion:"
    )
    print(f"  > Using input columns: {lat_input} and {lon_input}")

    # Determine the selected smoothing method
    selected_method = lat_input.split("smoothed_")[-1] if "smoothed" in lat_input else "raw"

    # Dynamically determine UTM zone based on mean longitude
    utm_zone = int((df[lon_input].mean() + 180) / 6) + 1
    is_northern = df[lat_input].mean() >= 0
    epsg_code = f"EPSG:{32600 + utm_zone if is_northern else 32700 + utm_zone}"
    print(f"  > UTM zone = {utm_zone}, EPSG = {epsg_code}")

    # Create Transformer
    transformer = pyproj.Transformer.from_crs("EPSG:4326", epsg_code, always_xy=True)

    # Convert coordinates
    df[x_col], df[y_col] = transformer.transform(df[lon_input].values, df[lat_input].values)

    # Add selected method
    df["selected_smoothing_method"] = selected_method

    # Add projection column
    df["projection_epsg"] = epsg_code

    print("[FUNCTION] data_convert_to_planar: END\n")

    return df

def data_parse_time(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Parse the given datetime column as pandas datetime and compute:
      - A time-based column relative to the first timestamp.

    Args:
        df: Input DataFrame.
        config: Configuration dictionary containing:
            - "datetime_col": Name of the column containing datetime information.
            - "elapsed_time_col": Name of the new column storing elapsed time.

    Returns:
        A copy of the DataFrame with the new elapsed time column.

    Raises:
        ValueError: If the datetime column cannot be parsed.
    """
    print("\n[FUNCTION] data_parse_time: START")

    datetime_col = config["DatumZeit"]
    elapsed_time_col = config["elapsed_time_s"]  # <-- Now configurable!

    df = df.copy()

    # Convert the column to datetime using explicit format (including milliseconds)
    try:
        df[datetime_col] = pd.to_datetime(df[datetime_col], format="%Y-%m-%d %H:%M:%S.%f")
    except Exception as e:
        raise ValueError(f"Error parsing datetime column '{datetime_col}': {e}")

    # Create a column with time in seconds relative to the first timestamp
    df[elapsed_time_col] = (df[datetime_col] - df[datetime_col].iloc[0]).dt.total_seconds()

    print(f"  > datetime_col = {datetime_col}")
    print(f"  > elapsed_time_col = {elapsed_time_col}")
    print("[FUNCTION] data_parse_time: END\n")

    return df


def data_compute_heading_dx_dy(df: pd.DataFrame, config: Dict[str, str]) -> pd.DataFrame:
    """
    Compute the heading based on the train's path using differences in coordinates.
    Heading is relative to north (0° = north). This is a custom approach.

    Args:
        df: DataFrame containing the coordinate data.
        config: Configuration dictionary with keys:
            - "x_col": Name of the x-coordinate column (default: "x").
            - "y_col": Name of the y-coordinate column (default: "y").
            - "heading_col": Name of the output heading column (default: "heading_dx_dy").

    Returns:
        The DataFrame with a new column for the path-based heading.
    """
    print("\n[FUNCTION] data_compute_heading_dx_dy: START")
    import numpy as np

    x_col = config["x"]
    y_col = config["y"]
    heading_col = config["heading_dx_dy_grad"]

    # Validate that required columns exist
    for col in [x_col, y_col]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    n = len(df)
    heading = np.full(n, np.nan)  # Fill with NaN by default

    x = df[x_col].values
    y = df[y_col].values

    # Compute heading
    for i in range(3, n - 3):
        x_before = np.mean(x[i - 3:i])
        y_before = np.mean(y[i - 3:i])
        x_after = np.mean(x[i + 1:i + 4])
        y_after = np.mean(y[i + 1:i + 4])

        dx = x_after - x_before
        dy = y_after - y_before

        heading[i] = np.degrees(np.arctan2(dx, dy)) % 360

    df[heading_col] = heading

    print(f"  > New heading column: {heading_col}")
    print("[FUNCTION] data_compute_heading_dx_dy: END\n")
    return df


def data_compute_heading_ds(df: pd.DataFrame, config: Dict[str, str]) -> pd.DataFrame:
    """
    Compute the heading (yaw angle in degrees) using a derivative approach wrt cumulative distance.
    0° = north. Columns used:
      - x_col
      - y_col
      - cum_dist_col
      - heading_col_ds

    Returns:
        df with new heading column: heading_deg_ds
    """
    print("\n[FUNCTION] data_compute_heading_ds: START")
    import numpy as np

    x_col = config["x"]
    y_col = config["y"]
    cum_dist_col = config["cumulative_distance_m"]
    heading_col = config["heading_dx_ds_grad"]

    # Validate
    for col in [x_col, y_col, cum_dist_col]:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain column '{col}'.")

    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()
    s = df[cum_dist_col].to_numpy()

    # Fix strictly increasing distances
    eps = 1e-6
    s_fixed = s.copy()
    for i in range(1, len(s_fixed)):
        if s_fixed[i] <= s_fixed[i - 1]:
            s_fixed[i] = s_fixed[i - 1] + eps

    # Compute derivatives
    dx_ds = np.gradient(x, s_fixed)
    dy_ds = np.gradient(y, s_fixed)

    # Compute heading in degrees
    heading = np.degrees(np.arctan2(dx_ds, dy_ds))
    heading = (heading + 360) % 360

    df[heading_col] = heading

    print(f"  > Heading column created: {heading_col}")
    print("[FUNCTION] data_compute_heading_ds: END\n")
    return df


def select_heading_columns_gui(heading_candidates, caller):
    """
    Opens a Tkinter GUI with a multi-select Listbox for heading columns.
    Returns a list of selected column names.
    """
    print("[GUI] select_heading_columns_gui: START")
    selected = []

    def on_ok():
        nonlocal selected
        indices = listbox.curselection()
        selected = [heading_candidates[i] for i in indices]
        root.destroy()

    root = tk.Tk()
    root.title(f"Select Heading Columns for the {caller}")
    root.geometry("400x300")
    root.minsize(300, 200)

    root.columnconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)

    label = ttk.Label(root, text=f"Select heading column(s) for the {caller}:")
    label.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")

    list_frame = ttk.Frame(root)
    list_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
    list_frame.columnconfigure(0, weight=1)
    list_frame.rowconfigure(0, weight=1)

    scrollbar = ttk.Scrollbar(list_frame, orient="vertical")
    scrollbar.grid(row=0, column=1, sticky="ns")

    listbox = tk.Listbox(list_frame, selectmode="multiple", yscrollcommand=scrollbar.set)
    listbox.grid(row=0, column=0, sticky="nsew")
    scrollbar.config(command=listbox.yview)

    # Insert heading candidates
    for col in heading_candidates:
        listbox.insert(tk.END, col)

    button_frame = ttk.Frame(root)
    button_frame.grid(row=2, column=0, padx=10, pady=(5, 10), sticky="ew")
    button_frame.columnconfigure(0, weight=1)

    ok_button = ttk.Button(button_frame, text="OK", command=on_ok)
    ok_button.grid(row=0, column=0, sticky="ew")

    root.mainloop()
    print("[GUI] select_heading_columns_gui: END")
    return selected


def data_compute_yaw_rate_from_heading(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Compute yaw rate (in degrees/second) from heading column(s) in the DataFrame.
    Uses 'elapsed_time_s' for time deltas. Yaw rate sign is forced positive by default.
    """
    print("\n[FUNCTION] data_compute_yaw_rate_from_heading: START")

    elapse_time = config["elapsed_time_s"]

    # Validate
    for col in [elapse_time]:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain column '{col}'.")

    time_delta_s = df[elapse_time].diff()
    time_delta_s = time_delta_s.fillna(0).replace(0, 1e-6)

    # Find candidate heading columns
    heading_candidates = [col for col in df.columns if "heading" in col.lower()]
    if not heading_candidates:
        messagebox.showinfo("Yaw Rate Computation", "No heading columns found in DataFrame.")
        print("[FUNCTION] data_compute_yaw_rate_from_heading: END\n")
        return df

    # If there's only one heading column, use it; otherwise, prompt user
    if len(heading_candidates) == 1:
        selected_heading = heading_candidates
        print(f"  > Only one heading column found: '{heading_candidates[0]}'. Using it.")
    else:
        selected_heading = select_heading_columns_gui(heading_candidates, caller="yaw rate computing")
        if not selected_heading:
            messagebox.showwarning("Yaw Rate Computation", "No heading columns selected.")
            print("[FUNCTION] data_compute_yaw_rate_from_heading: END\n")
            return df

    for heading_col in selected_heading:
        heading_diff = df[heading_col].diff()
        heading_diff_wrapped = (heading_diff + 180) % 360 - 180  # [-180, 180]
        yaw_rate = heading_diff_wrapped / time_delta_s
        new_col_name = f"yaw_rate_from_{heading_col}"
        df[new_col_name] = yaw_rate

        if "segment_marker" in df.columns:
            stop_indices = df["segment_marker"].isin(["STOP_START", "STOP_END"])
            df.loc[stop_indices, new_col_name] = 0

    # Optionally remove the first row after STOP_END
    if "segment_marker" in df.columns:
        indices_to_drop = []
        for idx in df[df["segment_marker"] == "STOP_END"].index:
            next_idx = idx + 1
            if next_idx in df.index and df.loc[next_idx, "segment_marker"] != "STOP_END":
                indices_to_drop.append(next_idx)
        if indices_to_drop:
            df = df.drop(indices_to_drop).reset_index(drop=True)

    print("[FUNCTION] data_compute_yaw_rate_from_heading: END\n")
    return df


def data_delete_the_one_percent(df: pd.DataFrame, config: Dict[str, str]) -> pd.DataFrame:
    """
    Delete rows with extreme yaw rate values based on lower/upper quantiles.
    If multiple yaw rate columns exist and none is specified in config,
    a nested GUI will prompt for which to use.
    """
    print("\n[FUNCTION] data_delete_the_one_percent: START")

    date_col = config["DatumZeit"]
    input_lower_bound = config["delete_lower_bound_percentage"] / 100.0
    input_upper_bound = config["delete_upper_bound_percentage"] / 100.0


    def select_yaw_rate_column_gui(candidates):
        root = tk.Tk()
        root.title("Select Yaw Rate Column")
        root.geometry("300x200")
        selected = tk.StringVar(root, value=candidates[0])

        label = ttk.Label(root, text="Select a yaw rate column for filtering:")
        label.pack(padx=10, pady=10)

        for candidate in candidates:
            rb = ttk.Radiobutton(root, text=candidate, variable=selected, value=candidate)
            rb.pack(anchor="w", padx=10)

        def on_submit():
            root.destroy()

        submit_button = ttk.Button(root, text="OK", command=on_submit)
        submit_button.pack(pady=10)

        root.mainloop()
        return selected.get()

    print(f"  > Lower quantile bound: {input_lower_bound} (fraction)")
    print(f"  > Upper quantile bound: {input_upper_bound} (fraction)")

    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' is missing from the CSV file.")

    candidates = [col for col in df.columns if "yaw_rate" in col.lower()]

    if len(candidates) == 0:
            raise ValueError("No yaw rate columns found in the CSV file.")
    elif len(candidates) == 1:
            yaw_rate_col = candidates[0]
            print(f"  > Only one yaw rate column: '{yaw_rate_col}'. Using it.")
    else:
            yaw_rate_col = select_yaw_rate_column_gui(candidates)
            print(f"  > You selected '{yaw_rate_col}'.")

    lower_bound = df[yaw_rate_col].quantile(input_lower_bound)
    upper_bound = df[yaw_rate_col].quantile(input_upper_bound)
    print(f"  > Filtering out values < {lower_bound} or > {upper_bound} in '{yaw_rate_col}'")

    df = df[(df[yaw_rate_col] >= lower_bound) & (df[yaw_rate_col] <= upper_bound)]
    print("[FUNCTION] data_delete_the_one_percent: END\n")
    return df


def data_rolling_windows_gps_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Process GPS data into rolling windows based on speed classes, with additional
    filtering and hysteresis to reduce noise. Creates 'segment_marker' columns.
    """
    print("\n[FUNCTION] data_rolling_windows_gps_data: START")
    import numpy as np

    elapsed_time = config["elapsed_time_s"]
    speed_col = config["speed_column"]
    speed_threshold_stopped = config["speed_threshold_stopped_rolling_windows"]
    distance_window_meters = config["distance_window_meters"]
    time_window_min = config["time_window_min"]
    time_window_max = config["time_window_max"]
    max_stop_window = config["max_stop_window"]
    speed_bins = config["speed_bins"]
    hysteresis_window = config["hysteresis_window"]

    lat_col_rol_win, lon_col_rol_win = csv_select_gps_columns(
        df,
        title="Select GPS Data for Rolling Windows",
        prompt="Select the GPS data to use \n as input for rolling windows:"
    )
    print(f"  > Using lat_col_rol_win='{lat_col_rol_win}', lon_col_rol_win='{lon_col_rol_win}'")

    lat_arr = df[lat_col_rol_win].to_numpy(dtype=float)
    lon_arr = df[lon_col_rol_win].to_numpy(dtype=float)
    t_arr = df[elapsed_time].values

    # Rolling (smoothed) speed
    spd_arr = df[speed_col].rolling(window=5, center=True, min_periods=1).mean().to_numpy(dtype=float)
    n = len(df)

    def get_speed_class(spd, bins):
        for i in range(len(bins) - 1):
            if bins[i] <= spd < bins[i + 1]:
                return i
        return len(bins) - 1

    def get_window_length(speed_value):
        if speed_value < speed_threshold_stopped:
            return None
        raw_window = distance_window_meters / (speed_value + 1e-6)
        return max(time_window_min, min(time_window_max, raw_window))


    grouped_rows = []
    i = 0

    while i < n:
        current_speed = spd_arr[i]
        current_class = get_speed_class(current_speed, speed_bins)

        # A) Stopped
        if current_class == 0:
            sum_lat, sum_lon, sum_spd = 0.0, 0.0, 0.0
            count = 0
            j = i
            while j < n:
                if get_speed_class(spd_arr[j], speed_bins) != 0:
                    end_idx = min(n, j + hysteresis_window)
                    if np.mean(spd_arr[j:end_idx]) >= speed_threshold_stopped:
                        break

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

            start_row_dict = df.iloc[i].copy().to_dict()
            #start_row_dict["time_numeric"] = t_arr[i]
            start_row_dict[speed_col] = mean_spd
            start_row_dict["GPS_lat_smoothed_rolling_windows"] = mean_lat
            start_row_dict["GPS_lon_smoothed_rolling_windows"] = mean_lon
            start_row_dict["segment_marker"] = "STOP_START"
            grouped_rows.append(start_row_dict)

            end_row_dict = df.iloc[j - 1].copy().to_dict()
       #    end_row_dict["time_numeric"] = t_arr[j - 1]
            end_row_dict[speed_col] = mean_spd
            end_row_dict["GPS_lat_smoothed_rolling_windows"] = mean_lat
            end_row_dict["GPS_lon_smoothed_rolling_windows"] = mean_lon
            end_row_dict["segment_marker"] = "STOP_END"
            grouped_rows.append(end_row_dict)

            i = j

        # B) Moving
        else:
            initial_speed = spd_arr[i]
            wlen = get_window_length(initial_speed)
            if wlen is None:
                wlen = time_window_min
            window_end = t_arr[i] + wlen

            sum_lat, sum_lon, sum_spd = 0.0, 0.0, 0.0
            count = 0
            j = i

            while j < n:
                if get_speed_class(spd_arr[j], speed_bins) != current_class:
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

            row_dict = df.iloc[i].copy().to_dict()
            midpoint_time = 0.5 * (t_arr[i] + t_arr[j - 1])
           # row_dict["time_numeric"] = midpoint_time
            row_dict[speed_col] = mean_spd
            row_dict["GPS_lat_smoothed_rolling_windows"] = mean_lat
            row_dict["GPS_lon_smoothed_rolling_windows"] = mean_lon
            speed_class = get_speed_class(mean_spd, speed_bins)  # Get the speed class
            row_dict["segment_marker"] = f"MOVING_{speed_class}"  # Append class info
            grouped_rows.append(row_dict)

            i = j

    df_grouped = pd.DataFrame(grouped_rows)
    print("[FUNCTION] data_rolling_windows_gps_data: END\n")
    return df_grouped


def data_get_elevation(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Fetch elevation data for GPS coordinates in a DataFrame using an external API.
    Appends the elevation data as a new column.

    Keys in config:
      - api_key
      - elevation_column
      - api_url
      - batch_size
      - threads
    """
    print("\n[FUNCTION] data_get_elevation: START")
    root = tk.Tk()
    root.withdraw()

    response = messagebox.askyesno(
        "Warning: API Key Usage",
        "⚠️ WARNING: This script uses a private API key.\n\n"
        "Do you want to proceed?"
    )
    root.destroy()

    if not response:
        exit("Execution aborted by user.")

    logger.info("Continuing with the script execution...")

    required_keys = ["api_key", "elevation", "api_url", "batch_size", "threads"]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(f"Missing configuration keys: {missing_keys}")

    api_key = config["api_key"]
    elevation_column = config["elevation"]
    api_url = config["api_url"]
    batch_size = config["batch_size"]
    threads = config["threads"]

    if df.empty:
        raise ValueError("DataFrame is empty.")

    lat_col, lon_col = csv_select_gps_columns(
        df,
        title="Select GPS Data for Elevation",
        prompt="Select the GPS data to use as input for elevation:"
    )
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError("Selected GPS columns not found in DataFrame.")

    logger.info(f"Using GPS columns: {lat_col} and {lon_col}")

    coords = list(zip(df[lat_col], df[lon_col]))
    total_rows = len(coords)

    elevations = [None] * total_rows
    batch_indices = [
        list(range(i, min(i + batch_size, total_rows)))
        for i in range(0, total_rows, batch_size)
    ]
    batches = [
        coords[indices[0]: indices[-1] + 1] for indices in batch_indices
    ]

    logger.info(f"Processing {len(batches)} batches with {threads} threads...")

    def get_elevation_batch(coords_batch: List[Tuple[Any, Any]]) -> List[Any]:
        locations_str = "|".join(f"{lat},{lon}" for lat, lon in coords_batch)
        params = {"locations": locations_str, "key": api_key}
        try:
            response = requests.get(api_url, params=params)
            response.raise_for_status()
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

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        results = list(executor.map(get_elevation_batch, batches))

    for batch_idx, batch_elevations in enumerate(results):
        for idx, elevation in zip(batch_indices[batch_idx], batch_elevations):
            elevations[idx] = elevation

    df[elevation_column] = elevations
    print(f"  > Created new elevation column: '{elevation_column}'")

    print("[FUNCTION] data_get_elevation: END\n")
    return df


def data_compute_traveled_distance(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute the traveled distance between consecutive GPS points using geodesic distances.
    Adds 'cumulative_distance_m' column.

    If possible, also attempt a spline-based approach, but only if enough unique points exist.
    """
    print("\n[FUNCTION] data_compute_traveled_distance: START")

    distance = config["cumulative_distance_m"]

    lat_col, lon_col = csv_select_gps_columns(
        df,
        title="Select GPS Data for Distances",
        prompt="Select the GPS data to use as input \n to calculate the distance traveled:"
    )
    print(f"  > Selected lat_col='{lat_col}', lon_col='{lon_col}'")

    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"Selected GPS columns '{lat_col}' or '{lon_col}' not found.")

    latitudes = df[lat_col].values
    longitudes = df[lon_col].values

    print(f"  > Number of rows in df: {len(df)}")

    coords = np.column_stack((latitudes, longitudes))
    mask = np.ones(len(coords), dtype=bool)
    mask[1:] = (coords[1:] != coords[:-1]).any(axis=1)
    filtered_coords = coords[mask]
    unique_coords = np.unique(filtered_coords, axis=0)
    print(f"  > Filtered data has {len(filtered_coords)} points; unique: {len(unique_coords)}.")

    # Attempt a spline if we have enough unique points
    tck, u, derivatives = None, None, None
    if len(unique_coords) >= 3:
        from scipy.interpolate import splprep, splev
        try:
            lat_filtered = filtered_coords[:, 0]
            lon_filtered = filtered_coords[:, 1]
            print("  > Building spline on filtered data (s=1e-6)...")
            tck, u = splprep([lat_filtered, lon_filtered], s=1e-6, full_output=False)
            derivatives = np.array(splev(u, tck, der=1)).T
            print(f"  > Spline built successfully. Derivatives shape: {derivatives.shape}")
        except Exception as e:
            print(f"  > [ERROR] Spline computation failed: {e}")
    else:
        print("  > Not enough points for spline. Skipping spline-based approach.")

    # Compute geodesic distances for every consecutive pair
    geodesic_distances = []
    for i in range(len(df) - 1):
        dist = geodesic((latitudes[i], longitudes[i]),
                        (latitudes[i + 1], longitudes[i + 1])).meters
        geodesic_distances.append(dist)
    geodesic_distances.insert(0, 0)

    df[distance] = np.cumsum(geodesic_distances)
    print(f"  > Added {distance} column. Sample: {df[distance].head()}")

    print("[FUNCTION] data_compute_traveled_distance: END\n")
    return df


def data_compute_gradient(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute the gradient (slope) and gradient in per mille between consecutive points using elevation
    and horizontal distance. Adds columns 'gradient' and 'gradient_promille' by default.
    """
    print("\n[FUNCTION] data_compute_gradient: START")
    elevation_col = config["elevation"]
    distance_col = config["cumulative_distance_m"]
    gradient_col = config["gradient"]
    gradient_promille_col = config["gradient_per_mille"]

    for col in [elevation_col, distance_col]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame.")

    gradients = [0.0]
    gradients_promille = [0.0]

    for i in range(1, len(df)):
        elev_diff = df.iloc[i][elevation_col] - df.iloc[i - 1][elevation_col]
        horiz_diff = df.iloc[i][distance_col] - df.iloc[i - 1][distance_col]
        grad = elev_diff / horiz_diff if horiz_diff != 0 else 0.0
        gradients.append(grad)
        gradients_promille.append(grad * 1000)
        

    df[gradient_col] = gradients
    df[gradient_promille_col] = gradients_promille

    print(f"  > Added columns: '{gradient_col}', '{gradient_promille_col}'")
    print("[FUNCTION] data_compute_gradient: END\n")
    return df


def data_compute_curvature_and_radius(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Computes curvature and radius from heading data for the selected heading columns.
    
    Process:
      1. Identify candidate heading columns in df that contain "heading" but not "yaw_rate".
      2. Use the select_heading_columns_gui() function to choose one or more heading columns.
      3. For each selected heading column:
         - Compute the heading differences (normalized to [-180, 180] degrees).
         - Convert the differences to radians.
         - Compute the incremental distance (ds) using the differences of the cumulative distance column.
         - Calculate:
                curvature = heading_diff_rad / ds
                radius = ds / heading_diff_rad
         - Apply the straight-line threshold: if |radius| > straight_threshold, set radius = inf and curvature = 0.
         - Store the computed radius and curvature in new uniquely named columns.
    
    The config dictionary is read using config["key"] notation.
    """
    print("[FUNCTION] data_compute_curvature_and_radius: START")

    # Identify candidate heading columns: include columns with "heading" but exclude those with "yaw_rate"
    heading_candidates = [col for col in df.columns
                          if "heading" in col.lower() and "yaw_rate" not in col.lower()]
    if not heading_candidates:
        print("[ERROR] No appropriate heading columns found in DataFrame.")
        return df

    # Use GUI to select heading columns
    selected_headings = select_heading_columns_gui(heading_candidates, caller="curvature and radius")
    if not selected_headings:
        print("[ERROR] No heading columns selected.")
        return df

    # Read configuration parameters
    cum_dist_col = config["cumulative_distance_m"]
    radius_col_base = config["radius_m"]
    curvature_col_base = config["curvature"]
    straight_threshold = config["straight_threshold"]

    # Compute incremental distance (ds) from the cumulative distance column
    ds = df[cum_dist_col].diff()
    ds.fillna(method="bfill", inplace=True)  # fill the first NaN with the next valid value

    for heading_col in selected_headings:
        print(f"[INFO] Processing heading column: {heading_col}")

        # Compute differences in heading (in degrees) and normalize to [-180, 180]
        heading_diff_deg = df[heading_col].diff()
        heading_diff_deg = (heading_diff_deg + 180) % 360 - 180
        heading_diff_deg = heading_diff_deg.fillna(0)
        print(f"[DEBUG] {heading_col} - Heading differences (deg) (first 5):")
        print(heading_diff_deg.head())

        # Convert differences to radians
        heading_diff_rad = heading_diff_deg * (np.pi / 180.0)
        print(f"[DEBUG] {heading_col} - Heading differences (rad) (first 5):")
        print(heading_diff_rad.head())

        # Replace zeros with NaN to avoid division by zero
        heading_diff_rad_no_zero = heading_diff_rad.replace(0, np.nan)

        # Compute curvature and radius using the incremental distance ds
        curvature = heading_diff_rad_no_zero / ds
        curvature = curvature.fillna(0)
        radius = ds / heading_diff_rad_no_zero
        radius = radius.fillna(np.inf)

        print(f"[DEBUG] {heading_col} - Computed raw radius (first 5):")
        print(radius.head())
        print(f"[DEBUG] {heading_col} - Computed raw curvature (first 5):")
        print(curvature.head())

        # # Apply the straight-line threshold: if |radius| > straight_threshold, treat as straight line
        # too_large = radius.abs() > straight_threshold
        # if too_large.sum() > 0:
        #     print(f"[DEBUG] {too_large.sum()} points in {heading_col} exceed threshold {straight_threshold}.")
        #     print(f"[DEBUG] Setting radius to inf and curvature to 0 for those points.")
        # radius[too_large] = np.inf
        # curvature[too_large] = 0.0

        # Create unique output column names for this heading column
        out_radius_col = f"{radius_col_base}_{heading_col}"
        out_curvature_col = f"{curvature_col_base}_{heading_col}"

        # Store computed values in the DataFrame
        df[out_radius_col] = radius
        df[out_curvature_col] = curvature
        print(f"[INFO] Completed processing for {heading_col}. Output columns: {out_radius_col}, {out_curvature_col}\n")

    print("[FUNCTION] data_compute_curvature_and_radius: END")
    return df




def data_segment_train_curves3(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Segments a railway path into:
      - "straight line": near-zero curvature
      - "Übergangsbogen": transition segments with large d(kappa)/ds
      - "curve": segments with fairly constant, nonzero curvature

    Now uses "cumulative_distance_m" to compute slope w.r.t. distance, not index.

    Required keys in 'config':
      - "curvature": column name for computed curvature
      - "radius": column name for computed radius
      - "smoothing_window_nope": window size for smoothing curvature
      - "straight_threshold_nope": max abs curvature for straight
      - "transition_slope_threshold_nope": min abs slope for transition
      - "curve_slope_threshold_nope": max abs slope for a steady curve
      - "min_segment_size_nope": minimum number of points in a segment
      - "radius_nope": column name for the computed radius
      - (Additionally, the DataFrame must have "cumulative_distance_m")
    """

    # 1) Unpack config
    curvature_col = config.get("curvature_nope", "curvature_heading_dx_ds_grad")
    radius_col = config.get("radius_nope", "radius_m_heading_dx_ds_grad")
    smoothing_window = config.get("smoothing_window_nope", 5)
    straight_threshold = config.get("straight_threshold_nope", 0.001)
    transition_slope_threshold = config.get("transition_slope_threshold_nope", 0.005)
    curve_slope_threshold = config.get("curve_slope_threshold_nope", 0.001)
    min_segment_size = config.get("min_segment_size_nope", 3)

    # Ensure 'cumulative_distance_m' is in df
    if "cumulative_distance_m" not in df.columns:
        raise ValueError("DataFrame must have a 'cumulative_distance_m' column for distance-based slope.")

    # 2) Smooth curvature (using Savitzky–Golay or rolling mean)
    #    Savitzky–Golay requires an odd window_length:
    if smoothing_window % 2 == 0:
        smoothing_window += 1

    # For demonstration, let's use Savitzky–Golay for curvature:
    from scipy.signal import savgol_filter
    polyorder = 2  # adjust if needed
    curvature_array = df[curvature_col].values

    curvature_smoothed = savgol_filter(
        curvature_array,
        window_length=smoothing_window,
        polyorder=polyorder,
        deriv=0
    )
    df["curvature_smoothed"] = curvature_smoothed

    # 3) Compute slope wrt distance using a centered difference:
    #    slope_i = (curv_{i+1} - curv_{i-1}) / (dist_{i+1} - dist_{i-1})
    #    We'll handle edges with forward/backward diff.

    # Shift arrays:
    curv_shift_fwd = df["curvature_smoothed"].shift(-1)
    curv_shift_bwd = df["curvature_smoothed"].shift(1)
    dist_shift_fwd = df["cumulative_distance_m"].shift(-1)
    dist_shift_bwd = df["cumulative_distance_m"].shift(1)

    # Centered difference:
    df["curvature_slope"] = (
        (curv_shift_fwd - curv_shift_bwd) /
        (dist_shift_fwd - dist_shift_bwd)
    )

    # For the very first and last row, we can't do a centered diff.
    # We'll fill them by forward/backward difference or just NaN:
    df["curvature_slope"].iloc[0] = (
        (df["curvature_smoothed"].iloc[1] - df["curvature_smoothed"].iloc[0]) /
        (df["cumulative_distance_m"].iloc[1] - df["cumulative_distance_m"].iloc[0])
    )
    df["curvature_slope"].iloc[-1] = (
        (df["curvature_smoothed"].iloc[-1] - df["curvature_smoothed"].iloc[-2]) /
        (df["cumulative_distance_m"].iloc[-1] - df["cumulative_distance_m"].iloc[-2])
    )

    # 4) (Optional) Smooth the slope a bit more if needed:
    df["curvature_slope"] = df["curvature_slope"].rolling(
        window=5, center=True, min_periods=1
    ).mean()

    # 5) Classify each point
    states = []
    for i in range(len(df)):
        curv = df["curvature_smoothed"].iloc[i]
        slope = df["curvature_slope"].iloc[i]

        abs_curv = abs(curv)
        abs_slope = abs(slope)

        # Example logic: prioritize transition detection if slope is large
        if abs_slope >= transition_slope_threshold:
            candidate_state = "Übergangsbogen"
        elif abs_curv < straight_threshold:
            candidate_state = "straight line"
        else:
            # Slope is not large, curvature is not near-zero => curve
            if abs_slope <= curve_slope_threshold:
                candidate_state = "curve"
            else:
                candidate_state = "curve"

        states.append(candidate_state)

    df["initial_state"] = states

    # 6) Group contiguous points into segments
    df["segment_id"] = (df["initial_state"] != df["initial_state"].shift(1)).cumsum()

    # 7) Finalize segment classification & compute radius
    segment_types = {}
    segment_radii = {}

    for seg_id, seg_rows in df.groupby("segment_id"):
        if len(seg_rows) < min_segment_size:
            segment_types[seg_id] = "undefined"
            segment_radii[seg_id] = np.nan
            continue

        seg_state = seg_rows["initial_state"].iloc[0]
        if seg_state == "Übergangsbogen":
            segment_types[seg_id] = "Übergangsbogen"
            segment_radii[seg_id] = np.nan
        elif seg_state == "straight line":
            segment_types[seg_id] = "straight line"
            segment_radii[seg_id] = np.nan
        else:
            # "curve"
            segment_types[seg_id] = "curve"
            valid_radius = seg_rows[radius_col].replace(np.inf, np.nan)
            segment_radii[seg_id] = valid_radius.mean()

    df["segment_type"] = df["segment_id"].map(segment_types)
    df["mittelradius"] = df["segment_id"].map(segment_radii)

    # 8) Clean up
    df.drop(
        columns=["curvature_smoothed", "curvature_slope", "initial_state", "segment_id"],
        inplace=True
    )

    return df



import pwlf
def data_segment_train_curves5(df, config):
    """
    Fit piecewise linear segments to curvature vs. distance (cumulated_distance_m).
    Then label each piece as 'straight line', 'transition', or 'curve'.

    config expects:
      - 'curvature': column name for curvature (e.g. 'curvature_heading_dx_ds_grad')
      - 'distance_col': column name for distance (e.g. 'cumulated_distance_m')
      - 'num_segments': how many linear segments to fit (e.g. 5)
      - 'straight_curv_tol': if a piece's mean curvature < this => 'straight line'
      - 'slope_tol': slope threshold to define 'transition' vs. 'steady' (e.g. 0.0005)
    """
    curvature_col = 'curvature_heading_dx_ds_grad'
    distance_col = 'cumulative_distance_m'
    num_segments = config.get('num_segments', 5)
    straight_curv_tol = config.get('straight_curv_tol', 0.00005)
    slope_tol = config.get('slope_tol', 0.0005)

    print("Configuration:")
    print("  curvature_col:", curvature_col)
    print("  distance_col:", distance_col)
    print("  num_segments:", num_segments)
    print("  straight_curv_tol:", straight_curv_tol)
    print("  slope_tol:", slope_tol)

    # Extract arrays
    x = df[distance_col].values
    y = df[curvature_col].values

    mask = np.isfinite(x) & np.isfinite(y)

    # Filter the arrays
    x = x[mask]
    y = y[mask]

    print("\nData arrays (first 5 entries):")
    print("  x:", x[:5])
    print("  y:", y[:5])

    # 1) Fit piecewise linear
    my_pwlf = pwlf.PiecewiseLinFit(x, y)
    print("test")
    # Fit with 'num_segments' line segments
    res = my_pwlf.fit(num_segments)  # returns SSE
    print("\nPiecewise linear fit completed.")
    print("  Sum of Squared Errors (SSE):", res)

    # 2) The breakpoints in 'x' are:
    breakpoints = my_pwlf.fit_breaks  # e.g. array([x_min, ..., x_max])
    print("\nBreakpoints:")
    print(" ", breakpoints)

    # 3) For each segment, retrieve slope & intercept and classify the segment.
    segment_info = []
    slopes = my_pwlf.slopes
    intercepts = my_pwlf.intercepts

    print("\nSegment slopes and intercepts:")
    print("  Slopes:", slopes)
    print("  Intercepts:", intercepts)

    # breakpoints => [s0, s1, s2, ..., sN], so segment i is in [s_{i}, s_{i+1}]
    for i in range(len(slopes)):
        seg_start = breakpoints[i]
        seg_end   = breakpoints[i+1]
        seg_slope = slopes[i]
        seg_intercept = intercepts[i]

        # Compute midpoint for average curvature estimation.
        mid_s = 0.5 * (seg_start + seg_end)
        mid_curv = seg_slope * mid_s + seg_intercept

        # Classification logic:
        abs_slope = abs(seg_slope)
        abs_mid_curv = abs(mid_curv)

        if abs_slope < slope_tol:
            if abs_mid_curv < straight_curv_tol:
                seg_type = "straight line"
            else:
                seg_type = "curve"
        else:
            seg_type = "Übergangsbogen"  # transition

        print(f"\nSegment {i}:")
        print("  Start distance:", seg_start)
        print("  End distance:", seg_end)
        print("  Slope:", seg_slope)
        print("  Intercept:", seg_intercept)
        print("  Midpoint:", mid_s)
        print("  Mean curvature estimate:", mid_curv)
        print("  Classified as:", seg_type)

        segment_info.append({
            "segment_index": i,
            "start_distance": seg_start,
            "end_distance": seg_end,
            "slope": seg_slope,
            "intercept": seg_intercept,
            "mean_curvature_est": mid_curv,
            "segment_type": seg_type
        })

    result_df = pd.DataFrame(segment_info)
    print("\nResulting segments DataFrame:")
    print(result_df)
    return result_df


import ruptures as rpt

def data_segment_train_curves15(df, config):
    """
    Fügt dem DataFrame zwei neue Spalten hinzu:
      - 'segment_class': Klassifikation des Segments anhand des Krümmungsverhaltens ("gerade Linie", "Kurve" oder "Übergangsbogen")
      - 'mittelradius': Berechneter Mittelradius des Segments (1/|Mittelwert der Krümmung|; bei nahezu 0 wird np.inf gesetzt)

    Parameter in config:
      - 'curvature': Spaltenname für die Krümmung (z.B. 'curvature_heading_dx_ds_grad')
      - 'distance_col': Spaltenname für die Distanz (z.B. 'cumulative_distance_m')
      - 'straight_curv_tol': Schwellenwert, unter dem die durchschnittliche Krümmung als "gerade Linie" gilt
      - 'delta_tol': Mindestdifferenz der Krümmung (von Anfang zu Ende eines Segments), um ein Segment als Übergangsbogen zu kennzeichnen
      - 'penalty': Penalty-Parameter für ruptures (je höher, desto weniger Change Points)
    """
    # Parameter aus config
    curvature_col    = config.get('curvature_nope', 'curvature_heading_dx_ds_grad')
    distance_col     = config.get('distance_col', 'cumulative_distance_m')
    straight_curv_tol = config.get('straight_curv_tol', 0.00005)
    delta_tol        = config.get('delta_tol', 0.03 )
    penalty          = config.get('penalty', 0.001)

    # Kopie des DataFrames, um das Original zu erhalten
    df_new = df.copy()

    # Extrahiere die Arrays (als float)
    x = np.array(df_new[distance_col].values, dtype=float)
    y = np.array(df_new[curvature_col].values, dtype=float)

    # Ersetze unendliche Werte (Inf) mit 0, da diese als gerade Linie interpretiert werden sollen
    y = np.where(np.isinf(y), 999, y)

    # Change-Point-Detection mit ruptures (Pelt-Algorithmus, Modell "l2")
    algo = rpt.Pelt(model="l2").fit(y)
    change_points = algo.predict(pen=penalty)
    print("Gefundene Change Points (als Indizes):", change_points)

    # Neue Spalte für die Klassifikation initialisieren
    df_new['segment_class'] = 'unclassified'
    # Neue Spalte für den Mittelradius initialisieren
    df_new['mittelradius'] = 0.0

    start_idx = 0
    for cp in change_points:
        seg_indices = np.arange(start_idx, cp)
        seg_y = y[start_idx:cp]
        seg_x = x[start_idx:cp]
        if len(seg_y) < 2:
            segment_class = 'zu klein'
            mittelradius = np.nan
        else:
            mean_curv = np.mean(seg_y)
            # Änderung der Krümmung vom Anfang bis zum Ende des Segments
            delta_curv = abs(seg_y[-1] - seg_y[0])
            print(f"Segment {start_idx}-{cp}: mean_curv = {mean_curv:.6f}, delta_curv = {delta_curv:.6f}")

            if mean_curv < straight_curv_tol:
                segment_class = "gerade Linie"
            elif delta_curv > delta_tol:
                segment_class = "Übergangsbogen"
            else:
                segment_class = "Kurve"

            # Berechnung des Mittelradius: Bei nahezu 0 Krümmung (gerade Linie) als unendlich setzen,
            # sonst 1/|mean_curv|
            if abs(mean_curv) < 1e-10:
                mittelradius = float('inf')
            else:
                mittelradius = 1 / abs(mean_curv)

        print(f"Segment von Index {start_idx} bis {cp} klassifiziert als: {segment_class}, Mittelradius: {mittelradius}")
        df_new.loc[seg_indices, 'segment_class'] = segment_class
        df_new.loc[seg_indices, 'mittelradius'] = mittelradius
        start_idx = cp

    return df_new




from scipy.signal import savgol_filter

def data_segment_train_curves(df, config):
    """
    Classify track segments (straight, transition, steady curve) and compute
    mean radius per contiguous segment.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing at least two columns:
        - dist_col (default 'distance'): cumulative distance along the track
        - curvature_col (default 'curvature'): measured or computed curvature
    config : dict
        Configuration dictionary with keys:
        - 'curvature_col' (str): column name for curvature data
        - 'dist_col' (str): column name for distance along the track (must be monotonic)
        - 'eps' (float): threshold below which curvature is considered 'zero' (straight track)
        - 'deriv_thresh' (float): threshold on curvature derivative for deciding transition vs steady curve
        - 'window_length' (int): window size for Savitzky-Golay filter (must be odd, < len(df))
        - 'polyorder' (int): polynomial order for Savitzky-Golay filter (must be < window_length)

    Returns
    -------
    df : pandas.DataFrame
        Same dataframe with added columns:
        - 'curvature_smooth': Smoothed curvature
        - 'dcurv': Numerical derivative of curvature w.r.t. distance
        - 'straight_mask': Boolean indicating near-zero curvature
        - 'steady_mask': Boolean indicating small derivative of curvature
        - 'segment_type': One of {'straight', 'transition', 'steady_curve', 'unknown'}
        - 'instant_radius': 1 / curvature_smooth (∞ if curvature_smooth is zero)
        - 'segment_id': Numeric ID for each contiguous segment
        - 'mean_radius': Mean radius in that segment (∞ if segment is straight)
    """

    # Extract config or set defaults
    curvature_col = config.get('curvature_col', 'curvature_heading_dx_ds_grad')
    dist_col      = config.get('dist_col', 'cumulative_distance_m')
    eps           = config.get('eps', 1e-4)
    deriv_thresh  = config.get('deriv_thresh', 1e-6)
    window_length = config.get('window_length', 3)
    polyorder     = config.get('polyorder', 1)

    print("\n=== dat_segment_track DEBUG START ===")
    print("Function parameters:")
    print(f"  curvature_col = {curvature_col}")
    print(f"  dist_col      = {dist_col}")
    print(f"  eps           = {eps}")
    print(f"  deriv_thresh  = {deriv_thresh}")
    print(f"  window_length = {window_length}")
    print(f"  polyorder     = {polyorder}")

    print("\nDataFrame info at start:")
    print("  df.shape:", df.shape)
    print("  df.columns:", list(df.columns))
    print("  First few rows:\n", df.head(3))

    # 1) Smooth curvature data
    try:
        print("\n[Step 1] Smoothing curvature data with savgol_filter...")

        # 1a) Convert the curvature column to a float NumPy array
        #     (This avoids any object dtypes, multi-index confusion, etc.)
        col_data = df[curvature_col].astype(float).to_numpy(copy=True)

        # 1b) Replace inf, -inf, 0 in that array
        col_data[np.isinf(col_data)] = 1e9
        col_data[col_data == 0] = 1e-9

        # 1c) Apply Savitzky-Golay filter on the array
        smooth_arr = savgol_filter(col_data, window_length=window_length, polyorder=polyorder)

        # 1d) Store back in the DataFrame
        df['curvature_smooth'] = smooth_arr

        print("  => df['curvature_smooth'] created successfully.")
    except Exception as e:
        print("  !! ERROR in smoothing step:", e)
        raise

    # 2) Identify near-zero curvature = straight
    try:
        print("\n[Step 2] Identifying near-zero curvature...")
        df['straight_mask'] = np.abs(df['curvature_smooth']) < eps
        print("  => df['straight_mask'] created. Example values:")
        print(df['straight_mask'].head(3))
    except Exception as e:
        print("  !! ERROR in straight_mask step:", e)
        raise

    # 3) Compute derivative of curvature w.r.t. distance
    try:
        print("\n[Step 3] Computing derivative dcurv via np.gradient...")
        df['dcurv'] = np.gradient(df['curvature_smooth'], df[dist_col])
        print("  => df['dcurv'] created. Example values:")
        print(df['dcurv'].head(3))
    except Exception as e:
        print("  !! ERROR in dcurv step:", e)
        raise

    # 4) Among non-straight points, check derivative magnitude
    try:
        print("\n[Step 4] Creating steady_mask...")
        df['steady_mask'] = (np.abs(df['dcurv']) < deriv_thresh) & ~df['straight_mask']
        print("  => df['steady_mask'] created. Example values:")
        print(df['steady_mask'].head(3))
    except Exception as e:
        print("  !! ERROR in steady_mask step:", e)
        raise

    # 5) Assign segment_type using np.select
    try:
        print("\n[Step 5] Assigning segment_type with np.select...")
        conditions = [
            df['straight_mask'],
            (~df['straight_mask']) & (~df['steady_mask']),
            df['steady_mask']
        ]
        choices = ['straight', 'transition', 'steady_curve']
        df['segment_type'] = np.select(conditions, choices, default='unknown')
        print("  => df['segment_type'] created. Example distribution:")
        print(df['segment_type'].value_counts())
    except Exception as e:
        print("  !! ERROR in segment_type step:", e)
        raise

    # 6) Compute instantaneous radius
    try:
        print("\n[Step 6] Computing instantaneous radius...")
        df['instant_radius'] = np.where(
            np.isclose(df['curvature_smooth'], 0.0, atol=1e-12),
            np.inf,
            1.0 / df['curvature_smooth']
        )
        print("  => df['instant_radius'] created. Example values:")
        print(df['instant_radius'].head(3))
    except Exception as e:
        print("  !! ERROR in instant_radius step:", e)
        raise

    # 7) Define a segment_id that increments when 'segment_type' changes
    try:
        print("\n[Step 7] Defining segment_id by checking changes in segment_type...")
        df['segment_id'] = (df['segment_type'] != df['segment_type'].shift(1)).cumsum()
        print("  => df['segment_id'] created. Example values:")
        print(df[['segment_type','segment_id']].head(10))
    except Exception as e:
        print("  !! ERROR in segment_id step:", e)
        raise

    # 8) Compute mean radius per segment
    try:
        print("\n[Step 8] Grouping by segment_id to compute mean_radius...")
        mean_radius_per_segment = df.groupby('segment_id')['instant_radius'].mean()
        df['mean_radius'] = df['segment_id'].map(mean_radius_per_segment)
        print("  => df['mean_radius'] assigned via groupby. Example values:")
        print(df['mean_radius'].head(3))
    except Exception as e:
        print("  !! ERROR in mean_radius step:", e)
        raise

    # 9) Force mean_radius to ∞ for straight segments
    try:
        print("\n[Step 9] Setting mean_radius = ∞ for straight segments...")
        df.loc[df['segment_type'] == 'straight', 'mean_radius'] = np.inf
        print("  => done. Checking final distribution of mean_radius:")
        print(df['mean_radius'].head(10))
    except Exception as e:
        print("  !! ERROR in final step for mean_radius assignment:", e)
        raise

    print("\n=== dat_segment_track DEBUG END ===\n")
    return df
