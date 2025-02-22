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
from scipy.interpolate import splprep, splev

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
    df["x"], df["y"] = transformer.transform(df[lon_input].values, df[lat_input].values)

    # Add selected method
    df["selected_smoothing_method"] = selected_method

    # Add projection column
    df["projection_epsg"] = epsg_code

    print("[FUNCTION] data_convert_to_planar: END\n")

    return df


def data_parse_time(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Parse the given datetime column as pandas datetime and compute:
      - 'elapsed_time_s': Time in seconds relative to the first timestamp.

    Args:
        df: Input DataFrame.
        config: Configuration dictionary containing:
            - "datetime_col": Name of the column containing datetime information
                              (defaults to "DatumZeit" if not provided).

    Returns:
        A copy of the DataFrame with the new column 'elapsed_time_s'.

    Raises:
        ValueError: If the datetime column cannot be parsed.
    """
    print("\n[FUNCTION] data_parse_time: START")
    datetime_col = config.get("datetime_col", "DatumZeit")
    df = df.copy()

    # Convert the column to datetime using explicit format (including milliseconds)
    try:
        df[datetime_col] = pd.to_datetime(df[datetime_col], format="%Y-%m-%d %H:%M:%S.%f")
    except Exception as e:
        raise ValueError(f"Error parsing datetime column '{datetime_col}': {e}")

    # Create a column with time in seconds relative to the first timestamp
    df["elapsed_time_s"] = (df[datetime_col] - df[datetime_col].iloc[0]).dt.total_seconds()

    print("  > datetime_col =", datetime_col)
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

    x_col = config.get("x_col", "x")
    y_col = config.get("y_col", "y")
    heading_col = config.get("heading_col_dx_dy", "heading_dx_dy")

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

    x_col = config.get("x_col", "x")
    y_col = config.get("y_col", "y")
    cum_dist_col = config.get("cum_dist_col", "cumulative_distance")
    heading_col = config.get("heading_col_ds", "heading_deg_ds")

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


def select_heading_columns_gui(heading_candidates):
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
    root.title("Select Heading Columns")
    root.geometry("400x300")
    root.minsize(300, 200)

    root.columnconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)

    label = ttk.Label(root, text="Select heading column(s):")
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

    if "elapsed_time_s" not in df.columns:
        raise ValueError("Column 'elapsed_time_s' is missing from DataFrame.")

    time_delta_s = df["elapsed_time_s"].diff()
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
        selected_heading = select_heading_columns_gui(heading_candidates)
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

    date_col = config.get("date_column", "DatumZeit")
    yaw_rate_col = config.get("yaw_rate_column", None)
    input_lower_bound = config["delete_lower_bound_percentage"] / 100.0
    input_upper_bound = config["delete_upper_bound_percentage"] / 100.0

    print(f"  > Lower quantile bound: {input_lower_bound} (fraction)")
    print(f"  > Upper quantile bound: {input_upper_bound} (fraction)")

    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' is missing from the CSV file.")

    if not yaw_rate_col or yaw_rate_col not in df.columns:
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

    date_col = config.get("date_column", "DatumZeit")
    speed_col = config["speed_column"]
    speed_threshold_stopped = config["speed_threshold_stopped_rolling_windows"]
    distance_window_meters = config["distance_window_meters"]
    time_window_min = config["time_window_min"]
    time_window_max = config["time_window_max"]
    max_stop_window = config["max_stop_window"]
    speed_bins = config["speed_bins"]

    # Convert date_col to datetime
    df[date_col] = pd.to_datetime(df[date_col], format="%Y-%m-%d %H:%M:%S.%f")
    t_arr = df[date_col].astype(np.int64) / 1e9

    lat_col_rol_win, lon_col_rol_win = csv_select_gps_columns(
        df,
        title="Select GPS Data for Rolling Windows",
        prompt="Select the GPS data to use as input for rolling windows:"
    )
    print(f"  > Using lat_col_rol_win='{lat_col_rol_win}', lon_col_rol_win='{lon_col_rol_win}'")

    lat_arr = df[lat_col_rol_win].to_numpy(dtype=float)
    lon_arr = df[lon_col_rol_win].to_numpy(dtype=float)

    # Rolling (smoothed) speed
    df['speed_filtered'] = df[speed_col].rolling(window=5, center=True, min_periods=1).mean()
    spd_arr = df['speed_filtered'].to_numpy(dtype=float)
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

    hysteresis_window = 10
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
            row_dict["time_numeric"] = midpoint_time
            row_dict[speed_col] = mean_spd
            row_dict["GPS_lat_smoothed_rolling_windows"] = mean_lat
            row_dict["GPS_lon_smoothed_rolling_windows"] = mean_lon
            row_dict["segment_marker"] = "MOVING"
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

    required_keys = ["api_key", "elevation_column", "api_url", "batch_size", "threads"]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(f"Missing configuration keys: {missing_keys}")

    api_key = config["api_key"]
    elevation_column = config["elevation_column"]
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
    Adds 'cumulative_distance' column.

    If possible, also attempt a spline-based approach, but only if enough unique points exist.
    """
    print("\n[FUNCTION] data_compute_traveled_distance: START")

    lat_col, lon_col = csv_select_gps_columns(
        df,
        title="Select GPS Data for Distances",
        prompt="Select the GPS data to use as input to calculate the distance traveled:"
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

    df["cumulative_distance"] = np.cumsum(geodesic_distances)
    print(f"  > Added 'cumulative_distance' column. Sample: {df['cumulative_distance'].head()}")

    print("[FUNCTION] data_compute_traveled_distance: END\n")
    return df


def data_compute_gradient(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute the gradient (slope) and gradient in per mille between consecutive points using elevation
    and horizontal distance. Adds columns 'gradient' and 'gradient_promille' by default.
    """
    print("\n[FUNCTION] data_compute_gradient: START")
    elevation_col = config.get("elevation_column", "elevation")
    horizontal_distance_col = config.get("horizontal_distance_column", "cumulative_distance")
    gradient_col = config.get("gradient_column", "gradient")
    gradient_promille_col = config.get("gradient_promille_column", "gradient_promille")

    for col in [elevation_col, horizontal_distance_col]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame.")

    gradients = [0.0]
    gradients_promille = [0.0]

    for i in range(1, len(df)):
        elev_diff = df.iloc[i][elevation_col] - df.iloc[i - 1][elevation_col]
        horiz_diff = df.iloc[i][horizontal_distance_col] - df.iloc[i - 1][horizontal_distance_col]
        grad = elev_diff / horiz_diff if horiz_diff != 0 else 0.0
        gradients.append(grad)
        gradients_promille.append(grad * 1000)

    df[gradient_col] = gradients
    df[gradient_promille_col] = gradients_promille

    print(f"  > Added columns: '{gradient_col}', '{gradient_promille_col}'")
    print("[FUNCTION] data_compute_gradient: END\n")
    return df

def data_compute_curvature_radius_and_detect_steady_curves(
    df: pd.DataFrame, config: dict = None
) -> pd.DataFrame:
    """
    1) Compute signed curvature & radius (with sign).
    2) Apply threshold for "straight line" if radius > e.g. 11000.
    3) Segment the data using a variable-window approach for 'steady curves'.
    4) Optionally compute mean radius/curvature for each segment.
    """
    print("\n[FUNCTION] data_compute_curvature_radius_and_detect_steady_curves: START")
    import numpy as np

    if config is None:
        config = {}

    def compute_radius_and_curvature_with_threshold(df_inner: pd.DataFrame, config: dict) -> None:
        print("[DEBUG] Starting compute_radius_and_curvature_with_threshold")
        distance_col = config.get("distance_col", "cumulative_distance")
        heading_col = config.get("heading_col", "heading_deg_ds")
        radius_col = config.get("radius_col", "radius_m")
        curvature_col = config.get("curvature_col", "curvature")
        straight_threshold = config.get("straight_threshold", 11000.0)

        # Compute heading differences in degrees
        heading_diff_deg = df_inner[heading_col].diff()
        heading_diff_deg = (heading_diff_deg + 180) % 360 - 180
        heading_diff_deg = heading_diff_deg.fillna(0)
        print("[DEBUG] Heading differences (degrees) - first 5 values:")
        print(heading_diff_deg.head())

        # Convert differences to radians
        heading_diff_rad = heading_diff_deg * (np.pi / 180.0)
        print("[DEBUG] Heading differences (radians) - first 5 values:")
        print(heading_diff_rad.head())

        heading_diff_rad_no_zero = heading_diff_rad.replace(0, np.nan)

        # Compute radius and curvature
        radius = df_inner[distance_col] / heading_diff_rad_no_zero
        radius = radius.fillna(np.inf)
        curvature = heading_diff_rad_no_zero / df_inner[distance_col]
        curvature = curvature.fillna(0)
        print("[DEBUG] Computed raw radius (first 5 values):")
        print(radius.head())
        print("[DEBUG] Computed raw curvature (first 5 values):")
        print(curvature.head())

        # Apply threshold for "straight line"
        too_large = radius.abs() > straight_threshold
        count_too_large = too_large.sum()
        if count_too_large > 0:
            print(f"[DEBUG] {count_too_large} points exceed threshold {straight_threshold}. Setting radius to inf and curvature to 0.")
        radius[too_large] = np.inf
        curvature[too_large] = 0.0

        df_inner[radius_col] = radius
        df_inner[curvature_col] = curvature
        print("[DEBUG] Finished compute_radius_and_curvature_with_threshold.\n")

    def detect_steady_curves_variable_window(df_inner: pd.DataFrame, config: dict) -> None:
        print("[DEBUG] Starting detect_steady_curves_variable_window")
        curvature_col = config.get("curvature_col", "curvature")
        std_threshold = config.get("curvature_std_thresh", 0.0001)
        min_segment_size = config.get("min_segment_size", 3)

        c = df_inner[curvature_col].to_numpy()
        n = len(c)

        steady_group_ids = np.zeros(n, dtype=int)
        group_id = 0
        start_idx = 0

        for i in range(n):
            seg_curv = c[start_idx : i + 1]
            curv_std = np.nanstd(seg_curv)
            print(f"[DEBUG] i={i}, start_idx={start_idx}, current segment std={curv_std:.6f}")

            if curv_std <= std_threshold:
                # Keep extending the segment
                continue
            else:
                seg_length = i - start_idx
                if seg_length >= min_segment_size:
                    group_id += 1
                    steady_group_ids[start_idx : i] = group_id
                    print(f"[DEBUG] Steady segment detected from index {start_idx} to {i-1} (length {seg_length}) with std {curv_std:.6f}; assigned group {group_id}")
                else:
                    print(f"[DEBUG] Segment from index {start_idx} to {i-1} too short (length {seg_length}); no group assigned.")
                start_idx = i

        # Final segment handling
        seg_length = n - start_idx
        if seg_length >= min_segment_size:
            group_id += 1
            steady_group_ids[start_idx : n] = group_id
            print(f"[DEBUG] Final steady segment from index {start_idx} to {n-1} (length {seg_length}) assigned group {group_id}")
        else:
            print(f"[DEBUG] Final segment from index {start_idx} to {n-1} too short (length {seg_length}); no group assigned.")

        df_inner["steady_group"] = steady_group_ids
        print(f"[DEBUG] Total steady segments detected: {group_id}\n")

    def compute_segment_means(df_inner: pd.DataFrame, config: dict) -> None:
        print("[DEBUG] Starting compute_segment_means")
        radius_col = config.get("radius_col", "radius_m")
        curvature_col = config.get("curvature_col", "curvature")

        df_inner["steady_mean_radius"] = df_inner.groupby("steady_group")[radius_col].transform("mean")
        df_inner["steady_mean_curvature"] = df_inner.groupby("steady_group")[curvature_col].transform("mean")

        # Print group statistics
        grouped = df_inner.groupby("steady_group").agg(
            count=("steady_group", "count"),
            mean_radius=(radius_col, "mean"),
            mean_curvature=(curvature_col, "mean")
        )
        print("[DEBUG] Segment means computed for each steady group:")
        print(grouped)
        print("[DEBUG] Finished compute_segment_means.\n")

    print(f"[DEBUG] Input dataframe shape: {df.shape}")
    compute_radius_and_curvature_with_threshold(df, config)
    detect_steady_curves_variable_window(df, config)
    compute_segment_means(df, config)

    print("[FUNCTION] data_compute_curvature_radius_and_detect_steady_curves: END\n")
    return df
