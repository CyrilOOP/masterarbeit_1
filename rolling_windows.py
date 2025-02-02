import pandas as pd

from csv_tools import csv_select_gps_columns


def rolling_windows_gps_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Add rolling-window smoothed GPS columns to a DataFrame using a speed-dependent time window.
    If the train is 'stopped' (speed < threshold), accumulate points until movement resumes.
    Otherwise, use a piecewise time window size based on speed.

    The original number of rows in `df` is retained, but two new columns:
       - GPS_lat_smoothed_rolling_windows
       - GPS_lon_smoothed_rolling_windows
    are added. All rows in the same window share the same averaged lat/lon.

    Parameters
    ----------
    df : pd.DataFrame
        Must include columns: ['timestamp', 'latitude', 'longitude', 'speed']
        (or whichever columns you select with your `csv_select_gps_columns` function).
    config : dict
        Contains all your user-configured parameters, e.g.:
        {
            "speed_threshold_stopped_rolling_windows": 0.5,
            "time_window_slow_rolling_windows": 5.0,
            "slow_speed_threshold_rolling_windows": 5.0,
            "time_rolling_window_mid": 2.0,
            "mid_speed_threshold_rolling_windows": 20.0,
            "time_rolling_window_fast": 1.0,
            "time_between_points": <Series or column with timestamps>,
            "speed_column": <Series or column with speed>,
            ...
        }

    Returns
    -------
    df : pd.DataFrame
        The same DataFrame, **with two extra columns**:
          - "GPS_lat_smoothed_rolling_windows"
          - "GPS_lon_smoothed_rolling_windows"
        populated by the rolling window logic.
    """

    # ----------------------------------------------------------------------------
    # 1. SELECT COLUMNS (e.g., via your csv_select_gps_columns function)
    # ----------------------------------------------------------------------------
    gps_lat_col, gps_lon_col = csv_select_gps_columns(
        df,
        title="Select GPS Data for rolling windows",
        prompt="Select the GPS data to use as input for rolling windows:"
    )
    print(f"Using GPS columns: {gps_lat_col} and {gps_lon_col}")

    # ----------------------------------------------------------------------------
    # 2. EXTRACT CONFIG AND GET THE RELEVANT SERIES OR ARRAYS
    # ----------------------------------------------------------------------------
    speed_threshold_stopped = config.get("speed_threshold_stopped_rolling_windows")
    time_window_slow = config.get("time_window_slow_rolling_windows")
    slow_speed_threshold = config.get("slow_speed_threshold_rolling_windows")
    time_window_mid = config.get("time_rolling_window_mid")
    mid_speed_threshold = config.get("mid_speed_threshold_rolling_windows")
    time_window_fast = config.get("time_rolling_window_fast")

    # The user might store the columns (timestamp, speed, etc.) in config under keys
    ts_series = config.get("time_between_points")  # your chosen time/timestamp column
    spd_series = config.get("speed_column")  # your chosen speed column

    # ----------------------------------------------------------------------------
    # 3. SORT DF BY TIMESTAMP (just to be sure) AND CONVERT TO NUMPY
    # ----------------------------------------------------------------------------
    df = df.sort_values(by='timestamp', ignore_index=True)

    ts_array = ts_series.to_numpy(dtype=float)  # times
    lat_array = gps_lat_col.to_numpy(dtype=float)  # latitudes
    lon_array = gps_lon_col.to_numpy(dtype=float)  # longitudes
    spd_array = spd_series.to_numpy(dtype=float)  # speeds

    n = len(df)

    # Prepare arrays to store the smoothed results for each row
    lat_smooth = np.empty(n, dtype=float)
    lon_smooth = np.empty(n, dtype=float)

    # ----------------------------------------------------------------------------
    # 4. TIME-WINDOW LOGIC
    # ----------------------------------------------------------------------------
    def get_time_window(speed_value):
        """Return a time window (in seconds) based on speed, or None if 'stopped'."""
        if speed_value < speed_threshold_stopped:
            return None  # Special case => treat as 'stopped'
        elif speed_value < slow_speed_threshold:
            return time_window_slow
        elif speed_value < mid_speed_threshold:
            return time_window_mid
        else:
            return time_window_fast

    # ----------------------------------------------------------------------------
    # 5. MAIN LOOP: FOR EACH "WINDOW," ASSIGN SMOOTHED LAT/LON TO ROWS
    # ----------------------------------------------------------------------------
    i = 0
    while i < n:
        current_speed = spd_array[i]
        wlen = get_time_window(current_speed)

        if wlen is not None:
            # Train is moving => gather points in [ts_array[i], ts_array[i] + wlen]
            window_end = ts_array[i] + wlen

            sum_lat = 0.0
            sum_lon = 0.0
            count = 0

            j = i
            while j < n and ts_array[j] <= window_end:
                sum_lat += lat_array[j]
                sum_lon += lon_array[j]
                count += 1
                j += 1

            # Compute the window's average
            mean_lat = sum_lat / count
            mean_lon = sum_lon / count

            # Assign these averages to all rows in [i..j-1]
            lat_smooth[i:j] = mean_lat
            lon_smooth[i:j] = mean_lon

            i = j  # jump past these rows

        else:
            # Train is stopped => accumulate until speed >= threshold
            sum_lat = 0.0
            sum_lon = 0.0
            count = 0

            j = i
            while j < n and spd_array[j] < speed_threshold_stopped:
                sum_lat += lat_array[j]
                sum_lon += lon_array[j]
                count += 1
                j += 1

            mean_lat = sum_lat / count
            mean_lon = sum_lon / count

            lat_smooth[i:j] = mean_lat
            lon_smooth[i:j] = mean_lon

            i = j

    # ----------------------------------------------------------------------------
    # 6. SAVE THE SMOOTHED COLUMNS INTO THE DF
    # ----------------------------------------------------------------------------
    df['GPS_lat_smoothed_rolling_windows'] = lat_smooth
    df['GPS_lon_smoothed_rolling_windows'] = lon_smooth

    # ----------------------------------------------------------------------------
    # 7. RETURN THE UPDATED DF
    # ----------------------------------------------------------------------------
    return df
