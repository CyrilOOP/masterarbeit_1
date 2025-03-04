import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tkinter as tk
from tkinter import ttk
from cycler import cycler
from matplotlib.collections import LineCollection
from matplotlib.dates import date2num
from matplotlib.lines import Line2D


def select_optional_signals_gui(optional_signals):
    """
    Opens a Tkinter window with a checkbox for each optional signal.
    Returns a list of selected signals (each is a tuple as defined in the optional_signals list).
    """
    # Create the root window and hide it
    root = tk.Tk()
    root.withdraw()

    # Create a new top-level window for selection
    window = tk.Toplevel(root)
    window.title("Select Optional Signals to Plot")
    window.geometry("300x200")

    # Bring window to the front
    window.lift()
    window.attributes('-topmost', True)
    window.after_idle(window.attributes, '-topmost', False)

    # Dictionary to hold the BooleanVar for each signal (keyed by the signal label)
    check_vars = {}

    # Create a frame for checkboxes
    frame = ttk.Frame(window, padding="10")
    frame.pack(fill="both", expand=True)

    # Add a label at the top
    label = ttk.Label(frame, text="Select signals to plot:")
    label.pack(anchor="w", pady=(0, 5))

    # Create a checkbox for each optional signal (default: selected)
    for sig in optional_signals:
        var = tk.BooleanVar(value=True)
        check_vars[sig[2]] = var  # sig[2] is the label text
        cb = ttk.Checkbutton(frame, text=sig[2], variable=var)
        cb.pack(anchor="w")

    selected_signals = []

    def on_ok():
        for sig in optional_signals:
            if check_vars[sig[2]].get():
                selected_signals.append(sig)
        window.destroy()

    # Add OK button at the bottom
    ok_button = ttk.Button(frame, text="OK", command=on_ok)
    ok_button.pack(pady=10)

    # Wait for the window to be closed
    window.wait_window()
    root.destroy()
    return selected_signals

def graph_yaw_rate_and_gier(processed_file: str, config):
    """
    Generates a plot where the base signal (Gier) and additional yaw rate signals are shown.

    The function examines the available optional signals and then uses a GUI for selection.
    The base signal "Gier" is always plotted (and its range sets the y‑axis), and then the
    selected optional signals are plotted on top.

    Signals:
      - Gier (base, raw yaw rate sensor) [always plotted]
      - Gier_GPS (yaw rate calculated by Ariane) [if available]
      - Any columns starting with "yaw_rate_from_heading" [if available]

    Parameters:
      processed_file (str): File path to the processed CSV file.
    """
    # Extract directory and filename from the processed file path
    directory, filename = os.path.split(processed_file)
    match = re.search(r"\d{4}-\d{2}-\d{2}", filename)
    date_str = match.group(0) if match else "Unknown Date"

    # Construct the raw CSV file path (assumed to be in the same directory)
    raw_file = os.path.join(directory, f"{date_str}.csv")

    # Define the datetime format
    date_format = "%Y-%m-%d %H:%M:%S.%f"

    # Load processed data
    try:
        data_proc = pd.read_csv(processed_file)
    except Exception as e:
        print(f"Error loading processed file: {e}")
        return

    # Load raw data
    try:
        data_raw = pd.read_csv(raw_file)
    except Exception as e:
        print(f"Error loading raw file: {e}")
        return

    # Convert 'DatumZeit' columns to datetime
    if "DatumZeit" in data_proc.columns:
        data_proc["DatumZeit"] = pd.to_datetime(data_proc["DatumZeit"], format=date_format)
    else:
        print("Processed file missing 'DatumZeit' column")
        return

    if "DatumZeit" in data_raw.columns:
        data_raw["DatumZeit"] = pd.to_datetime(data_raw["DatumZeit"], format=date_format)
    else:
        print("Raw file missing 'DatumZeit' column")
        return

    # Determine the overall Gier range from the raw data
    gier_series = []
    if "Gier" in data_raw.columns:
        gier_series.append(data_raw["Gier"])
    else:
        print("Raw file missing 'Gier' column")
    if "Gier_GPS" in data_raw.columns:
        gier_series.append(data_raw["Gier_GPS"])
    else:
        print("Raw file missing 'Gier_GPS' column")

    if not gier_series:
        print("No Gier data available")
        return

    gier_combined = pd.concat(gier_series)
    gier_min = gier_combined.min()
    gier_max = gier_combined.max()

    # Create the plot on a single axis
    fig, ax = plt.subplots(figsize=(100, 6), dpi=300)  # Adjust figsize as needed

    # Plot base Gier data (the main reference signal)
    if "Gier" in data_raw.columns:
        ax.plot(data_raw["DatumZeit"], data_raw["Gier"], label="Gier (raw)", alpha=1, color="black", linewidth=2)
    else:
        print("Gier column missing in raw data.")
        return

    # Gather optional signals to plot
    optional_signals = []
    # Option: Gier_GPS from raw data
    if "Gier_GPS" in data_raw.columns:
        optional_signals.append((
            "Gier_GPS",               # key
            data_raw["Gier_GPS"],     # series
            "Gier Rate from Ariane",  # label for legend
            "red",                    # default color (will be overridden by cycler)
            0.7                       # alpha (transparency)
        ))
    # Any columns starting with "yaw_rate_from_heading"
    heading_cols = [col for col in data_proc.columns if col.startswith("yaw_rate_from_heading")]
    for col in heading_cols:
        optional_signals.append((
            col,
            data_proc[col],
            col,
            None,  # No default color provided
            0.7
        ))

    # Use the GUI to select which optional signals to plot
    if optional_signals:
        selected_signals = select_optional_signals_gui(optional_signals)
    else:
        selected_signals = []
        print("No optional signals found to plot.")

    # Setup cycler for color only
    num_optional = len(selected_signals)
    if num_optional > 0:
        colors = plt.cm.viridis(np.linspace(0, 1, num_optional))
    else:
        colors = ['blue']

    # Apply only color cycling, remove linestyle cycling
    ax.set_prop_cycle(cycler('color', colors))

    # Plot each selected optional signal with specified thickness
    for key, series, label, _, alpha in selected_signals:
        time_series = data_proc["DatumZeit"] if key.startswith("yaw_rate") else data_raw["DatumZeit"]
        ax.plot(time_series, series, label=label, alpha=alpha, linewidth=2)  # Adjust linewidth as needed

    # Set the y-axis limits explicitly to the Gier range (base signal)
    ax.set_ylim(gier_min, gier_max)

    # Format the plot
    ax.set_xlabel("Time")
    ax.set_ylabel("Yaw Rate / Gier Values")
    ax.set_title(f"Yaw Rate and Gier Over Time ({date_str})")
    ax.legend(loc="upper right")

    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    ax.grid(True, which="major", linestyle="--", linewidth=1, alpha=0.8)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.5)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_curvature_with_mittelradius(processed_file, config):
    """
    Loads GPS data from a CSV file and creates a plot with two x-axes.
      - Bottom x-axis: values from the column specified by config["x_col"] (default "cumulative_distance_m").
      - Top x-axis: time values from the column specified by config["time_col"] (default "DatumZeit").

    The y-axis is determined by automatically searching for any column containing
    'curvature' in its name. If more than one candidate is found, the user is prompted
    to choose one.

    The curvature line is drawn as a continuous line whose color changes according to
    its attribute in the segment_type column.

    Markers (annotated with the corresponding "mittelradius" value) are plotted once per
    contiguous segment in the column specified by config["marker_col"].

    Parameters:
        processed_file (str): Full path to the CSV file.
        config (dict): Configuration dictionary with possible keys:
            "x_col", "time_col", "marker_col", "segment_type_col",
            "curvature_limits", "figsize", "title", etc.
    """



    # Split the path to obtain directory and filename.
    directory, filename = os.path.split(processed_file)
    print(f"Directory: {directory}\nFilename: {filename}")

    # Load the CSV data.
    df = pd.read_csv(processed_file)

    # Determine column names (with defaults from config).
    x_col = config.get("x_col", "cumulative_distance_m")
    time_col = config.get("time_col", "DatumZeit")
    marker_col = config.get("marker_col", "mean_radius")
    seg_type_col = config.get("segment_type_col_________", "segment_type")

    # Search for any column with 'curvature' (case-insensitive) for the y-axis.
    y_candidates = [col for col in df.columns if "curvature" in col.lower()]
    if len(y_candidates) == 0:
        raise ValueError("No column containing 'curvature' found in the data.")
    elif len(y_candidates) == 1:
        y_col = y_candidates[0]
        print(f"Using curvature column: {y_col}")
    else:
        print("Multiple curvature columns found:")
        for i, col in enumerate(y_candidates):
            print(f"  {i}: {col}")
        try:
            index = int(input("Enter the index of the column to use: "))
            y_col = y_candidates[index]
        except Exception:
            raise ValueError("Invalid selection for curvature column.")

    # Convert the time column to datetime.
    df[time_col] = pd.to_datetime(df[time_col])

    # Set up the figure.
    figsize = config.get("figsize", (220, 6))  # Adjusted default size (inches)
    curvature_limits = config.get("curvature_limits", (-0.005, 0.005))
    fig, ax1 = plt.subplots(figsize=figsize, dpi=300)

    # Map segment types to numeric values for coloring.
    segment_types = df[seg_type_col].unique()
    seg_to_num = {seg: i for i, seg in enumerate(segment_types)}
    # Create a categorical colormap. We'll use tab10.
    cmap = plt.cm.get_cmap("viridis", len(segment_types))

    # Prepare data for LineCollection.
    x = df[x_col].values
    y = df[y_col].values
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # For each segment, use the seg_type of the left endpoint.
    seg_numeric = df[seg_type_col].map(seg_to_num).values[:-1]

    # Create the LineCollection.
    norm = plt.Normalize(0, len(segment_types) - 1)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(seg_numeric)
    lc.set_linewidth(2)
    ax1.add_collection(lc)

    # Set axes limits and labels.
    ax1.set_xlim(x.min(), x.max())
    ax1.set_ylim(curvature_limits)
    ax1.set_xlabel(x_col)
    ax1.set_ylabel(y_col)
    ax1.set_title(config.get("title", "Curvature Plot with Mittelradius Markers"))
    ax1.grid(True)

    # Create a secondary x-axis on the top for time.
    ax2 = ax1.twiny()

    # Map cumulative distance to time.
    x_vals = x
    time_nums = df[time_col].map(date2num).values

    def x_to_time(x_val):
        return np.interp(x_val, x_vals, time_nums)

    bottom_ticks = ax1.get_xticks()
    ax2.set_xticks(bottom_ticks)
    tick_labels = [mdates.num2date(x_to_time(x_val)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                   for x_val in bottom_ticks]
    ax2.set_xticklabels(tick_labels, rotation=45, ha='left')
    ax2.set_xlabel(time_col)

    # Plot markers for each contiguous segment with constant marker value,
    # only for segments of type "steady_curve" and if the marker value is within the allowed range.
    df["marker_group"] = (df[marker_col].shift() != df[marker_col]).cumsum()
    for group_id, group in df.groupby("marker_group"):
        # Only add marker if the segment type for this group is "steady_curve".
        if group[seg_type_col].iloc[0] != "steady_curve":
            continue
        marker_val = group[marker_col].iloc[0]
        # Check that the marker value is not NaN and its absolute value is between 400 and 20000.
        if pd.notna(marker_val) and (400 <= abs(marker_val) <= 30000):
            mid_x = group[x_col].median()
            mid_y = group[y_col].median()
            ax1.plot(mid_x, mid_y, marker='o', markersize=8, color='black', linestyle='None')
            ax1.annotate(f"{marker_val:.1f}", (mid_x, mid_y),
                         textcoords="offset points", xytext=(0, 10),
                         ha='center', color='black')


    # Create a custom legend for the segment types.
    legend_handles = [
        Line2D([0], [0],
               color=cmap(seg_to_num[seg] / (len(segment_types) - 1) if len(segment_types) > 1 else 0),
               lw=2, label=str(seg))
        for seg in segment_types
    ]
    ax1.legend(handles=legend_handles, loc="upper left")

    plt.tight_layout()
    plt.show()
