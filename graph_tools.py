import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tkinter as tk
from tkinter import ttk


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
        # Optional: print a debug message to confirm the button was clicked
        # print("OK button clicked")
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


def graph_yaw_rate_and_gier(processed_file: str):
    """
    Generates a plot where the base signal (Gier) and additional yaw rate signals are shown.

    The function examines the available optional signals and then uses a GUI for selection.
    The base signal "Gier" is always plotted (and its range sets the y‑axis), and then the
    selected optional signals are plotted on top.

    Signals:
      - Gier (base, raw yaw rate sensor) [always plotted]
      - Gier_GPS (yaw rate calculated by Ariane) [if available]
      - Any columns starting with "yaw_rate_from_heading_dx_dy" [if available]
      - The column "yaw_rate_from_heading_deg_ds" [if available]

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
    fig, ax = plt.subplots(figsize=(220, 6))  # Adjust figsize as needed

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
            "Gier_GPS",  # key
            data_raw["Gier_GPS"],  # series
            "Gier GPS (raw)",  # label for legend
            "red",  # color
            0.7  # alpha (transparency)
        ))
    # Options: yaw_rate_from_heading_dx_dy from processed data
    heading_dx_dy_cols = [col for col in data_proc.columns if col.startswith("yaw_rate_from_heading_dx_dy")]
    for col in heading_dx_dy_cols:
        optional_signals.append((
            col,
            data_proc[col],
            col,
            None,  # Let matplotlib choose the color
            0.7
        ))
    # Option: yaw_rate_from_heading_deg_ds from processed data
    if "yaw_rate_from_heading_deg_ds" in data_proc.columns:
        optional_signals.append((
            "yaw_rate_from_heading_deg_ds",
            data_proc["yaw_rate_from_heading_deg_ds"],
            "yaw_rate_from_heading_deg_ds",
            "blue",
            0.7
        ))

    # Use the GUI to select which optional signals to plot
    if optional_signals:
        selected_signals = select_optional_signals_gui(optional_signals)
    else:
        selected_signals = []
        print("No optional signals found to plot.")

    # Plot each selected optional signal
    for key, series, label, color, alpha in selected_signals:
        # Decide which time axis to use: processed signals use DatumZeit from data_proc,
        # while raw signals (like Gier_GPS) use DatumZeit from data_raw.
        time_series = data_proc["DatumZeit"] if key.startswith("yaw_rate") else data_raw["DatumZeit"]
        ax.plot(time_series, series, label=label, alpha=alpha, color=color)

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
