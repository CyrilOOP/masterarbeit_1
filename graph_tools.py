import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def graph_yaw_rate_and_gier(processed_file: str):
    """
    Generates a plot of yaw rate and gier over time using data from the provided processed CSV file.

    This function assumes:
      - The processed CSV file (e.g. "2024-04-02_rollingW_planar_time_headingDS_yawRate_curvature_infra.csv")
        is located in a folder (e.g. "subsets_by_date/2024-04-02/").
      - A corresponding raw CSV file exists in the same folder and is named as "YYYY-MM-DD.csv"
        (e.g. "2024-04-02.csv").
      - The CSV files contain a column named 'DatumZeit' that stores datetime values in the format
        "%Y-%m-%d %H:%M:%S.%f".
      - The raw CSV contains columns "Gier" and "Gier_GPS" while the processed CSV contains the column
        "yaw_rate_deg_s".

    Parameters:
        processed_file (str): File path to the processed CSV file.
    """
    # Extract the directory and filename from the processed file path
    directory, filename = os.path.split(processed_file)

    # Extract the date from the filename using regex (expects a pattern like YYYY-MM-DD)
    match = re.search(r"\d{4}-\d{2}-\d{2}", filename)
    if match:
        date_str = match.group(0)
    else:
        date_str = "Unknown Date"

    # Construct the raw file path; assumes the raw file is named as "YYYY-MM-DD.csv" in the same directory
    raw_file = os.path.join(directory, f"{date_str}.csv")

    # Define the datetime format (adjust if needed)
    date_format = "%Y-%m-%d %H:%M:%S.%f"

    # Load the processed data
    try:
        data = pd.read_csv(processed_file)
    except Exception as e:
        print(f"Error loading processed data from {processed_file}: {e}")
        return

    # Load the raw data
    try:
        data_raw = pd.read_csv(raw_file)
    except Exception as e:
        print(f"Error loading raw data from {raw_file}: {e}")
        return

    # Convert the 'DatumZeit' column to datetime for both datasets
    if "DatumZeit" in data.columns:
        data["DatumZeit"] = pd.to_datetime(data["DatumZeit"], format=date_format)
    else:
        print("Processed data does not contain the 'DatumZeit' column.")
        return

    if "DatumZeit" in data_raw.columns:
        data_raw["DatumZeit"] = pd.to_datetime(data_raw["DatumZeit"], format=date_format)
    else:
        print("Raw data does not contain the 'DatumZeit' column.")
        return

    # Create the figure and axis for plotting
    fig, ax = plt.subplots(figsize=(220, 6))

    # Plot "Gier" from the raw data if available
    if "Gier" in data_raw.columns:
        ax.plot(data_raw["DatumZeit"], data_raw["Gier"], label="Gier from raw", alpha=0.7)
    else:
        print("Raw data does not contain 'Gier' column.")

    # Plot "Gier_GPS" (e.g. from Ariane) from the raw data if available
    if "Gier_GPS" in data_raw.columns:
        ax.plot(data_raw["DatumZeit"], data_raw["Gier_GPS"], label="Gier from Ariane", alpha=0.7, color="red")
    else:
        print("Raw data does not contain 'Gier_GPS' column.")

    # Plot the yaw rate (deg/s) from the processed data if available
    if "yaw_rate_deg_s" in data.columns:
        ax.plot(data["DatumZeit"], data["yaw_rate_deg_s"], label="Yaw Rate (deg/s)", alpha=0.7)
    else:
        print("Processed data does not contain 'yaw_rate_deg_s' column.")

    # Add labels, title, and legend to the plot
    ax.set_xlabel("Time")
    ax.set_ylabel("Values")
    ax.set_title(f"Yaw Rate and Gier Over Time ({date_str})")
    ax.legend()

    # Set major and minor ticks for the x-axis for better granularity
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    # Add grid lines
    ax.grid(True, which="major", linestyle="--", linewidth=1, alpha=0.8)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.5)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Display the plot
    plt.show()