import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import re

# Load the CSV file
file_path = "subsets_by_date/2024-04-02/2024-04-02_rollingW_planar_time_heading_headingDS_yawRate.csv"  # Replace with your file path
file_path_raw = "subsets_by_date/2024-04-02/2024-04-02.csv"

# Extract the date from the file path using regex
match = re.search(r"\d{4}-\d{2}-\d{2}", file_path)
if match:
    date_str = match.group(0)  # Extracted date in YYYY-MM-DD format
else:
    date_str = "Unknown Date"

# Read the CSV into a DataFrame, parsing the DatumZeit column as datetime
date_format = "%Y-%m-%d %H:%M:%S.%f"

# Read the CSV files without converting dates automatically
data = pd.read_csv(file_path)
data_raw = pd.read_csv(file_path_raw)

# Convert the 'DatumZeit' column using the specified format
data["DatumZeit"] = pd.to_datetime(data["DatumZeit"], format=date_format)
data_raw["DatumZeit"] = pd.to_datetime(data_raw["DatumZeit"], format=date_format)

# Create the figure and axis
fig, ax = plt.subplots(figsize=(120, 6))  # Adjust figure size

# Plot Gier
ax.plot(data_raw["DatumZeit"], data_raw["Gier"], label="Gier from raw", alpha=0.7)

# Plot yaw_rate_deg_s
ax.plot(data["DatumZeit"], data["yaw_rate_deg_s"], label="Yaw Rate (deg/s)", alpha=0.7)

# Add labels, title, and legend
ax.set_xlabel("Time")
ax.set_ylabel("Values")
ax.set_title(f"Yaw Rate and Gier Over Time ({date_str})")  # Add the extracted date to the title
ax.legend()

# Set major and minor ticks for better granularity
ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))  # Major ticks every 15 minutes
ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=5))  # Minor ticks every 5 minutes
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))  # Format time as HH:MM

# Add grid with major and minor ticks
ax.grid(True, which="major", linestyle="--", linewidth=1, alpha=0.8)  # Major grid
ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.5)  # Minor grid (finer)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Tight layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()
