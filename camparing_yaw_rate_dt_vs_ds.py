import pandas as pd
import matplotlib.pyplot as plt

# Set your CSV file path here
csv_file = "subsets_by_date/2024-04-02/2024-04-02_rollingW_planar_headingDT_distance_headingDS_yawRate.csv"  # <-- Change this to your CSV file path

# Load the CSV file
df = pd.read_csv(csv_file)

# Convert 'DatumZeit' to datetime
df["DatumZeit"] = pd.to_datetime(df["DatumZeit"])

# Define yaw rate column names
dt_col = "yaw_rate_from_heading_deg_dt"
ds_col = "yaw_rate_from_heading_deg_ds"

# Check that required columns exist
if dt_col not in df.columns:
    print(f"Error: Column '{dt_col}' not found in the CSV file.")
    exit(1)
if ds_col not in df.columns:
    print(f"Error: Column '{ds_col}' not found in the CSV file.")
    exit(1)

# Plot both yaw rate columns against time
plt.figure(figsize=(210, 6))
plt.plot(df["DatumZeit"], df[dt_col], label=dt_col, marker="o", linestyle="-")
plt.plot(df["DatumZeit"], df[ds_col], label=ds_col, marker="s", linestyle="--")
plt.xlabel("Time (DatumZeit)")
plt.ylabel("Yaw Rate (deg/s)")
plt.title("Yaw Rate vs Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
