import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "subsets_by_date/2024-04-09/2024-04-09_savitzky_gaussian_planar_dist_time_heading_headingDS_yawRate_1percent_kalman.csv"  # Replace with your file path

# Read the CSV into a DataFrame, parsing the DatumZeit column as datetime
data = pd.read_csv(file_path, parse_dates=["DatumZeit"], dayfirst=True)

# Check if the required columns are in the DataFrame
required_columns = ["DatumZeit", "yaw_rate_kalman", "Gier"]
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"The CSV file must contain the following columns: {required_columns}")

# Plot the data
plt.figure(figsize=(120, 6))  # Increased the figure width for a wider graph

# Plot yaw_rate_deg_s
plt.plot(data["DatumZeit"], data["yaw_rate_kalman"], label="Yaw Rate (deg/s)", alpha=0.7)

# Plot Gier
plt.plot(data["DatumZeit"], data["Gier"], label="Gier", alpha=0.7)

# Add labels, title, and legend
plt.xlabel("Time")
plt.ylabel("Values")
plt.title("Yaw Rate and Gier Over Time")
plt.legend()

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45)

# Tight layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()
