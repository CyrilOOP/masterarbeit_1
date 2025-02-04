import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "subsets_by_date/2024-05-17/2024-05-17_rollingW_planar_time_headingDS_yawRate_delBoundaries.csv"  # Replace with your file path
file_path_raw = "subsets_by_date/2024-05-17/2024-05-17.csv"
# Read the CSV into a DataFrame, parsing the DatumZeit column as datetime


date_format = "%Y-%m-%d %H:%M:%S.%f"

# Read the CSV files without converting dates automatically.
data = pd.read_csv(file_path)
data_raw = pd.read_csv(file_path_raw)

# Now convert the 'DatumZeit' column using the specified format.
data["DatumZeit"] = pd.to_datetime(data["DatumZeit"], format=date_format)
data_raw["DatumZeit"] = pd.to_datetime(data_raw["DatumZeit"], format=date_format)





'''
# Check if the required columns are in the DataFrame
required_columns = ["DatumZeit", "yaw_rate_deg_s", "Gier"]
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"The CSV file must contain the following columns: {required_columns}")
'''
# Plot the data
plt.figure(figsize=(120, 6))  # Increased the figure width for a wider graph



# Plot Gier
plt.plot(data_raw["DatumZeit"], data_raw["Gier"], label="Gier from raw", alpha=0.7)

# Plot yaw_rate_deg_s
plt.plot(data["DatumZeit"], data["yaw_rate_deg_s"], label="Yaw Rate (deg/s)", alpha=0.7)


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
