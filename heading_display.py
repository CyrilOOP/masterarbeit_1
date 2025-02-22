import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Specify the CSV file name (update this with your actual file name)
csv_file = 'subsets_by_date/2024-04-02/2024-04-02_time_rollingW_planar_distance_headingDX_headingDS.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file)

# Check that the required columns are present
required_cols = ['x', 'y', 'heading_deg_ds', 'heading_dx_dy']
if not all(col in df.columns for col in required_cols):
    raise ValueError(f"CSV file must contain the columns: {required_cols}")

# Extract the x, y, and both heading columns
x = df['x']
y = df['y']
heading_ds = df['heading_deg_ds']  # Heading from dx/ds calculation
heading_dx = df['heading_dx_dy']  # Heading from dy/dx calculation

# Convert headings from degrees to radians
heading_ds_rad = np.deg2rad(heading_ds)
heading_dx_rad = np.deg2rad(heading_dx)

# Compute the directional vector components for the arrows.
# Using sin for the x-component and cos for the y-component aligns 0° with north.
arrow_length = 1.0
dx_ds = arrow_length * np.sin(heading_ds_rad)
dy_ds = arrow_length * np.cos(heading_ds_rad)

dx_dx = arrow_length * np.sin(heading_dx_rad)
dy_dx = arrow_length * np.cos(heading_dx_rad)

# Plot the train path
plt.figure(figsize=(110, 60))
plt.plot(x, y, 'b-', label='Train Path')  # Plot the train path

# Select every 50th row to display arrows for clarity
indices = np.arange(0, len(df), 50)

# Plot arrows for heading from dx/ds (in red)
plt.quiver(x.iloc[indices], y.iloc[indices],
           dx_ds.iloc[indices], dy_ds.iloc[indices],
           color='r', scale=50, width=0.005, label='Heading (dx/ds)')

# Plot arrows for heading from dy/dx (in green)
plt.quiver(x.iloc[indices], y.iloc[indices],
           dx_dx.iloc[indices], dy_dx.iloc[indices],
           color='g', scale=50, width=0.001, label='Heading (dy/dx)')

plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Train Path with Dual Heading Display (Every 50 Rows)')
plt.legend()
plt.grid(True)
plt.show()
