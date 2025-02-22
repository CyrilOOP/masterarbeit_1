import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read CSV
df = pd.read_csv('subsets_by_date/2024-04-02/2024-04-02_time_rollingW_planar_distance_headingDX_headingDS_yawRate.csv')

# We'll assume these columns all exist
x = df['x']
y = df['y']
heading_ds = df['heading_deg_ds']      # 0°=North, +clockwise
heading_dx = df['heading_dx_dy']       # 0°=North, +clockwise

# Convert to standard math angles (0°=East, +CCW)
theta_ds = np.deg2rad(90 - heading_ds)
theta_dx = np.deg2rad(90 - heading_dx)

arrow_length = 10.0

# Now get correct dx, dy for quiver
dx_ds = arrow_length * np.cos(theta_ds)
dy_ds = arrow_length * np.sin(theta_ds)

dx_dx = arrow_length * np.cos(theta_dx)
dy_dx = arrow_length * np.sin(theta_dx)

plt.figure(figsize=(100, 100))
plt.plot(x, y, 'b-', label='Train Path')

# Plot every Nth arrow
indices = np.arange(0, len(df), 20)

plt.quiver(x.iloc[indices], y.iloc[indices],
           dx_ds.iloc[indices], dy_ds.iloc[indices],
           color='b', scale=500, width=0.0001, label='Heading (dx/ds)')

plt.quiver(x.iloc[indices], y.iloc[indices],
           dx_dx.iloc[indices], dy_dx.iloc[indices],
           color='g', scale=500, width=0.0001, label='Heading (dx/dy)')

plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Train Path with Corrected Heading Display')
plt.legend()
plt.grid(True)
plt.show()
