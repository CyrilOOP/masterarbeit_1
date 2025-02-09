import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_curvature(csv_path, lat_col, lon_col, curvature_limits=(-0.2, 0.2), curvature_threshold=1e-5):
    """Load GPS data, compute cumulative distance from precomputed segment distances, and plot curvature."""

    df = pd.read_csv(csv_path)

    # Ensure the necessary distance column exists
    if 'distance_spline_segment' not in df.columns:
        raise ValueError("CSV file must contain 'distance_spline_segment' column with segment-wise distances.")

    # Compute cumulative distance using precomputed segment distances
    df["distance_from_origin"] = df["distance_spline_segment"].cumsum()

    # Ensure required curvature columns exist
    if 'curvature_yaw_distance' not in df.columns or 'curvature_yaw_rate' not in df.columns:
        raise ValueError("CSV file must contain 'curvature_yaw_distance' and 'curvature_yaw_rate' columns.")

    # Debug: Print curvature values
    print(df[['curvature_yaw_distance', 'curvature_yaw_rate']].head(10))

    # Compute radius of curvature (1/curvature), preserving correct behavior
    df['radius_yaw_distance'] = np.where(
        np.abs(df['curvature_yaw_distance']) > curvature_threshold,
        1 / df['curvature_yaw_distance'],
        np.nan
    )
    df['radius_yaw_rate'] = np.where(
        np.abs(df['curvature_yaw_rate']) > curvature_threshold,
        1 / df['curvature_yaw_rate'],
        np.nan
    )

    # Plot
    fig, ax1 = plt.subplots(figsize=(20, 5))  # Adjusted size

    # Plot curvature
    ax1.plot(df["distance_from_origin"], df["curvature_yaw_distance"], label="Curvature (yaw distance)", color="blue")
    ax1.plot(df["distance_from_origin"], df["curvature_yaw_rate"], label="Curvature (yaw rate)", color="red", linestyle='dashed')

    # Add Krümmungsband (Radius Labels)
    for i in range(0, len(df), max(1, len(df) // 20)):  # Reduce label clutter
        if not np.isnan(df['radius_yaw_distance'][i]):
            ax1.text(df['distance_from_origin'][i], df['curvature_yaw_distance'][i],
                     f"{df['radius_yaw_distance'][i]:.0f}m", fontsize=9,
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))

    ax1.set_xlabel("Distance from Origin (m)")
    ax1.set_ylabel("Curvature (1/m)")
    ax1.set_title("Curvature Along the Railway Path with Krümmungsband")

    ax1.set_ylim(curvature_limits)
    ax1.legend(loc="upper left")
    ax1.grid()
    plt.show()

# Example usage:
csv_path = "subsets_by_date/2024-04-02/2024-04-02_rollingW_planar_time_headingDS_yawRate_curvature.csv"
plot_curvature(csv_path, lat_col="GPS_lat_smoothed_rolling_windows", lon_col="GPS_lon_smoothed_rolling_windows")
