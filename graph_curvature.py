import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_curvature_with_mittelradius(csv_path, lat_col, lon_col, curvature_limits=(-0.005, 0.005)):
    """
    Loads GPS data from a CSV file, plots curvature versus cumulative distance,
    and adds markers (with the mean radius, "mittelradius") at the middle of each steady curve segment.

    Assumptions:
      - The CSV contains a column "cumulative_distance_m".
      - The computed curvature is stored in "curvature_heading_dx_ds_grad".
      - The computed radius is stored in "radius_heading_dx_ds".
      - The DataFrame already contains segmentation columns:
            "segment_type": "straight line", "Übergangsbogen", or "curve"
            "mittelradius": mean radius (for curve segments; NaN otherwise)
    """
    # Load the CSV data
    df = pd.read_csv(csv_path)

    # Optionally, if segmentation hasn't been computed, you would call your segmentation function here.
    # For example:
    # config = {
    #     "curvature": "curvature_heading_dx_ds_grad",
    #     "radius_m": "radius_heading_dx_ds",
    #     "smoothing_window": 5,
    #     "straight_curvature_threshold": 0.001,
    #     "steady_std_threshold": 0.005,
    #     "min_segment_size": 10,
    # }
    # df = data_segment_train_curves(df, config)

    # Plot curvature vs. cumulative distance.
    fig, ax1 = plt.subplots(figsize=(220, 5))
    ax1.plot(df["cumulative_distance_m"], df["curvature_heading_dx_ds_grad"], label="Curvature", color="blue")

    ax1.set_xlabel("Distance from Origin (m)")
    ax1.set_ylabel("Curvature (1/m)")
    ax1.set_title("Curvature Along the Railway Path with Krümmungsband")
    ax1.set_ylim(curvature_limits)
    ax1.grid()

    # Group contiguous segments based on the "segment_type".
    # (We create a temporary grouping by detecting changes in the segment type.)
    df["seg_group"] = (df["segment_type"] != df["segment_type"].shift(1)).cumsum()

    # For each group that is a steady curve ("curve"), determine the midpoint and plot a marker.
    curve_groups = df[df["segment_type"] == "steady_curve"].groupby("seg_group")
    for group_id, group in curve_groups:
        # Use the median cumulative distance of the segment as the x-coordinate for the marker.
        mid_distance = group["cumulative_distance_m"].median()
        # Use the median curvature value for the y-coordinate.
        mid_curvature = group["curvature_heading_dx_ds_grad"].median()
        # Get the mean radius for the segment (should be the same for all points in a steady segment).
        mittelradius_val = group["mean_radius"].iloc[0]

        # Plot a red marker at the midpoint.
        ax1.plot(mid_distance, mid_curvature, marker='o', markersize=8, color='red', linestyle='None')
        # Annotate the marker with the mean radius (rounded to one decimal place).
        ax1.annotate(f"{mittelradius_val:.1f}", (mid_distance, mid_curvature),
                     textcoords="offset points", xytext=(0, 10), ha='center', color='red')

    ax1.legend(loc="upper left")
    plt.show()


