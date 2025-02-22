import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any

# All configuration variables are defined here.
CONFIG: Dict[str, Any] = {
    "csv_file": "subsets_by_date/2024-04-02/2024-04-02_rollingW_planar_time_headingDS_yawRate_elevation_distance_gradient_smooothGradient.csv",
    # Path to your CSV file
    "elevation_column": "elevation",  # Column name for elevation data
    "gradient_promille_column": "gradient_promille",  # Column name for raw gradient (promille) data
    "smoothed_gradient_promille_column": "smoothed_gradient_promille",  # Column name for smoothed gradient (promille)
    "horizontal_distance_column": "cumulative_distance"  # Column name for cumulative distance data
}

def plot_elevation_and_gradient(df: pd.DataFrame, config: Dict[str, Any]) -> None:
    """
    Plot elevation, raw gradient (promille), and smoothed gradient (promille) on the same graph using dual y-axes.
    The gradient y-axis is adjusted dynamically based on the smoothed gradient values.

    Parameters:
        df (pd.DataFrame): DataFrame containing the necessary columns.
        config (Dict[str, Any]): Configuration dictionary with keys:
            - "elevation_column": Name of the elevation column.
            - "gradient_promille_column": Name of the raw gradient per mille column.
            - "smoothed_gradient_promille_column": Name of the smoothed gradient per mille column.
            - "horizontal_distance_column": Name of the cumulative distance column.
    """
    # Retrieve column names from the configuration
    elevation_col = config["elevation_column"]
    gradient_promille_col = config["gradient_promille_column"]
    smoothed_gradient_promille_col = config["smoothed_gradient_promille_column"]
    horizontal_distance_col = config["horizontal_distance_column"]

    # Create the figure and the first axis
    fig, ax1 = plt.subplots(figsize=(210, 6))

    # Plot the elevation on the left y-axis
    color_elev = 'tab:blue'
    ax1.set_xlabel('Cumulative Distance (m)')
    ax1.set_ylabel('Elevation (m)', color=color_elev)
    ax1.plot(df[horizontal_distance_col], df[elevation_col], color=color_elev, label='Elevation', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color_elev)

    # Create a second y-axis for the gradient (promille)
    ax2 = ax1.twinx()
    color_grad = 'tab:red'
    color_smooth_grad = 'tab:green'
    ax2.set_ylabel('Gradient (promille)', color=color_grad)

    # Plot raw gradient (dashed red)
    ax2.plot(df[horizontal_distance_col], df[gradient_promille_col], color=color_grad, label="Raw Gradient (promille)",
             linewidth=1.5, linestyle='--', alpha=0.6)

    # Plot smoothed gradient (solid green)
    ax2.plot(df[horizontal_distance_col], df[smoothed_gradient_promille_col], color=color_smooth_grad,
             label="Smoothed Gradient (promille)", linewidth=2)

    ax2.tick_params(axis='y', labelcolor=color_grad)

    # Adjust y-axis limits based on the smoothed gradient values
    min_smooth = df[smoothed_gradient_promille_col].min()
    max_smooth = df[smoothed_gradient_promille_col].max()
    range_smooth = max_smooth - min_smooth
    buffer = range_smooth * 0.1  # Add 10% buffer for better visualization
    ax2.set_ylim(min_smooth - buffer, max_smooth + buffer)

    # Title and layout improvements
    plt.title("Elevation, Raw Gradient & Smoothed Gradient vs. Cumulative Distance")
    fig.tight_layout()

    # Merged Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.show()

if __name__ == '__main__':
    # Read the CSV file specified in the configuration
    csv_file = CONFIG["csv_file"]
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        raise FileNotFoundError(f"Unable to read CSV file '{csv_file}': {e}")

    # Plot the elevation and gradient on the same graph
    plot_elevation_and_gradient(df, CONFIG)
