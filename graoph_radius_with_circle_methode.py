#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt


def main():
    # --- Configuration Variables ---
    # Set the path to your CSV file.
    csv_file = "subsets_by_date/2024-04-02/2024-04-02_time_rollingW_planar_distance_headingDX_headingDS_yawRate_radius_circle.csv"  # Replace with the path to your CSV file.

    # Column names in your CSV.
    x_col = "x"  # Column name for x coordinate.
    y_col = "y"  # Column name for y coordinate.
    radius_col = "radius_m_circle"  # Column name for the computed circle radius.

    # --- Read the CSV ---
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error readi"
              f"ng CSV file '{csv_file}': {e}")
        return

    # --- Check that required columns exist ---
    for col in [x_col, y_col, radius_col]:
        if col not in df.columns:
            print(f"Error: Column '{col}' not found in CSV file.")
            return

    # --- Plot the Path and Annotate with Circle Radius ---
    plt.figure(figsize=(210, 70))
    plt.plot(df[x_col], df[y_col], marker="o", linestyle="-", label="Path")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Path with Circle Radius Annotations")

    # Annotate points with their circle radius (if not NaN)
    for idx, row in df.iterrows():
        radius = row[radius_col]
        if pd.notna(radius):
            plt.annotate(f"{radius:.1f}",
                         (row[x_col], row[y_col]),
                         textcoords="offset points", xytext=(5, 5),
                         fontsize=8, color="red")

    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
