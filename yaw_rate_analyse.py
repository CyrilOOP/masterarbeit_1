import pandas as pd

def process_csv(csv_file, output_file):
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Check if required columns exist
    if 'yaw_rate_deg_s' not in df.columns or 'DatumZeit' not in df.columns:
        raise ValueError("The required columns 'yaw_rate_deg_s' or 'DatumZeit' are missing from the CSV file.")

    # Calculate 1% and 99% quantiles for the 'yaw_rate_deg_s' column
    lower_bound = df['yaw_rate_deg_s'].quantile(0.01)
    upper_bound = df['yaw_rate_deg_s'].quantile(0.99)

    # Filter rows within the quantile range
    filtered_df = df[(df['yaw_rate_deg_s'] >= lower_bound) & (df['yaw_rate_deg_s'] <= upper_bound)]

    # Save the filtered data to a new CSV file
    filtered_df.to_csv(output_file, index=False)

    print(f"Filtered CSV saved as {output_file}.")

if __name__ == "__main__":
    # Replace 'input_file.csv' with the path to your CSV file
    input_file = 'subsets_by_date/2024-05-17/2024-05-17_savitzky_gaussian_planar_dist_time_heading_yaw_rate_noNA.csv'
    output_file = 'subsets_by_date/2024-05-17/test.csv'

    try:
        process_csv(input_file, output_file)
    except Exception as e:
        print(f"Error: {e}")
