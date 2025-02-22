import pandas as pd
import os
from typing import List, Optional, Dict, Any, Tuple
from tkinter import Tk, filedialog, messagebox, ttk
import tkinter as tk

def csv_load(file_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.
    If no file_path is provided, open a file dialog to allow the user to browse for a file.

    Args:
        file_path: Path to the CSV file. If None, a file dialog will be shown.
        config: Optional configuration dictionary for additional settings (e.g., encoding).

    Returns:
        A pandas DataFrame containing the CSV data.

    Raises:
        FileNotFoundError: If no file is selected or the provided path does not exist.
        ValueError: If there is an error reading the CSV file.
    """
    if file_path is None:
        root = Tk()
        root.withdraw()  # Hide the root Tkinter window
        file_path = filedialog.askopenfilename(
            title="Select a CSV File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not file_path:  # No file selected
            raise FileNotFoundError("No file selected. Operation cancelled.")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    try:
        # Use encoding from config if provided
        encoding = config.get("encoding", "utf-8") if config else "utf-8"
        df = pd.read_csv(file_path, encoding=encoding)
        return df.copy()
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")


def csv_save(df: pd.DataFrame, file_path: str, config: Optional[Dict[str, Any]] = None) -> None:
    """
    Save a pandas DataFrame to a CSV file.

    Args:
        df: The DataFrame to save.
        file_path: The complete file path, including suffixes and extension.
        config: Optional configuration dictionary for additional settings (e.g., ensure_folder, run_stats).

    Raises:
        ValueError: If there is an error saving the CSV file.
    """
    if config is None:
        config = {}

    # Ensure folder exists if specified in config
    if config.get("ensure_folder", False):
        parent_dir = os.path.dirname(file_path)
        os.makedirs(parent_dir, exist_ok=True)
        print(f"Created directory: {parent_dir}")

    try:
        df.to_csv(file_path, index=False)
        print(f"File saved to: {file_path}")
    except Exception as e:
        raise ValueError(f"Error saving CSV file: {e}")

    # Generate statistics if specified in config
    if config.get("enable_statistics_on_save", False):
        csv_get_statistics(file_path, config)


def csv_get_files_in_subfolders(folder_path: str, config: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Recursively searches for files with the specified extension in all subfolders.

    Args:
        folder_path: The root folder to start searching.
        config: Optional configuration dictionary for additional settings (e.g., file_extension).

    Returns:
        A list of file paths relative to the root folder.
    """
    file_extension = config.get("file_extension", ".csv") if config else ".csv"
    file_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(file_extension):
                file_paths.append(os.path.relpath(os.path.join(root, file), folder_path))
    return file_paths


def csv_drop_na(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Remove rows with NA values in the specified columns.
    If no columns are specified, remove rows with NA in any column.

    Args:
        df: The DataFrame to process.
        config: Optional configuration dictionary for additional settings (e.g., columns to check).

    Returns:
        A DataFrame with rows containing NA values removed.
    """
    columns = config.get("columns", []) if config else []
    if columns:
        return df.dropna(subset=columns)
    return df.dropna()


def csv_group_by_date_and_save(df: pd.DataFrame, output_folder: str, config: Optional[Dict[str, Any]] = None) -> None:
    """
    Groups the DataFrame by the date part of a datetime column, creates a subfolder for each date,
    and saves the corresponding data into a CSV file inside that subfolder.

    Args:
        df: pandas DataFrame containing the data.
        output_folder: The folder where the output subfolders and CSV files will be stored.
        config: Optional configuration dictionary for additional settings (e.g., column_name).

    Raises:
        ValueError: If there is an error grouping or saving the data.
    """
    if config is None:
        config = {}

    column_name = config.get("column_name", "DatumZeit")
    try:
        df[column_name] = pd.to_datetime(df[column_name])
        grouped = df.groupby(df[column_name].dt.date)

        os.makedirs(output_folder, exist_ok=True)

        for date, group in grouped:
            date_str = date.strftime('%Y-%m-%d')
            date_folder_path = os.path.join(output_folder, date_str)
            os.makedirs(date_folder_path, exist_ok=True)

            group_file_path = os.path.join(date_folder_path, f"{date_str}.csv")

            if os.path.exists(group_file_path):
                os.remove(group_file_path)
                print(f"Existing file '{group_file_path}' deleted.")

            group.to_csv(group_file_path, index=False)
            print(f"Saved data for {date_str} to {group_file_path}")
    except Exception as e:
        raise ValueError(f"Error grouping and saving data: {e}")


def csv_get_statistics(file_path: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generate and save enhanced statistics for a CSV file, including missing, zero value analysis,
    and specific datetime analysis, to a text file. Also extracts and returns selected_smoothing_method
    and min_distance from the DataFrame.

    Args:
        file_path: Path to the CSV file.
        config: Optional configuration dictionary for additional settings (e.g., encoding).

    Returns:
        A dictionary containing:
            - selected_smoothing_method: The smoothing method extracted from the DataFrame.
            - min_distance: The minimum distance value extracted from the DataFrame.

    Raises:
        ValueError: If there is an error generating or saving statistics.
    """
    import os
    import numpy as np
    import pandas as pd
    from typing import Optional, Dict, Any

    if config is None:
        config = {}

    encoding = config.get("encoding", "utf-8")
    try:
        df = pd.read_csv(file_path, encoding=encoding)
    except Exception as e:
        raise ValueError(f"Error loading file {file_path}: {e}")

    # Extract selected_smoothing_method and min_distance from the DataFrame
    if "selected_smoothing_method" in df.columns:
        selected_smoothing_method = df["selected_smoothing_method"].iloc[0]  # Assuming it's the same for all rows
    else:
        selected_smoothing_method = "default_smoothing_method"  # Default value if column not found

    if "min_distance" in df.columns:
        min_distance = df["min_distance"].iloc[0]  # Assuming it's the same for all rows
    else:
        min_distance = 0  # Default value if column not found

    stats_report = [
        f"=== CSV File Statistics ===\n",
        f"File: {file_path}\n",
        f"Number of Rows: {df.shape[0]}\n",
        f"Number of Columns: {df.shape[1]}\n\n",
        "=== Column Data Types ===\n",
        df.dtypes.to_string() + "\n\n",
        "=== Missing and Zero Value Analysis ===\n",
    ]

    missing_values = df.isnull().sum()
    zero_values = (df == 0).sum(numeric_only=True)
    combined_stats = pd.DataFrame({"Missing Values": missing_values, "Zero Values": zero_values})
    stats_report.append(combined_stats.to_string() + "\n\n")

    stats_report.append("=== Numerical Column Statistics ===\n")
    # Replace infinities with NaN in numerical columns before generating statistics
    numeric_df = df.select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan)
    if not numeric_df.empty:
        stats_report.append(numeric_df.describe().to_string() + "\n\n")
    else:
        stats_report.append("No numerical columns found.\n\n")

    if 'DatumZeit' in df.columns:
        stats_report.append("=== DatumZeit Column Analysis ===\n")
        try:
            df['DatumZeit'] = pd.to_datetime(df['DatumZeit'], errors='coerce')
            if df['DatumZeit'].isnull().all():
                stats_report.append("Failed to parse DatumZeit column as datetime.\n\n")
            else:
                stats_report.append(f"Total non-null datetime entries: {df['DatumZeit'].notnull().sum()}\n")
                stats_report.append(f"Earliest timestamp: {df['DatumZeit'].min()}\n")
                stats_report.append(f"Latest timestamp: {df['DatumZeit'].max()}\n")
                stats_report.append("Entries per day:\n")
                stats_report.append(df['DatumZeit'].dt.date.value_counts().to_string() + "\n\n")
        except Exception as e:
            stats_report.append(f"Error processing DatumZeit column: {e}\n\n")
    else:
        stats_report.append("DatumZeit column not found.\n\n")

    # Add selected_smoothing_method and min_distance to the statistics report
    stats_report.append("=== Additional Information ===\n")
    stats_report.append(f"Selected Smoothing Method: {selected_smoothing_method}\n")
    stats_report.append(f"Min Distance: {min_distance}\n\n")

    output_file = f"{os.path.splitext(file_path)[0]}_statistics.txt"
    try:
        with open(output_file, "w", encoding=encoding) as f:
            f.write("".join(stats_report))
        print(f"Statistics saved to {output_file}")
    except Exception as e:
        raise ValueError(f"Error writing statistics to file: {e}")

    # Return selected_smoothing_method and min_distance
    return {
        "selected_smoothing_method": selected_smoothing_method,
        "min_distance": min_distance
    }


def subsets_by_date(folder_path: str, config: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Generate a list of file paths grouped by date from a folder containing CSV files.

    Args:
        folder_path: The folder containing CSV files.
        config: Optional configuration dictionary for additional settings (e.g., file_extension).

    Returns:
        A list of file paths grouped by date.
    """
    file_extension = config.get("file_extension", ".csv") if config else ".csv"
    file_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(file_extension):
                file_paths.append(os.path.join(root, file))
    return file_paths


# ============================================================================
# Helper: Select GPS Columns (raw or preprocessed) via GUI
# ============================================================================
def csv_select_gps_columns(df: pd.DataFrame,
                       title: str = "Select GPS Data",
                       prompt: str = "Select the GPS data to use (latitude and longitude):"
                       ) -> Tuple[str, str]:
    """
    Search the DataFrame for candidate GPS column pairs and return the selected pair.
    It considers:
      - Raw data: 'GPS_lat' and 'GPS_lon'
      - Preprocessed data: any pair with names matching
          'GPS_lat_smoothed_<method>' and 'GPS_lon_smoothed_<method>'

    If more than one candidate exists, a simple GUI dialog is shown.

    Returns:
        A tuple (lat_column, lon_column) chosen by the user.

    Raises:
        KeyError: If no valid candidate pair is found.
    """
    # Build a dictionary of candidate pairs.
    # Key: display label, Value: (lat_column, lon_column)
    candidates: Dict[str, Tuple[str, str]] = {}

    # Check for raw data.
    if "GPS_lat" in df.columns and "GPS_lon" in df.columns:
        candidates["raw (GPS_lat, GPS_lon)"] = ("GPS_lat", "GPS_lon")

    # Check for preprocessed columns.
    for col in df.columns:
        prefix = "GPS_lat_smoothed_"
        if col.startswith(prefix):
            method = col[len(prefix):]
            lon_candidate = f"GPS_lon_smoothed_{method}"
            if lon_candidate in df.columns:
                candidates[f"preprocessed ({method})"] = (col, lon_candidate)

    if not candidates:
        raise KeyError("No valid GPS data found. Expected raw 'GPS_lat'/'GPS_lon' or preprocessed "
                       "versions with pattern 'GPS_lat_smoothed_<method>' and 'GPS_lon_smoothed_<method>'.")

    # If only one candidate is available, return it immediately.
    options: List[str] = list(candidates.keys())
    if len(options) == 1:
        return candidates[options[0]]

    # If multiple candidates, present a GUI dialog for selection.
    chosen_label = csv_choose_from_options(title, prompt, options)
    return candidates[chosen_label]


def csv_choose_from_options(title: str, prompt: str, options: List[str]) -> str:
    """
    A simple GUI helper function that displays a combobox of options
    and returns the selected option.
    """
    selected_value = {"value": None}

    def on_ok():
        selected = combobox.get()
        if selected not in options:
            messagebox.showerror("Invalid Selection", "Please select a valid option.")
            return
        selected_value["value"] = selected
        dialog.destroy()

    dialog = tk.Tk()
    dialog.title(title)
    dialog.resizable(False, False)

    # Center the window on the screen.
    dialog.update_idletasks()
    width = 350
    height = 150
    x = (dialog.winfo_screenwidth() // 2) - (width // 2)
    y = (dialog.winfo_screenheight() // 2) - (height // 2)
    dialog.geometry(f"{width}x{height}+{x}+{y}")

    # Prompt label.
    label = tk.Label(dialog, text=prompt)
    label.pack(pady=(20, 10))

    # Combobox for options.
    combobox = ttk.Combobox(dialog, values=options, state="readonly", width=30)
    combobox.pack(pady=5)
    combobox.current(0)  # default selection

    # OK button.
    ok_button = tk.Button(dialog, text="OK", command=on_ok)
    ok_button.pack(pady=(10, 20))

    dialog.mainloop()

    if selected_value["value"] is None:
        raise ValueError("No selection was made.")
    return selected_value["value"]
