import sys
import os
from typing import Dict, List, Tuple, Any
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QListWidget, QLineEdit, QPushButton, QDoubleSpinBox, QMessageBox, QProgressBar, QFileDialog, QCheckBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon

# Import your existing modules
from csv_tools import (
    csv_load, csv_save, csv_group_by_date_and_save, csv_get_statistics,
    csv_get_files_in_subfolders, subsets_by_date
)
from data_tools import (
    data_convert_to_planar, data_compute_heading_from_xy, parse_time_and_compute_dt,
    data_filter_points_by_distance, data_compute_yaw_rate_from_heading,
    data_smooth_gps_savitzky, data_smooth_gps_gaussian
)
from map_generator import generate_map_from_csv


class DataProcessingApp(QMainWindow):
    def __init__(self, default_config: Dict[str, bool], subset_folder: str, pre_selected_date: str = None):
        super().__init__()
        self.default_config = default_config
        self.subset_folder = subset_folder
        self.pre_selected_date = pre_selected_date
        self.selected_steps = {}
        self.selected_subsets = []
        self.min_distance = 1.0
        self.input_file_path = None  # Store the selected file path

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Data Processing Toolkit")
        self.setGeometry(100, 100, 800, 600)
        self.setWindowIcon(QIcon("icon.png"))

        # Apply dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2E3440;
            }
            QLabel {
                color: #D8DEE9;
                font-size: 14px;
            }
            QListWidget {
                background-color: #3B4252;
                color: #D8DEE9;
                border: 1px solid #4C566A;
                border-radius: 5px;
                padding: 5px;
            }
            QLineEdit {
                background-color: #3B4252;
                color: #D8DEE9;
                border: 1px solid #4C566A;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton {
                background-color: #5E81AC;
                color: #ECEFF4;
                border: none;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #81A1C1;
            }
            QPushButton:pressed {
                background-color: #4C566A;
            }
            QDoubleSpinBox {
                background-color: #3B4252;
                color: #D8DEE9;
                border: 1px solid #4C566A;
                border-radius: 5px;
                padding: 5px;
            }
            QProgressBar {
                background-color: #3B4252;
                color: #D8DEE9;
                border: 1px solid #4C566A;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #5E81AC;
                border-radius: 5px;
            }
            QCheckBox {
                color: #D8DEE9;
                font-size: 14px;
            }
        """)

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # --- First Function: Create Subsets by Date ---
        first_function_group = QWidget()
        first_function_layout = QVBoxLayout(first_function_group)
        first_function_layout.addWidget(QLabel("Create Subsets by Date", font=QFont("Arial", 12, QFont.Bold)))

        # Button to execute the first function
        self.first_function_button = QPushButton("Select File and Create Subsets")
        self.first_function_button.setFont(QFont("Arial", 12))
        self.first_function_button.clicked.connect(self.execute_first_function)
        first_function_layout.addWidget(self.first_function_button)

        # Refresh button
        self.refresh_button = QPushButton("Refresh Subsets")
        self.refresh_button.setFont(QFont("Arial", 12))
        self.refresh_button.clicked.connect(self.refresh_subsets)
        first_function_layout.addWidget(self.refresh_button)

        main_layout.addWidget(first_function_group)

        # --- Steps to Run (Checkboxes) ---
        steps_group = QWidget()
        steps_layout = QVBoxLayout(steps_group)
        steps_layout.addWidget(QLabel("Processing Steps", font=QFont("Arial", 12, QFont.Bold)))

        self.checkboxes = {}
        for step_name, enabled_by_default in self.default_config.items():
            if step_name == "create_subsets_by_date":
                continue  # Skip the first function (handled separately)

            checkbox = QCheckBox(step_name.replace("_", " ").title())
            checkbox.setChecked(enabled_by_default)
            self.checkboxes[step_name] = checkbox
            steps_layout.addWidget(checkbox)

            # Connect the "statistics" checkbox to the deselect_other_functions method
            if step_name == "statistics":
                checkbox.stateChanged.connect(self.deselect_other_functions)

        main_layout.addWidget(steps_group)

        # --- Subsets to Process (List with Search) ---
        subsets_group = QWidget()
        subsets_layout = QVBoxLayout(subsets_group)
        subsets_layout.addWidget(QLabel("Subsets to Process", font=QFont("Arial", 12, QFont.Bold)))

        # Search bar
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search subsets...")
        self.search_bar.textChanged.connect(self.filter_subsets)
        subsets_layout.addWidget(self.search_bar)

        # List of subsets
        self.subset_list = QListWidget()
        self.subset_list.setSelectionMode(QListWidget.MultiSelection)
        subsets_layout.addWidget(self.subset_list)

        # Load subset files using the subsets_by_date function
        self.refresh_subsets()

        main_layout.addWidget(subsets_group)

        # --- Minimum Distance Input ---
        distance_group = QWidget()
        distance_layout = QHBoxLayout(distance_group)
        distance_layout.addWidget(QLabel("Minimum Distance (meters):"))

        self.distance_input = QDoubleSpinBox()
        self.distance_input.setRange(0.1, 1000.0)
        self.distance_input.setValue(1.0)
        self.distance_input.setSingleStep(0.1)
        self.distance_input.setToolTip("Set the minimum distance for filtering points")
        distance_layout.addWidget(self.distance_input)

        main_layout.addWidget(distance_group)

        # --- Progress Bar ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # --- Start Processing Button ---
        self.start_button = QPushButton("Start Processing")
        self.start_button.setFont(QFont("Arial", 14, QFont.Bold))
        self.start_button.setIcon(QIcon("start_icon.png"))
        self.start_button.setToolTip("Start processing the selected subsets")
        self.start_button.clicked.connect(self.on_submit)
        main_layout.addWidget(self.start_button)

    def deselect_other_functions(self, state: int):
        """
        Deselect all other checkboxes when the "statistics" checkbox is checked.

        Args:
            state: The state of the "statistics" checkbox (2 for checked, 0 for unchecked).
        """
        if state == 2:  # Checked
            for step_name, checkbox in self.checkboxes.items():
                if step_name != "statistics":
                    checkbox.setChecked(False)

    def refresh_subsets(self):
        """Refresh the list of subsets."""
        self.subset_files = subsets_by_date(self.subset_folder)
        self.file_paths = []  # Keeps the original relative paths
        self.subset_list.clear()
        for relative_path in self.subset_files:
            # Ensure the relative path does not include the subset_folder prefix
            relative_path = os.path.relpath(relative_path, self.subset_folder)
            file_name = os.path.basename(relative_path)
            self.file_paths.append(relative_path)
            self.subset_list.addItem(file_name)
            if self.pre_selected_date and file_name.strip() == self.pre_selected_date.strip():
                self.subset_list.findItems(file_name, Qt.MatchExactly)[0].setSelected(True)

    def execute_first_function(self):
        """Execute the first function: Create subsets by date."""
        # Prompt the user to select a file
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Input File", "", "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            self.input_file_path = file_path
            print(f"Selected file: {file_path}")

            # Execute the first function immediately after file selection
            df = csv_load(self.input_file_path)
            if df.empty:
                QMessageBox.warning(self, "Empty File", "The selected CSV file is empty.")
                return

            # Group by date and save subsets
            csv_group_by_date_and_save(df, self.subset_folder, column_name="DatumZeit")
            print("Grouping by date completed.")

            # Refresh the subset list
            self.refresh_subsets()
        else:
            QMessageBox.warning(self, "No File Selected", "Please select a valid CSV file.")

    def filter_subsets(self):
        """Filter the subset list based on the search query."""
        query = self.search_bar.text().lower()
        self.subset_list.clear()
        for relative_path in self.subset_files:
            file_name = os.path.basename(relative_path)
            if query in file_name.lower():
                self.subset_list.addItem(file_name)

    def on_submit(self):
        """Handle the submit button click."""
        # Validate minimum distance
        min_distance = self.distance_input.value()
        if min_distance <= 0:
            QMessageBox.warning(self, "Invalid Input", "Minimum distance must be a positive number.")
            return

        # Validate selected subsets
        selected_indices = self.subset_list.selectedItems()
        if not selected_indices:
            QMessageBox.warning(self, "No Subsets Selected", "Please select at least one subset to process.")
            return

        # Update step config
        self.selected_steps = {step_name: checkbox.isChecked() for step_name, checkbox in self.checkboxes.items()}

        # Gather the **relative paths** of the selected files
        self.selected_subsets = [self.file_paths[self.subset_list.row(item)] for item in selected_indices]
        self.min_distance = min_distance

        # Debug: Print the selected subsets
        print("Selected subsets:")
        for subset in self.selected_subsets:
            print(subset)

        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Simulate processing (replace with actual processing logic)
        for i in range(101):
            self.progress_bar.setValue(i)
            QApplication.processEvents()  # Update the UI
            import time
            time.sleep(0.05)  # Simulate work

        # Close the window
        self.close()

    def get_results(self) -> Tuple[Dict[str, bool], List[str], float]:
        """Return the selected steps, subsets, and minimum distance."""
        return self.selected_steps, self.selected_subsets, self.min_distance

def select_steps_and_subsets_with_gui(
    default_config: Dict[str, bool],
    subset_folder: str,
    pre_selected_date: str = None
) -> Tuple[Dict[str, bool], List[str], float]:
    """
    Launch a PyQt5-based GUI for selecting processing steps, subsets, and settings.
    """
    app = QApplication(sys.argv)
    window = DataProcessingApp(default_config, subset_folder, pre_selected_date)
    window.show()
    app.exec_()

    return window.get_results()


def main(config: Dict[str, Any], subsets: List[str]) -> None:
    """
    Main entry point for the data processing workflow.
    """
    # Process each subset file
    for subset_file in subsets:
        # subset_file is a *relative path* from the chosen folder
        subset_full_path = os.path.join(config["output_folder_for_subsets_by_date"], subset_file)

        # Debug: Print the file path
        print(f"Loading file: {subset_full_path}")

        # Check if the file exists
        if not os.path.exists(subset_full_path):
            print(f"File not found: {subset_full_path}")
            continue  # Skip this file and move to the next one

        df_subset = csv_load(subset_full_path)
        processed_suffixes = []

        if df_subset.empty:
            print(f"Subset '{subset_file}' is empty. Skipping.")
            continue

        # Apply processing steps
        steps = [
            ("smooth_gps_data_savitzky", data_smooth_gps_savitzky, "savitzky"),
            ("smooth_gps_data_gaussian", data_smooth_gps_gaussian, "gaussian"),
            ("convert_to_planar", data_convert_to_planar, "planar"),
            ("filter_with_distances", data_filter_points_by_distance, "dist"),
            ("parse_time", parse_time_and_compute_dt, "time"),
            ("compute_heading_from_xy", data_compute_heading_from_xy, "heading"),
            ("compute_yaw_rate_from_heading", data_compute_yaw_rate_from_heading, "yaw_rate"),
        ]

        for step_name, step_function, suffix in steps:
            if config.get(step_name, True):
                df_subset = step_function(df_subset, config)
                processed_suffixes.append(suffix)

        # Save processed data to CSV
        if config.get("save_to_csv", True):
            suffix_string = "_".join(processed_suffixes)
            base_filename = os.path.splitext(subset_file)[0]
            processed_filename = f"{base_filename}_{suffix_string}.csv"
            save_path = os.path.join(config["output_folder_for_subsets_by_date"], processed_filename)
            csv_save(df_subset, save_path, config)

        # Generate statistics
        if config.get("statistics", False):
            print(f"Saving statistics for: {subset_full_path}")
            csv_get_statistics(subset_full_path, config)

        # Generate the map
        if config.get("generate_map", False):
            map_source_path = save_path if config.get("save_to_csv", True) else subset_full_path
            print(f"Generating map using: {map_source_path}")
            generate_map_from_csv(map_source_path)



if __name__ == "__main__":
    default_config = {
        "statistics": False,
        "smooth_gps_data_savitzky": True,
        "smooth_gps_data_gaussian": True,
        "convert_to_planar": True,
        "filter_with_distances": True,
        "parse_time": True,
        "compute_heading_from_xy": True,
        "compute_yaw_rate_from_heading": True,
        "generate_map": False,
        "save_to_csv": True,
        "enable_statistics_on_save": True,
    }

    subset_folder = "subsets_by_date"
    pre_selected_date = None

    selected_steps, selected_subsets, min_distance = select_steps_and_subsets_with_gui(
        default_config, subset_folder, pre_selected_date
    )

    # Merge selected steps into final config
    config = {
        "output_folder_for_subsets_by_date": subset_folder,
        "date_column": "DatumZeit",
        "speed_column": "Geschwindigkeit in m/s",
        "lat_col": "GPS_lat",
        "lon_col": "GPS_lon",
        "x_col": "x",
        "y_col": "y",
        "lat_col_smooth": "GPS_lat_smooth",
        "lon_col_smooth": "GPS_lon_smooth",
        "distance_col": "distance",
        "time_between_points": "dt",
        "min_distance": min_distance,
        **selected_steps
    }

    # Run the main data processing workflow
    main(config, selected_subsets)