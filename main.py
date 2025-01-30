import sys
import os
from typing import Dict, List, Tuple, Any
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QListWidget, QLineEdit, QPushButton, QDoubleSpinBox, QMessageBox, QFileDialog, QCheckBox, QGridLayout
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
        self.input_file_path = None
        self.distance_input = None

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Data Processing Toolkit")
        self.setWindowState(Qt.WindowMaximized)
        self.setWindowIcon(QIcon("icon.png"))

        # Apply dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2E3440;
            }
            QLabel {
                color: #D8DEE9;
                font-size: 18px;
                font-weight: bold;
                text-transform: uppercase;
                letter-spacing: 1px;
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
        first_function_layout.addWidget(QLabel("Create Subsets by Date"))

        self.first_function_button = QPushButton("Select File and Create Subsets")
        self.first_function_button.setFont(QFont("Arial", 12))
        self.first_function_button.clicked.connect(self.execute_first_function)
        first_function_layout.addWidget(self.first_function_button)

        main_layout.addWidget(first_function_group)

        # --- Processing Steps ---
        steps_group = QWidget()
        steps_layout = QGridLayout(steps_group)

        steps_layout.addWidget(QLabel("Processing Steps"), 0, 0, 1, 2)

        self.checkboxes = {}
        steps = [step for step in list(self.default_config.items()) if step[0] != "create_subsets_by_date"]

        num_steps = len(steps)
        num_rows = (num_steps + 1) // 2

        for index, (step_name, enabled_by_default) in enumerate(steps):
            checkbox = QCheckBox(step_name.replace("_", " ").title())
            checkbox.setChecked(enabled_by_default)
            self.checkboxes[step_name] = checkbox

            row = index % num_rows + 1
            col = index // num_rows
            steps_layout.addWidget(checkbox, row, col)

            if step_name == "statistics":
                checkbox.stateChanged.connect(self.deselect_other_functions)

        # Control buttons
        buttons_group = QWidget()
        buttons_layout = QHBoxLayout(buttons_group)
        self.select_all_button = QPushButton("Select All")
        self.select_all_button.clicked.connect(self.select_all_steps)
        buttons_layout.addWidget(self.select_all_button)
        self.unselect_all_button = QPushButton("Unselect All")
        self.unselect_all_button.clicked.connect(self.unselect_all_steps)
        buttons_layout.addWidget(self.unselect_all_button)
        steps_layout.addWidget(buttons_group, num_rows + 1, 0, 1, 2)

        main_layout.addWidget(steps_group)

        # --- Minimum Distance Input ---
        distance_group = QWidget()
        distance_layout = QHBoxLayout(distance_group)
        distance_label = QLabel("Minimum Distance (meters):")
        self.distance_input = QDoubleSpinBox()
        self.distance_input.setMinimum(0.01)
        self.distance_input.setMaximum(1000.0)
        self.distance_input.setSingleStep(0.1)
        self.distance_input.setValue(self.min_distance)
        distance_layout.addWidget(distance_label)
        distance_layout.addWidget(self.distance_input)
        main_layout.addWidget(distance_group)

        # --- Subsets to Process ---
        subsets_group = QWidget()
        subsets_layout = QVBoxLayout(subsets_group)
        subsets_layout.addWidget(QLabel("Subsets to Process"))

        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search subsets...")
        self.search_bar.textChanged.connect(self.filter_subsets)
        subsets_layout.addWidget(self.search_bar)

        self.subset_list = QListWidget()
        self.subset_list.setSelectionMode(QListWidget.MultiSelection)
        subsets_layout.addWidget(self.subset_list)

        self.refresh_subsets()
        main_layout.addWidget(subsets_group)

        # --- Start Processing Button ---
        self.start_button = QPushButton("Start Processing")
        self.start_button.setFont(QFont("Arial", 14, QFont.Bold))
        self.start_button.setIcon(QIcon("start_icon.png"))
        self.start_button.clicked.connect(self.on_submit)
        main_layout.addWidget(self.start_button)

    def select_all_steps(self):
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(True)

    def unselect_all_steps(self):
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(False)

    def deselect_other_functions(self, state: int):
        if state == 2:
            for step_name, checkbox in self.checkboxes.items():
                if step_name != "statistics":
                    checkbox.setChecked(False)

    def refresh_subsets(self):
        self.subset_files = subsets_by_date(self.subset_folder)
        self.file_paths = []
        self.subset_list.clear()
        for relative_path in self.subset_files:
            relative_path = os.path.relpath(relative_path, self.subset_folder)
            file_name = os.path.basename(relative_path)
            self.file_paths.append(relative_path)
            self.subset_list.addItem(file_name)
            if self.pre_selected_date and file_name.strip() == self.pre_selected_date.strip():
                self.subset_list.findItems(file_name, Qt.MatchExactly)[0].setSelected(True)

    def execute_first_function(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Input File", "", "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            self.input_file_path = file_path
            df = csv_load(self.input_file_path)
            if df.empty:
                QMessageBox.warning(self, "Empty File", "The selected CSV file is empty.")
                return

            csv_group_by_date_and_save(df, self.subset_folder)
            self.refresh_subsets()
        else:
            QMessageBox.warning(self, "No File Selected", "Please select a valid CSV file.")

    def filter_subsets(self):
        query = self.search_bar.text().lower()
        self.subset_list.clear()
        for relative_path in self.subset_files:
            file_name = os.path.basename(relative_path)
            if query in file_name.lower():
                self.subset_list.addItem(file_name)

    def on_submit(self):
        self.min_distance = self.distance_input.value()
        selected_indices = self.subset_list.selectedItems()

        if not selected_indices:
            QMessageBox.warning(self, "No Subsets Selected", "Please select at least one subset to process.")
            return

        self.selected_steps = {step_name: checkbox.isChecked() for step_name, checkbox in self.checkboxes.items()}
        self.selected_subsets = [self.file_paths[self.subset_list.row(item)] for item in selected_indices]
        self.close()

    def get_results(self) -> Tuple[Dict[str, bool], List[str], float]:
        return self.selected_steps, self.selected_subsets, self.min_distance


def select_steps_and_subsets_with_gui(
        default_config: Dict[str, bool],
        subset_folder: str,
        pre_selected_date: str = None
) -> Tuple[Dict[str, bool], List[str], float]:
    app = QApplication(sys.argv)
    window = DataProcessingApp(default_config, subset_folder, pre_selected_date)
    window.show()
    app.exec_()
    return window.get_results()


def main(config: Dict[str, Any], subsets: List[str]) -> None:
    for subset_file in subsets:
        subset_full_path = os.path.join(config["output_folder_for_subsets_by_date"], subset_file)

        if not os.path.exists(subset_full_path):
            print(f"File not found: {subset_full_path}")
            continue

        df_subset = csv_load(subset_full_path)
        processed_suffixes = []

        if df_subset.empty:
            print(f"Subset '{subset_file}' is empty. Skipping.")
            continue

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

        if config.get("save_to_csv", True):
            suffix_string = "_".join(processed_suffixes)
            base_filename = os.path.splitext(subset_file)[0]
            processed_filename = f"{base_filename}_{suffix_string}.csv"
            save_path = os.path.join(config["output_folder_for_subsets_by_date"], processed_filename)
            csv_save(df_subset, save_path, config)

        if config.get("statistics", False):
            csv_get_statistics(subset_full_path, config)

        if config.get("generate_map", False):
            map_source_path = save_path if config.get("save_to_csv", True) else subset_full_path
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
        "save_to_csv": True,
        "enable_statistics_on_save": True,
        "generate_map": False,
    }

    subset_folder = "subsets_by_date"
    pre_selected_date = None

    selected_steps, selected_subsets, min_distance = select_steps_and_subsets_with_gui(
        default_config, subset_folder, pre_selected_date
    )

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

    main(config, selected_subsets)