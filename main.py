"""
main.py

A PyQt-based GUI application (DataProcessingApp) and a main data-processing
pipeline (main function) for handling CSV subsets, applying transformations,
and optionally generating statistics, maps, and graphs.

Author: Cyril
"""

# =============================================================================
# 1) IMPORTS
# =============================================================================
import sys
import os
from typing import Dict, List, Tuple, Any

# --- PyQt5 ---
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QCheckBox,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QFormLayout,
    QDoubleSpinBox,
    QMessageBox,
    QFileDialog
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon

# --- Local/Custom Modules ---
from csv_tools import (
    csv_load,
    csv_save,
    csv_get_statistics,
    csv_group_by_date_and_save,
    subsets_by_date
)
from data_tools import (
    data_convert_to_planar,
    data_parse_time,
    data_compute_heading_dx_dy,
    data_compute_yaw_rate_from_heading,
    data_delete_the_one_percent,
    data_compute_heading_ds,
    data_rolling_windows_gps_data,
    data_get_elevation,
    data_compute_traveled_distance,
    data_compute_gradient,
    data_compute_curvature_and_radius,
    data_segment_train_curves,
)
from map_generator import generate_map_from_csv
import graph_tools  # Module containing graph functions

import pandas as pd


# =============================================================================
# 2) PyQt5 APPLICATION CLASS
# =============================================================================
class DataProcessingApp(QMainWindow):
    """
    The PyQt-based GUI window that:
      - Displays available CSV subsets for selection and filtering
      - Allows the user to enable/disable various data-processing steps
      - Provides input fields for minimum distance and quantile boundaries
      - Offers graphing options via checkboxes
    """
    def __init__(self, default_config: Dict[str, bool], subset_folder: str, graph_config: Dict[str, Any] = None, pre_selected_date: str = None):
        super().__init__()
        self.default_config = default_config
        self.graph_config = graph_config if graph_config is not None else {}
        self.subset_folder = subset_folder
        self.pre_selected_date = pre_selected_date

        # Tracking user selections
        self.selected_steps: Dict[str, bool] = {}
        self.selected_subsets: List[str] = []
        self.min_distance: float = 1.0

        # Percentages
        self.delete_lower_percentage: float = 1.0
        self.delete_upper_percentage: float = 99.0

        # Filtering in the GUI
        self.subset_files: List[str] = []
        self.current_filter: str = ""

        # Graphing dictionary: (human-readable name -> function name in graph_tools)
        self.available_graphs = {
            "Yaw rate comparison": "graph_yaw_rate_and_gier",
            "curvature and radius": "plot_curvature_with_mittelradius",
        }

        # Widgets
        self.checkboxes = {}
        self.graph_checkboxes = {}

        # Initialize
        self.init_ui()
        self.refresh_subsets()

    # -------------------------------------------------------------------------
    # GUI Initialization
    # -------------------------------------------------------------------------
    def init_ui(self):
        """Sets up the window layout and widgets."""
        self.setWindowTitle("Data Processing Toolkit")
        self.setWindowState(Qt.WindowMaximized)
        self.setWindowIcon(QIcon("icon.png"))
        self.setup_style()

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # UI Groups
        main_layout.addWidget(self.create_file_selection_group())
        main_layout.addWidget(self.create_processing_steps_group())
        main_layout.addWidget(self.create_distance_and_percentage_input_group())
        main_layout.addWidget(self.create_subsets_group())
        main_layout.addWidget(self.create_action_button())
        main_layout.addWidget(self.create_graphs_group())

    def setup_style(self):
        """Applies CSS-like styling to the application."""
        self.setStyleSheet("""
            QMainWindow { background-color: #2E3440; }
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
            QLineEdit, QDoubleSpinBox {
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
            QPushButton:hover { background-color: #81A1C1; }
            QPushButton:pressed { background-color: #4C566A; }
            QCheckBox { color: #D8DEE9; font-size: 14px; }
        """)

    # -------------------------------------------------------------------------
    # GUI: File Selection
    # -------------------------------------------------------------------------
    def create_file_selection_group(self) -> QWidget:
        """Widget area for selecting a CSV file and creating subsets."""
        group = QWidget()
        layout = QVBoxLayout(group)

        layout.addWidget(QLabel("Create Subsets by Date"))
        self.btn_select_file = QPushButton("Select File and Create Subsets")
        self.btn_select_file.setFont(QFont("Arial", 12))
        self.btn_select_file.clicked.connect(self.on_select_file)
        layout.addWidget(self.btn_select_file)

        return group

    def on_select_file(self):
        """Opens a FileDialog to choose a CSV and create subsets."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input File",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            QMessageBox.warning(self, "Cancelled", "No file selected")
            return

        try:
            df = csv_load(path)
            if df.empty:
                raise ValueError("The chosen CSV file is empty.")
            csv_group_by_date_and_save(df, self.subset_folder)
            self.refresh_subsets()
            QMessageBox.information(self, "Success", "Subsets created successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process file:\n{str(e)}")

    # -------------------------------------------------------------------------
    # GUI: Processing Steps
    # -------------------------------------------------------------------------
    def create_processing_steps_group(self) -> QWidget:
        """Checkboxes for various data-processing steps."""
        group = QWidget()
        layout = QGridLayout(group)
        layout.addWidget(QLabel("Processing Steps"), 0, 0, 1, 2)

        steps = [(k, v) for k, v in self.default_config.items() if k != "create_subsets_by_date"]
        num_steps = len(steps)
        num_columns = 2
        num_rows = (num_steps + num_columns - 1) // num_columns

        for idx, (step_name, enabled) in enumerate(steps):
            checkbox = QCheckBox(step_name.replace("_", " ").title())
            checkbox.setChecked(enabled)
            self.checkboxes[step_name] = checkbox
            row = idx % num_rows + 1  # offset for header row
            col = idx // num_rows
            layout.addWidget(checkbox, row, col)

        btn_group = QWidget()
        btn_layout = QHBoxLayout(btn_group)
        btn_select_all = QPushButton("Select All")
        btn_unselect_all = QPushButton("Unselect All")
        btn_select_all.clicked.connect(lambda: self.toggle_all_steps(True))
        btn_unselect_all.clicked.connect(lambda: self.toggle_all_steps(False))
        btn_layout.addWidget(btn_select_all)
        btn_layout.addWidget(btn_unselect_all)
        layout.addWidget(btn_group, num_rows + 1, 0, 1, 2)

        return group

    def toggle_all_steps(self, state: bool):
        """Sets all checkboxes for processing steps to on/off."""
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(state)

    # -------------------------------------------------------------------------
    # GUI: Distance & Percentage Input
    # -------------------------------------------------------------------------
    def create_distance_and_percentage_input_group(self) -> QWidget:
        """Creates input fields for minimum distance and percentage thresholds."""
        group = QWidget()
        form_layout = QFormLayout(group)
        form_layout.setSpacing(8)

        label_style = "font-size: 14px; font-weight: normal; text-transform: none;"

        min_label = QLabel("Minimum Distance (meters):")
        min_label.setObjectName("smallLabel")
        min_label.setStyleSheet("#smallLabel { " + label_style + " }")
        self.distance_input = QDoubleSpinBox()
        self.distance_input.setRange(0.001, 1000.0)
        self.distance_input.setSingleStep(0.1)
        self.distance_input.setValue(self.min_distance)
        self.distance_input.setMaximumWidth(100)
        form_layout.addRow(min_label, self.distance_input)

        lower_label = QLabel("Delete Lower Boundary (%):")
        lower_label.setObjectName("smallLabelLower")
        lower_label.setStyleSheet("#smallLabelLower { " + label_style + " }")
        self.lower_percentage_input = QDoubleSpinBox()
        self.lower_percentage_input.setRange(0.0, 100.0)
        self.lower_percentage_input.setSingleStep(0.1)
        self.lower_percentage_input.setValue(1.0)
        self.lower_percentage_input.setMaximumWidth(100)
        form_layout.addRow(lower_label, self.lower_percentage_input)

        upper_label = QLabel("Delete Upper Boundary (%):")
        upper_label.setObjectName("smallLabelUpper")
        upper_label.setStyleSheet("#smallLabelUpper { " + label_style + " }")
        self.upper_percentage_input = QDoubleSpinBox()
        self.upper_percentage_input.setRange(0.0, 100.0)
        self.upper_percentage_input.setSingleStep(0.1)
        self.upper_percentage_input.setValue(99.0)
        self.upper_percentage_input.setMaximumWidth(100)
        form_layout.addRow(upper_label, self.upper_percentage_input)

        return group

    # -------------------------------------------------------------------------
    # GUI: Subsets List
    # -------------------------------------------------------------------------
    def create_subsets_group(self) -> QWidget:
        """Displays available subset files in a list for multi-selection."""
        group = QWidget()
        layout = QVBoxLayout(group)
        layout.addWidget(QLabel("Subsets to Process"))

        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search subsets...")
        self.search_bar.textChanged.connect(self.filter_subsets)
        layout.addWidget(self.search_bar)

        self.subset_list = QListWidget()
        self.subset_list.setSelectionMode(QListWidget.ExtendedSelection)
        layout.addWidget(self.subset_list)

        return group

    def refresh_subsets(self):
        """Loads subset CSV files in the specified folder and updates the list widget."""
        self.subset_files = subsets_by_date(self.subset_folder)
        self.subset_list.clear()
        for full_path in self.subset_files:
            rel_path = os.path.relpath(full_path, self.subset_folder)
            item = QListWidgetItem(os.path.basename(rel_path))
            item.setData(Qt.UserRole, rel_path)
            self.subset_list.addItem(item)

        self.filter_subsets()
        self.select_preselected_date()

    def select_preselected_date(self):
        """Auto-selects the subset if pre_selected_date is set."""
        if not self.pre_selected_date:
            return
        for i in range(self.subset_list.count()):
            item = self.subset_list.item(i)
            if self.pre_selected_date in item.text():
                item.setSelected(True)
                self.subset_list.scrollToItem(item)
                break

    def filter_subsets(self):
        """Hides items that do not match the current search query."""
        query = self.search_bar.text().lower()
        self.current_filter = query
        for i in range(self.subset_list.count()):
            item = self.subset_list.item(i)
            item.setHidden(query not in item.text().lower())

    # -------------------------------------------------------------------------
    # GUI: Action Button (Processing)
    # -------------------------------------------------------------------------
    def create_action_button(self) -> QWidget:
        """Returns a button to start the data-processing pipeline."""
        group = QWidget()
        layout = QHBoxLayout(group)
        self.btn_process = QPushButton("Start Processing")
        self.btn_process.setFont(QFont("Arial", 14, QFont.Bold))
        self.btn_process.clicked.connect(self.on_process)
        layout.addWidget(self.btn_process)
        return group

    def on_process(self):
        """Reads user settings and closes the window so processing can continue."""
        self.min_distance = self.distance_input.value()
        self.delete_lower_percentage = self.lower_percentage_input.value()
        self.delete_upper_percentage = self.upper_percentage_input.value()

        selected_items = self.subset_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select at least one subset!")
            return

        self.selected_steps = {name: cb.isChecked() for name, cb in self.checkboxes.items()}
        self.selected_subsets = [
            os.path.join(self.subset_folder, item.data(Qt.UserRole))
            for item in selected_items
        ]

        self.close()

    # -------------------------------------------------------------------------
    # GUI: Graphing Options
    # -------------------------------------------------------------------------
    def create_graphs_group(self) -> QWidget:
        """Creates a section of checkboxes for graphing options."""
        group = QWidget()
        layout = QVBoxLayout(group)
        layout.addWidget(QLabel("Graphing Options"))

        for graph_name in self.available_graphs.keys():
            cb = QCheckBox(graph_name)
            cb.setChecked(False)
            self.graph_checkboxes[graph_name] = cb
            layout.addWidget(cb)

        self.btn_generate_graphs = QPushButton("Generate Graphs")
        self.btn_generate_graphs.setFont(QFont("Arial", 14, QFont.Bold))
        self.btn_generate_graphs.clicked.connect(self.on_generate_graphs)
        layout.addWidget(self.btn_generate_graphs)

        return group

    def on_generate_graphs(self):
        selected_graphs = [name for name, cb in self.graph_checkboxes.items() if cb.isChecked()]
        if not selected_graphs:
            QMessageBox.warning(self, "Warning", "Please select at least one graph type!")
            return

        selected_items = self.subset_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select at least one subset from the list!")
            return

        selected_subsets = [
            os.path.join(self.subset_folder, item.data(Qt.UserRole))
            for item in selected_items
        ]

        for subset_file in selected_subsets:
            for graph_type in selected_graphs:
                func_name = self.available_graphs.get(graph_type)
                if func_name:
                    try:
                        graph_func = getattr(graph_tools, func_name)
                        # Use the instance attribute self.graph_config instead of undefined config
                        graph_func(subset_file, self.graph_config)
                    except Exception as e:
                        QMessageBox.critical(
                            self, "Error",
                            f"Error generating {graph_type} for {subset_file}:\n{str(e)}"
                        )
                        continue

        QMessageBox.information(self, "Success", "Graphs generated successfully!")

    # -------------------------------------------------------------------------
    # GET RESULTS
    # -------------------------------------------------------------------------
    def get_results(self) -> Tuple[Dict[str, bool], List[str], float, float, float]:
        """
        Returns user selections after the GUI closes:
          - A dictionary of (step_name -> bool) for processing steps
          - A list of file paths for the selected subsets
          - The minimum distance
          - The lower quantile percentage
          - The upper quantile percentage
        """
        return (
            self.selected_steps,
            self.selected_subsets,
            self.min_distance,
            self.delete_lower_percentage,
            self.delete_upper_percentage
        )


# =============================================================================
# 3) MAIN DATA PROCESSING PIPELINE
# =============================================================================
def main(config: Dict[str, Any], subsets: List[str]) -> None:
    """
    Main Processing Pipeline.

    Iterates over each selected subset, loads it, and applies the
    selected data-processing steps in sequence. If configured, saves
    the results, generates statistics, and possibly a map.

    Args:
        config (Dict[str, Any]): A configuration dictionary that includes
            booleans indicating which steps to run, as well as other
            relevant parameters (e.g., 'save_to_csv', 'statistics', 'generate_map').
        subsets (List[str]): A list of file paths (CSV subsets) to process.

    Returns:
        None
    """
    processing_steps = [
        ("parse_time", data_parse_time, "time"),
        ("filter_GPS_with_rolling_windows", data_rolling_windows_gps_data, "rollingW"),
        ("convert_to_planar", data_convert_to_planar, "planar"),
        ("compute_traveled_distance", data_compute_traveled_distance, "distance"),
        ("compute_heading_with_dy/dx", data_compute_heading_dx_dy, "headingDX"),
        ("compute_heading_with_dx/ds", data_compute_heading_ds, "headingDS"),
#        ("data_add_heading_column", data_add_heading_column, "NEW!"),
        ("compute_yaw_rate_from_heading", data_compute_yaw_rate_from_heading, "yawRate"),
        ("delete_the_boundaries", data_delete_the_one_percent, "delBoundaries"),
        ("compute_curvature_and_radius", data_compute_curvature_and_radius, "radius"),
        ("bogen_übergangsbogen", data_segment_train_curves, "übogen"),
      #  ("radius_with_circle_method", data_circle_fit_compute_radius, "circle"),
        ("get_elevation", data_get_elevation, "elevation"),
        ("get_gradient", data_compute_gradient, "gradient"),
    ]

    for subset_path in subsets:
        try:
            if not os.path.exists(subset_path):
                raise FileNotFoundError(f"Subset not found: {subset_path}")

            df = csv_load(subset_path, config)
            if df.empty:
                print(f"Skipping empty subset: {subset_path}")
                continue

            processed_suffixes = []
            for step_name, step_func, suffix in processing_steps:
                if config.get(step_name, False):
                    df = step_func(df, config)
                    processed_suffixes.append(suffix)

            if config.get("save_to_csv", False):
                base_name = os.path.splitext(os.path.basename(subset_path))[0]
                new_name = f"{base_name}_{'_'.join(processed_suffixes)}.csv"
                save_dir = os.path.dirname(subset_path)
                save_path = os.path.join(save_dir, new_name)
                csv_save(df, save_path, config)

            if config.get("statistics", False):
                final_file = save_path if config.get("save_to_csv", False) else subset_path
                csv_get_statistics(final_file, config)

            if config.get("generate_map", False):
                final_file = save_path if config.get("save_to_csv", False) else subset_path
                generate_map_from_csv(final_file)

        except Exception as e:
            print(f"Error processing {subset_path}: {str(e)}")
            continue


# =============================================================================
# 4) SCRIPT ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    # Default configuration
    DEFAULT_CONFIG = {
        "statistics": False,
        "parse_time": True,
        "filter_GPS_with_rolling_windows": True,
        "convert_to_planar": True,
        "compute_traveled_distance": True,
        "compute_heading_with_dy/dx": True,
        "compute_heading_with_dx/ds": True,
       # "data_add_heading_column" :True,
        "compute_yaw_rate_from_heading": True,
        "compute_curvature_and_radius": True,
        "bogen_übergangsbogen" : True,
      #  "radius_with_circle_method" :True,
        "delete_the_boundaries": False,
        "get_elevation": False,
        "get_gradient": False,
        "smoothed_gradient": False,
        "save_to_csv": True,
        "enable_statistics_on_save": True,
        "generate_map": False,
    }

    app = QApplication(sys.argv)
    window = DataProcessingApp(DEFAULT_CONFIG, "subsets_by_date")
    window.show()
    app.exec_()

    selected_steps, selected_subsets, min_distance, lower_perc, upper_perc = window.get_results()

    CONFIG = {
        "DatumZeit" : "DatumZeit",
        "x": "x",
        "y": "y",
        "elapsed_time_s" :"elapsed_time_s",
        "heading_dx_dy_grad" : "heading_dx_dy_grad",
        "elapsed_time": "elapsed_time_s",
        "cumulative_distance_m" : "cumulative_distance_m",
        "heading_dx_ds_grad" : "heading_dx_ds_grad",
        "speed_column": "Geschwindigkeit in m/s",

        # for the rolling windows
        "speed_threshold_stopped_rolling_windows": 0.5,
        "distance_window_meters": 20,
        "time_window_min": 1.0,
        "time_window_max": 9999.0,
        "max_stop_window": 999999999.0,
        "speed_bins": [0.0, 0.5, 2.0, 5.0, 15.0, 30.0, float("inf")],
        "hysteresis_window" : 10,
        "output_folder_for_subsets_by_date": "subsets_by_date",

        "encoding": "utf-8",

        "gradient" :"gradient",
        "gradient_per_mille" : "gradient_per_mille",


        #### for the one percent
        "delete_lower_bound_percentage": lower_perc,
        "delete_upper_bound_percentage": upper_perc,


        "elevation" : "elevation",

        "api_key": "AIzaSyAb7ec8EGcU5MiHlQ9jJETvABNYNyiq-WE",
        "api_url": "https://maps.googleapis.com/maps/api/elevation/json",
        "batch_size": 280,
        "threads": 10,


        #for the curvature
        "curvature" : "curvature",
        "straight_threshold" : 50000.0,


        "curvature_fuck" : "curvature_heading_dx_ds_grad",
        "curvature_std_thresh" : 0,
        "min_segment_size" : 6,
        "smoothing_window_radius" : 5,
        "smoothing_window_2" : 5,
        "straight_curvature_threshold" : 0.01,
        "steady_std_threshold" : 0.0005,

        "radius_m" : "radius_m",
        "radius_m_fuck" : "radius_m_heading_dx_ds_grad",
        **selected_steps
    }

    main(CONFIG, selected_subsets)
