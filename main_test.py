import sys
from typing import Dict, List, Tuple, Any
import concurrent.futures

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QCheckBox, QLineEdit, QListWidget, QListWidgetItem,
    QFormLayout, QDoubleSpinBox, QMessageBox, QFileDialog
)

from csv_tools import (
    csv_load,
    csv_save,
    csv_get_statistics,
    csv_group_by_date_and_save,
    subsets_by_date
)
from data_tools import (
    # data_smooth_gps_savitzky,
    # data_smooth_gps_gaussian,
    data_convert_to_planar,
    # data_filter_points_by_distance,
    data_parse_time_and_compute_dt,
    data_compute_heading_dx_dy,
    data_compute_yaw_rate_from_heading,
    data_delete_the_one_percent,
    data_compute_heading_ds,
    # data_kalman_on_yaw_rate, data_particle_filter,
    # data_remove_gps_outliers,
    data_rolling_windows_gps_data,
    # data_compute_curvature,
    # data_add_infrastructure_status,
    data_get_elevation,
    data_compute_traveled_distance,
    data_compute_gradient,
    data_compute_curvature_radius_and_detect_steady_curves,
    # data_smooth_gradient,
    # data_rollWin_Kalman_on_gps,
)
from map_generator import generate_map_from_csv

import os
from typing import Dict, List, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon

import pandas as pd  # For loading CSV files
import graph_tools  # Your module that contains graph functions


# Make sure that functions like 'plot_histogram', 'plot_line_chart', etc.
# are defined in your graph_tools.py module.

class DataProcessingApp(QMainWindow):
    """
    Das PyQt-basierte GUI-Fenster, das:
     - Die zu verarbeitenden Teildateien (Subsets) anzeigt und filtern lässt
     - Processing-Schritte auswählbar macht
     - Min. Abstandswert (min_distance) und zwei Prozentwerte (untere und obere Grenze) erfassen lässt
     - Graphing-Optionen anbietet, die per Checkboxen festgelegt werden
    """

    def __init__(self, default_config: Dict[str, bool], subset_folder: str, pre_selected_date: str = None):
        # Define a dictionary mapping a human-readable graph type to the function name in graph_tools
        self.available_graphs = {
            "Yaw rate comparison": "graph_yaw_rate_and_gier",

        }
        super().__init__()
        self.default_config = default_config
        self.subset_folder = subset_folder
        self.pre_selected_date = pre_selected_date

        # Speicher für ausgewählte Optionen
        self.selected_steps: Dict[str, bool] = {}
        self.selected_subsets: List[str] = []
        self.min_distance: float = 1.0

        # Prozentwerte
        self.delete_lower_percentage: float = 1.0  # Default: 1%
        self.delete_upper_percentage: float = 99.0  # Default: 99%

        # Zum Filtern in der GUI
        self.subset_files: List[str] = []
        self.current_filter: str = ""

        self.init_ui()
        self.refresh_subsets()

    def init_ui(self):
        """Definiert das GUI-Layout und die Widgets."""
        self.setWindowTitle("Data Processing Toolkit")
        self.setWindowState(Qt.WindowMaximized)
        self.setWindowIcon(QIcon("icon.png"))
        self.setup_style()

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # Bestehende Gruppen
        main_layout.addWidget(self.create_file_selection_group())
        main_layout.addWidget(self.create_processing_steps_group())
        main_layout.addWidget(self.create_distance_and_percentage_input_group())
        main_layout.addWidget(self.create_subsets_group())
        main_layout.addWidget(self.create_action_button())  # Start Processing

        # Neue Gruppe: Graphing Options mit Checkboxen
        main_layout.addWidget(self.create_graphs_group())

    def setup_style(self):
        """Setzt ein paar Styles mittels CSS-ähnlicher Syntax."""
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

    def create_file_selection_group(self) -> QWidget:
        """Erzeugt den Bereich, in dem man über PyQt eine CSV-Datei auswählen kann."""
        group = QWidget()
        layout = QVBoxLayout(group)

        layout.addWidget(QLabel("Create Subsets by Date"))
        self.btn_select_file = QPushButton("Select File and Create Subsets")
        self.btn_select_file.setFont(QFont("Arial", 12))
        self.btn_select_file.clicked.connect(self.on_select_file)
        layout.addWidget(self.btn_select_file)

        return group

    def create_processing_steps_group(self) -> QWidget:
        """Checkboxen für die verschiedenen Processing-Schritte."""
        group = QWidget()
        layout = QGridLayout(group)
        layout.addWidget(QLabel("Processing Steps"), 0, 0, 1, 2)

        self.checkboxes = {}
        # Filtert den Schritt "create_subsets_by_date" heraus
        steps = [(k, v) for k, v in self.default_config.items() if k != "create_subsets_by_date"]

        num_steps = len(steps)
        num_columns = 2
        num_rows = (num_steps + num_columns - 1) // num_columns

        for idx, (step_name, enabled) in enumerate(steps):
            checkbox = QCheckBox(step_name.replace("_", " ").title())
            checkbox.setChecked(enabled)
            self.checkboxes[step_name] = checkbox
            row = idx % num_rows + 1  # +1 wegen der Überschriftszeile
            col = idx // num_rows
            layout.addWidget(checkbox, row, col)

        # "Select All"/"Unselect All" Buttons
        btn_group = QWidget()
        btn_layout = QHBoxLayout(btn_group)
        btn_select_all = QPushButton("Select All")
        btn_select_all.clicked.connect(lambda: self.toggle_all_steps(True))
        btn_unselect_all = QPushButton("Unselect All")
        btn_unselect_all.clicked.connect(lambda: self.toggle_all_steps(False))
        btn_layout.addWidget(btn_select_all)
        btn_layout.addWidget(btn_unselect_all)
        layout.addWidget(btn_group, num_rows + 1, 0, 1, 2)

        return group

    def create_distance_and_percentage_input_group(self) -> QWidget:
        """
        Erzeugt Eingabefelder für Mindestdistanz und Prozentwerte.
        """
        group = QWidget()
        form_layout = QFormLayout(group)
        form_layout.setSpacing(8)

        label_style = "font-size: 14px; font-weight: normal; text-transform: none;"

        # Mindestdistanz
        min_label = QLabel("Minimum Distance (meters):")
        min_label.setObjectName("smallLabel")
        min_label.setStyleSheet("#smallLabel { " + label_style + " }")
        self.distance_input = QDoubleSpinBox()
        self.distance_input.setRange(0.001, 1000.0)
        self.distance_input.setSingleStep(0.1)
        self.distance_input.setValue(self.min_distance)
        self.distance_input.setMaximumWidth(100)
        form_layout.addRow(min_label, self.distance_input)

        # Untere Prozentgrenze
        lower_label = QLabel("Delete Lower Boundary (%):")
        lower_label.setObjectName("smallLabelLower")
        lower_label.setStyleSheet("#smallLabelLower { " + label_style + " }")
        self.lower_percentage_input = QDoubleSpinBox()
        self.lower_percentage_input.setRange(0.0, 100.0)
        self.lower_percentage_input.setSingleStep(0.1)
        self.lower_percentage_input.setValue(1.0)
        self.lower_percentage_input.setMaximumWidth(100)
        form_layout.addRow(lower_label, self.lower_percentage_input)

        # Obere Prozentgrenze
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

    def create_subsets_group(self) -> QWidget:
        """
        Zeigt die vorhandenen Subset-Dateien an und erlaubt die Mehrfachauswahl.
        """
        group = QWidget()
        layout = QVBoxLayout(group)
        layout.addWidget(QLabel("Subsets to Process"))

        # Suchfeld
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search subsets...")
        self.search_bar.textChanged.connect(self.filter_subsets)
        layout.addWidget(self.search_bar)

        # Liste der Dateien
        self.subset_list = QListWidget()
        self.subset_list.setSelectionMode(QListWidget.ExtendedSelection)
        layout.addWidget(self.subset_list)

        return group

    def create_action_button(self) -> QWidget:
        """Button zum Starten des Verarbeitungsprozesses."""
        group = QWidget()
        layout = QHBoxLayout(group)
        self.btn_process = QPushButton("Start Processing")
        self.btn_process.setFont(QFont("Arial", 14, QFont.Bold))
        self.btn_process.clicked.connect(self.on_process)
        layout.addWidget(self.btn_process)
        return group

    def create_graphs_group(self) -> QWidget:
        """
        Neue Gruppe: Graphing Options.
        Hier werden Checkboxen angeboten, die das jeweilige Graph-Tool in graph_tools.py ein-/ausschalten.
        """
        group = QWidget()
        layout = QVBoxLayout(group)
        layout.addWidget(QLabel("Graphing Options"))

        # Erstelle Checkboxen für jeden Graph-Typ basierend auf self.available_graphs
        self.graph_checkboxes = {}
        for graph_name in self.available_graphs.keys():
            cb = QCheckBox(graph_name)
            cb.setChecked(False)  # Standardmäßig nicht ausgewählt
            self.graph_checkboxes[graph_name] = cb
            layout.addWidget(cb)

        # Button, um die Graphen zu generieren
        self.btn_generate_graphs = QPushButton("Generate Graphs")
        self.btn_generate_graphs.setFont(QFont("Arial", 14, QFont.Bold))
        self.btn_generate_graphs.clicked.connect(self.on_generate_graphs)
        layout.addWidget(self.btn_generate_graphs)

        return group

    def toggle_all_steps(self, state: bool):
        """Schaltet alle Häkchen in den Processing-Schritten an oder aus."""
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(state)

    def refresh_subsets(self):
        """
        Liest die vorhandenen CSV-Dateien im Ordner self.subset_folder und zeigt sie in der Liste an.
        """
        # Hier musst du sicherstellen, dass die Funktion subsets_by_date definiert oder importiert ist.
        self.subset_files = subsets_by_date(self.subset_folder)  # Beispiel: eine Liste von Dateipfaden
        self.subset_list.clear()

        for full_path in self.subset_files:
            rel_path = os.path.relpath(full_path, self.subset_folder)
            item = QListWidgetItem(os.path.basename(rel_path))
            item.setData(Qt.UserRole, rel_path)
            self.subset_list.addItem(item)

        self.filter_subsets()
        self.select_preselected_date()

    def select_preselected_date(self):
        """Markiert automatisch das Subset, wenn pre_selected_date gesetzt ist."""
        if not self.pre_selected_date:
            return

        for i in range(self.subset_list.count()):
            item = self.subset_list.item(i)
            if self.pre_selected_date in item.text():
                item.setSelected(True)
                self.subset_list.scrollToItem(item)
                break

    def filter_subsets(self):
        """Blendet Einträge aus, die nicht dem Suchstring entsprechen."""
        query = self.search_bar.text().lower()
        self.current_filter = query

        for i in range(self.subset_list.count()):
            item = self.subset_list.item(i)
            item.setHidden(query not in item.text().lower())

    def on_select_file(self):
        """
        Öffnet einen FileDialog, um eine CSV-Datei auszuwählen und in subsets aufzuteilen.
        """
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
            df = csv_load(path)  # Stelle sicher, dass csv_load importiert oder definiert ist
            if df.empty:
                raise ValueError("The chosen CSV file is empty.")

            # Gruppiert und speichert die Splits in subsets_by_date/<YYYY-MM-DD>
            csv_group_by_date_and_save(df, self.subset_folder)  # Auch diese Funktion muss vorhanden sein
            self.refresh_subsets()
            QMessageBox.information(self, "Success", "Subsets created successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process file:\n{str(e)}")

    def on_process(self):
        """
        Liest die eingestellten Optionen aus und schließt das Fenster,
        damit im Hauptteil weitergemacht werden kann.
        """
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

    def on_generate_graphs(self):
        """
        Wird aufgerufen, wenn der "Generate Graphs"-Button geklickt wird.
        Für jedes ausgewählte Subset wird das CSV geladen (z. B. via pandas),
        und für jeden aktivierten Graph-Typ wird die entsprechende Funktion in graph_tools.py aufgerufen.
        """
        # Ermittele, welche Graph-Typen ausgewählt wurden
        selected_graphs = [name for name, cb in self.graph_checkboxes.items() if cb.isChecked()]
        if not selected_graphs:
            QMessageBox.warning(self, "Warning", "Please select at least one graph type!")
            return

        # Retrieve the currently selected subset items directly from the list widget
        selected_items = self.subset_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select at least one subset from the list!")
            return

        # Build the list of subset file paths
        selected_subsets = [
            os.path.join(self.subset_folder, item.data(Qt.UserRole))
            for item in selected_items
        ]

        # For each selected subset file, call the corresponding graphing function
        for subset_file in selected_subsets:
            for graph_type in selected_graphs:
                func_name = self.available_graphs.get(graph_type)
                if func_name:
                    try:
                        # Get the graphing function from graph_tools
                        graph_func = getattr(graph_tools, func_name)
                        # Call the function with the file path (or with data if your function is adjusted accordingly)
                        graph_func(subset_file)
                    except Exception as e:
                        QMessageBox.critical(
                            self, "Error",
                            f"Error generating {graph_type} for {subset_file}:\n{str(e)}"
                        )
                        continue

        QMessageBox.information(self, "Success", "Graphs generated successfully!")

    def get_results(self) -> Tuple[Dict[str, bool], List[str], float, float, float]:
        """
        Gibt nach dem GUI-Lauf die ausgewählten Optionen zurück:
         - das Dictionary mit den Processing-Schritten,
         - die Liste der Subset-Pfade,
         - die Mindestdistanz,
         - den unteren Prozentwert,
         - den oberen Prozentwert.
        """
        return (self.selected_steps, self.selected_subsets,
                self.min_distance, self.delete_lower_percentage, self.delete_upper_percentage)


def main(config: Dict[str, Any], subsets: List[str]) -> None:
    """
    Main Processing Pipeline.
    Geht alle ausgewählten Subsets durch, lädt sie und führt die angehakten
    Verarbeitungsschritte durch. Abschließend wird gespeichert.
    """
    for subset_path in subsets:
        try:
            if not os.path.exists(subset_path):
                raise FileNotFoundError(f"Subset not found: {subset_path}")

            # Lade das CSV
            df = csv_load(subset_path, config)
            if df.empty:
                print(f"Skipping empty subset: {subset_path}")
                continue

            # Mapping der möglichen Schritte (Name im config -> (Funktion, suffix))
            processing_steps = [
                #   ("remove_the_outliers", data_remove_gps_outliers, "outliers"),
                ("parse_time", data_parse_time_and_compute_dt, "time"),
                ("filter_GPS_with_rolling_windows", data_rolling_windows_gps_data, "rollingW"),
                #   ("smooth_gps_data_savitzky",  data_smooth_gps_savitzky,   "savitzky"),
                # ("smooth_gps_data_gaussian",  data_smooth_gps_gaussian,   "gaussian"),
                #  ("smooth_gps_particule_filter", data_particle_filter, "particule"),
                ("convert_to_planar", data_convert_to_planar, "planar"),
                #  ( "rolling_windows_+_kalman_on GPS", data_rollWin_Kalman_on_gps, "rollKal"),
                #  ("filter_with_distances",     data_filter_points_by_distance, "distFilt"),
                ("compute_traveled_distance", data_compute_traveled_distance, "distance"),
                ("compute_heading_with_dy/dx", data_compute_heading_dx_dy, "headingDX"),
                ("compute_heading_with_dx/ds", data_compute_heading_ds, "headingDS"),
                ("compute_yaw_rate_from_heading", data_compute_yaw_rate_from_heading, "yawRate"),
                #  ("use_kalman_on_yaw_rate",    data_kalman_on_yaw_rate, "kalman"),
                ("delete_the_boundaries", data_delete_the_one_percent, "delBoundaries"),
                ("compute_curvature_and_radius", data_compute_curvature_radius_and_detect_steady_curves, "radius"),
                #   ("infrastructure_identifier", data_add_infrastructure_status, "infra"),
                ("get_elevation", data_get_elevation, "elevation"),
                ("get_gradient", data_compute_gradient, "gradient"),
                #   ("smoothed_gradient",         data_smooth_gradient, "smooothGradient"),

            ]

            processed_suffixes = []
            # Nacheinander die aktivierten Schritte ausführen
            for step_name, step_func, suffix in processing_steps:
                if config.get(step_name, False):
                    df = step_func(df, config)  # ggf. config hier übergeben
                    processed_suffixes.append(suffix)

            # Datei speichern, falls gewünscht
            if config.get("save_to_csv", False):
                base_name = os.path.splitext(os.path.basename(subset_path))[0]
                # z. B. "2020-01-01.csv" -> "2020-01-01_savitzky_gaussian.csv"
                # Suffixe, die die aktivierten Steps widerspiegeln
                new_name = f"{base_name}_{'_'.join(processed_suffixes)}.csv"

                # In dasselbe Verzeichnis wie subset_path ablegen
                save_dir = os.path.dirname(subset_path)
                save_path = os.path.join(save_dir, new_name)

                # csv_save ruft, wenn enable_statistics_on_save=True, intern csv_get_statistics() auf
                csv_save(df, save_path, config)

            # Falls wir hier oder an anderer Stelle *zusätzlich* `statistics` wollen,
            # können wir das wahlweise direkt auf dem Original- oder dem neu erzeugten File machen.
            if config.get("statistics", False):
                final_file = save_path if config.get("save_to_csv", False) else subset_path
                csv_get_statistics(final_file, config)

            # Karte generieren?
            if config.get("generate_map", False):
                final_file = save_path if config.get("save_to_csv", False) else subset_path
                generate_map_from_csv(final_file)

        except Exception as e:
            print(f"Error processing {subset_path}: {str(e)}")
            continue


if __name__ == "__main__":
    # Voreinstellungen
    DEFAULT_CONFIG = {
        "statistics": False,
        # "remove_the_outliers": True,
        "parse_time": True,
        "filter_GPS_with_rolling_windows": True,
        #  "smooth_gps_data_savitzky": True,
        #  "smooth_gps_data_gaussian": True,
        #  "smooth_gps_particule_filter": True,
        "convert_to_planar": True,
        #  "rolling_windows_+_kalman_on GPS" : True,
        #  "filter_with_distances": True,
        "compute_traveled_distance": True,
        "compute_heading_with_dy/dx": True,
        "compute_heading_with_dx/ds": True,
        "compute_yaw_rate_from_heading": True,
        "compute_curvature_and_radius": True,
        #  "use_kalman_on_yaw_rate": True,
        "delete_the_boundaries": True,
        # "infrastructure_identifier" : True,
        "get_elevation": True,
        "get_gradient": True,
        "smoothed_gradient": True,
        "save_to_csv": True,
        "enable_statistics_on_save": True,  # bedeutet: csv_save ruft csv_get_statistics automatisch auf
        "generate_map": False,
    }

    app = QApplication(sys.argv)
    window = DataProcessingApp(DEFAULT_CONFIG, "subsets_by_date")
    window.show()
    app.exec_()

    # Hole die vom Nutzer getroffene Auswahl
    (selected_steps, selected_subsets, min_distance,
     delete_lower_percentage, delete_upper_percentage) = window.get_results()

    # Baue das finale Config-Dict
    CONFIG = {
        # Mögliche globale Einstellungen

        "elapsed_time": "elapsed_time_s",
        "dt": "delta_time_s",

        "output_folder_for_subsets_by_date": "subsets_by_date",
        "column_name": "DatumZeit",  # Wenn du eine Spalte fürs Gruppieren benötigst
        "encoding": "utf-8",
        "date_column": "DatumZeit",
        "speed_column": "Geschwindigkeit in m/s",
        "acc_col_for_particule_filter": "Beschleunigung in m/s2",
        "lat_col": "GPS_lat",
        "lon_col": "GPS_lon",

        'mid_speed_threshold_rolling_windows': 50.0,
        "time_rolling_window_fast": 1.0,
        "x_col": "x",
        "y_col": "y",
        "lat_col_smooth": "GPS_lat_smooth",
        "lon_col_smooth": "GPS_lon_smooth",
        "distance_col": "distance",
        "time_between_points": "dt",
        "heading_col_for_yaw_rate_function": "heading_deg_ds",
        "yaw_col_for_kalman": "yaw_rate_deg_s",
        "N_for_particule_filter": 1000,
        "threshold_for_outliers_removing": 0.005,
        "min_distance": min_distance,
        "delete_lower_bound_percentage": delete_lower_percentage,
        "delete_upper_bound_percentage": delete_upper_percentage,

        # for the rolling window
        "speed_threshold_stopped_rolling_windows": 0.5,
        "distance_window_meters": 20,
        "time_window_min": 1.0,
        "time_window_max": 9999.0,
        "max_stop_window": 999999999.0,
        "speed_bins": [0.0, 0.5, 2.0, 5.0, 15.0, 30.0, float("inf")],

        # for curvature
        "yaw": "heading_deg_ds",
        "yaw_rate": "yaw_rate_deg_s",
        "speed": "Geschwindigkeit in m/s",

        # for the remove outliers fonction
        "speed_threshold_outliers": 2,
        "dbscan_eps": 10,
        "min_samples": 3,

        # for the bridges and tunnels
        "overpass_url": "https://overpass-api.de/api/interpreter",
        "bbox": [47.2, 5.9, 55.1, 15.0],
        "structure_threshold": 0.01,  # km: adjust based on your GPS quality
        "bridge_file": "bridges.csv",
        "tunnel_file": "tunnels.csv",
        "gps_quality_col": "GPS Qualität",

        # === Configuration Variables for elevation===
        "api_key": 'AIzaSyAb7ec8EGcU5MiHlQ9jJETvABNYNyiq-WE',  # replace with your google elevation api key
        "api_url": 'https://maps.googleapis.com/maps/api/elevation/json',
        "batch_size": 280,  # google allows up to 512 per request, but we use 100 for safety
        "threads": 10,  # number of parallel api requests

        # for the gradient
        "elevation_column": "elevation",
        "horizontal_distance_column": "cumulative_distance",

        # for smoothing the gradient
        "gradient_promille_column": "gradient_promille",
        "smoothing_windows": 50000,

        # for heading
        "cumulative_distance": "cumulative_distance",

        # for Rollwin + kalman
        "speed_move": 0.8,
        "speed_stop": 0.5,
        'time_window': 3,
        'time_step': 1.0,
        'process_noise': 0.01,
        'measurement_noise': 3.0,
        'move_duration': 3,
        'stop_duration': 5,
        'time_column': "DatumZeit",

        # for the heading with ds :
        "x_col_heading_ds": "filtered_x",
        "y_col_heading_ds": "filtered_y",

        **selected_steps
    }

    # Starte die Verarbeitung
    main(CONFIG, selected_subsets)