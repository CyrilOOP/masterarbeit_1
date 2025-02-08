import sys
import os
from typing import Dict, List, Tuple, Any

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QListWidget, QLineEdit, QPushButton, QDoubleSpinBox, QMessageBox, QFileDialog,
    QCheckBox, QGridLayout, QListWidgetItem, QFormLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon

# Importiere deine bestehenden CSV-/Daten-Funktionen
from csv_tools import (
    csv_load, csv_save, csv_get_statistics, csv_group_by_date_and_save, subsets_by_date
)
from data_tools import (
    data_smooth_gps_savitzky,
    data_smooth_gps_gaussian,
    data_convert_to_planar,
    data_filter_points_by_distance,
    parse_time_and_compute_dt,
    data_compute_heading_from_xy,
    data_compute_yaw_rate_from_heading,
    data_delete_the_one_percent,
    data_compute_heading_from_ds,
    data_kalman_on_yaw_rate, data_particle_filter,
    data_remove_gps_outliers, data_rolling_windows_gps_data,
)
from map_generator import generate_map_from_csv


class DataProcessingApp(QMainWindow):
    """
    Das PyQt-basierte GUI-Fenster, das:
     - Die zu verarbeitenden Teildateien (Subsets) anzeigt und filtern lässt
     - Processing-Schritte auswählbar macht
     - Min. Abstandswert (min_distance) und nun auch zwei Prozentwerte (untere und obere Grenze) erfassen lässt
    """
    def __init__(self, default_config: Dict[str, bool], subset_folder: str, pre_selected_date: str = None):
        super().__init__()
        self.default_config = default_config
        self.subset_folder = subset_folder
        self.pre_selected_date = pre_selected_date

        # Hier "zwischenspeichern", was im GUI ausgewählt wird
        self.selected_steps: Dict[str, bool] = {}
        self.selected_subsets: List[str] = []
        self.min_distance: float = 1.0

        # Neue Attribute für die Prozentangaben
        self.delete_lower_percentage: float = 1.0   # Default: 1%
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

        # Baue die GUI-Komponenten auf
        main_layout.addWidget(self.create_file_selection_group())
        main_layout.addWidget(self.create_processing_steps_group())
        main_layout.addWidget(self.create_distance_and_percentage_input_group())
        main_layout.addWidget(self.create_subsets_group())
        main_layout.addWidget(self.create_action_button())  # Start Processing

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
        """Erzeugt den Bereich, in dem man über PyQt eine CSV-Datei auswählen kann,
           um sie nach Datum zu unterteilen (csv_group_by_date_and_save)."""
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
        # Wir filtern hier nur die Steps, die NICHT "create_subsets_by_date" heißen
        steps = [(k, v) for k, v in self.default_config.items()
                 if k != "create_subsets_by_date"]

        # Grid: 2 Spalten, so viele Zeilen wie nötig
        num_steps = len(steps)
        num_columns = 2
        num_rows = (num_steps + num_columns - 1) // num_columns  # Ceiling division

        for idx, (step_name, enabled) in enumerate(steps):
            checkbox = QCheckBox(step_name.replace("_", " ").title())
            checkbox.setChecked(enabled)
            self.checkboxes[step_name] = checkbox

            # Reihen und Spalten berechnen
            row = idx % num_rows + 1  # +1 wegen der Überschriftszeile
            col = idx // num_rows
            layout.addWidget(checkbox, row, col)

        # "Select All" / "Unselect All" Buttons unter den Checkboxen
        btn_group = QWidget()
        btn_layout = QHBoxLayout(btn_group)
        btn_select_all = QPushButton("Select All")
        btn_select_all.clicked.connect(lambda: self.toggle_all_steps(True))
        btn_unselect_all = QPushButton("Unselect All")
        btn_unselect_all.clicked.connect(lambda: self.toggle_all_steps(False))

        btn_layout.addWidget(btn_select_all)
        btn_layout.addWidget(btn_unselect_all)

        # Span across both columns
        layout.addWidget(btn_group, num_rows + 1, 0, 1, 2)

        return group

    def create_distance_and_percentage_input_group(self) -> QWidget:
        """
        Creates an input area with a QFormLayout where each label is paired with its input field.
        The QDoubleSpinBoxes are constrained in width to prevent them from stretching too wide.
        This version uses custom styles for the labels to override the global QLabel stylesheet.
        """
        group = QWidget()
        form_layout = QFormLayout(group)
        form_layout.setSpacing(8)  # Adjust spacing between rows as needed

        # Define a custom style for our labels that we want to be smaller and not uppercase or bold.
        # Note: Qt's QSS may not support text-transform. If that's the case, the text might still appear
        # in uppercase because of the global rule. In that case, consider manually setting the text.
        label_style = "font-size: 14px; font-weight: normal; text-transform: none;"

        # Minimum Distance row
        min_label = QLabel("Minimum Distance (meters):")
        min_label.setObjectName("smallLabel")

        # Use an ID selector to override the global styling:
        min_label.setStyleSheet("#smallLabel { " + label_style + " }")
        self.distance_input = QDoubleSpinBox()
        self.distance_input.setRange(0.001, 1000.0)
        self.distance_input.setSingleStep(0.1)
        self.distance_input.setValue(self.min_distance)
        self.distance_input.setMaximumWidth(100)  # Limit the width of the input
        form_layout.addRow(min_label, self.distance_input)

        # Delete Lower Boundary Percentage row
        lower_label = QLabel("Delete Lower Boundary (%):")
        lower_label.setObjectName("smallLabelLower")
        lower_label.setStyleSheet("#smallLabelLower { " + label_style + " }")
        self.lower_percentage_input = QDoubleSpinBox()
        self.lower_percentage_input.setRange(0.0, 100.0)
        self.lower_percentage_input.setSingleStep(0.1)
        self.lower_percentage_input.setValue(1.0)  # Default: 1%
        self.lower_percentage_input.setMaximumWidth(100)  # Limit the width
        form_layout.addRow(lower_label, self.lower_percentage_input)

        # Delete Upper Boundary Percentage row
        upper_label = QLabel("Delete Upper Boundary (%):")
        upper_label.setObjectName("smallLabelUpper")
        upper_label.setStyleSheet("#smallLabelUpper { " + label_style + " }")
        self.upper_percentage_input = QDoubleSpinBox()
        self.upper_percentage_input.setRange(0.0, 100.0)
        self.upper_percentage_input.setSingleStep(0.1)
        self.upper_percentage_input.setValue(99.0)  # Default: 99%
        self.upper_percentage_input.setMaximumWidth(100)  # Limit the width
        form_layout.addRow(upper_label, self.upper_percentage_input)

        return group

    def create_subsets_group(self) -> QWidget:
        """
        Dieser Bereich zeigt die vorhandenen Subset-Dateien
        in der `subsets_by_date`-Struktur an und erlaubt Mehrfachauswahl.
        """
        group = QWidget()
        layout = QVBoxLayout(group)
        layout.addWidget(QLabel("Subsets to Process"))

        # Suchfeld
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search subsets...")
        self.search_bar.textChanged.connect(self.filter_subsets)
        layout.addWidget(self.search_bar)

        # Liste der verfügbaren Dateien
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

    def toggle_all_steps(self, state: bool):
        """Schaltet alle Häkchen an oder aus."""
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(state)

    def refresh_subsets(self):
        """
        Liest die vorhandenen CSV-Dateien in `self.subset_folder`
        (bspw. `subsets_by_date/YYYY-MM-DD/*.csv`) und zeigt sie in der Liste an.
        """
        self.subset_files = subsets_by_date(self.subset_folder)
        self.subset_list.clear()

        for full_path in self.subset_files:
            # Relativen Pfad ermitteln, um den Ordner-Namen in der GUI zu sehen
            rel_path = os.path.relpath(full_path, self.subset_folder)
            item = QListWidgetItem(os.path.basename(rel_path))  # Nur der Dateiname
            item.setData(Qt.UserRole, rel_path)  # Speichere den relativen Pfad
            self.subset_list.addItem(item)

        self.filter_subsets()
        self.select_preselected_date()

    def select_preselected_date(self):
        """
        Falls beim Erstellen dieses Fensters ein `pre_selected_date`
        gegeben wurde, wird diese in der Liste gesucht und direkt markiert.
        """
        if not self.pre_selected_date:
            return

        for i in range(self.subset_list.count()):
            item = self.subset_list.item(i)
            if self.pre_selected_date in item.text():
                item.setSelected(True)
                self.subset_list.scrollToItem(item)
                break

    def filter_subsets(self):
        """Versteckt alle Einträge, die nicht zum Suchstring passen."""
        query = self.search_bar.text().lower()
        self.current_filter = query

        for i in range(self.subset_list.count()):
            item = self.subset_list.item(i)
            item.setHidden(query not in item.text().lower())

    def on_select_file(self):
        """
        Öffnet ein PyQt-FileDialog, um eine CSV zu wählen und per
        `csv_group_by_date_and_save()` in dateibasierte Subsets aufzuteilen.
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
            # Hier übergeben wir den Pfad an csv_load. => Kein zweites Tk-Fenster.
            df = csv_load(path)
            if df.empty:
                raise ValueError("The chosen CSV file is empty.")

            # Gruppiert und speichert die Splits in "subsets_by_date/<YYYY-MM-DD>"
            # Standardmäßig wird der Name "DatumZeit" genutzt (siehe csv_tools.py).
            csv_group_by_date_and_save(df, self.subset_folder)

            # Liste im GUI aktualisieren
            self.refresh_subsets()
            QMessageBox.information(self, "Success", "Subsets created successfully!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process file:\n{str(e)}")

    def on_process(self):
        """
        Liest die eingestellten Optionen aus und schließt das Fenster,
        damit im Hauptteil (unten) weitergemacht werden kann.
        """
        # Lese den min_distance und die neuen Prozent-Werte aus
        self.min_distance = self.distance_input.value()
        self.delete_lower_percentage = self.lower_percentage_input.value()
        self.delete_upper_percentage = self.upper_percentage_input.value()

        selected_items = self.subset_list.selectedItems()

        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select at least one subset!")
            return

        # Welche Steps wurden angehakt?
        self.selected_steps = {
            name: cb.isChecked()
            for name, cb in self.checkboxes.items()
        }

        # Baue die vollständigen Pfade (absolut) aus den relativen Pfaden
        self.selected_subsets = [
            os.path.join(self.subset_folder, item.data(Qt.UserRole))
            for item in selected_items
        ]

        # Fensterschluss
        self.close()

    def get_results(self) -> Tuple[Dict[str, bool], List[str], float, float, float]:
        """
        Gibt nach dem GUI-Lauf:
         - das Dictionary mit den Schritten,
         - die Liste der ausgewählten Subset-Pfade,
         - die eingegebene Mindestdistanz,
         - den unteren Prozentwert,
         - den oberen Prozentwert
        zurück.
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
                ("remove_the_outliers", data_remove_gps_outliers, "outliers"),
                ("filter_GPS_with_rolling_windows", data_rolling_windows_gps_data, "rollingW"),
                ("smooth_gps_data_savitzky",  data_smooth_gps_savitzky,   "savitzky"),
                ("smooth_gps_data_gaussian",  data_smooth_gps_gaussian,   "gaussian"),
                ("smooth_gps_particule_filter", data_particle_filter, "particule"),
                ("convert_to_planar",         data_convert_to_planar,     "planar"),
                ("filter_with_distances",     data_filter_points_by_distance, "dist"),
                ("parse_time",                parse_time_and_compute_dt,   "time"),
                ("compute_heading_from_xy",   data_compute_heading_from_xy,"heading"),
                ("compute_heading_from_ds",   data_compute_heading_from_ds, "headingDS" ),
                ("compute_yaw_rate_from_heading", data_compute_yaw_rate_from_heading, "yawRate"),
                ("use_kalman_on_yaw_rate",    data_kalman_on_yaw_rate, "kalman"),
                ("delete_the_boundaries",    data_delete_the_one_percent, "delBoundaries"),
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
        "remove_the_outliers": True,
        "filter_GPS_with_rolling_windows": True,
        "smooth_gps_data_savitzky": True,
        "smooth_gps_data_gaussian": True,
        "smooth_gps_particule_filter": True,
        "convert_to_planar": True,
        "filter_with_distances": True,
        "parse_time": True,
        "compute_heading_from_xy": True,
        "compute_heading_from_ds": True,
        "compute_yaw_rate_from_heading": True,
        "use_kalman_on_yaw_rate": True,
        "delete_the_boundaries": True,
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

        #for the rolling window
        "speed_threshold_stopped_rolling_windows": 0.5,
        "distance_window_meters" : 10,
        "time_window_min": 1.0,
        "time_window_max": 300.0,
        "speed_bins": [0.0, 0.5, 5.0, 15.0, 30.0, float("inf")],



        # for the remove outliers fonction
        "speed_threshold_outliers" : 2,
        "dbscan_eps" : 10,
        "min_samples" : 3,


        **selected_steps
    }

    # Starte die Verarbeitung
    main(CONFIG, selected_subsets)
