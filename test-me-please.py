#!/usr/bin/env python3
"""
This script extracts railway segments that pass under/on bridges and identifies railway tunnels.
It produces a combined output GeoJSON in which each feature gets an "infrastructure" property
("bridge" or "tunnel") and a "bundesland" property indicating the region.

For bridges:
  - Extract ways with `bridge=yes` and `railway=*` from an OSM PBF.
  - Buffer bridges (5m) and spatially join with railways.
  - Compute the local orientation using spline interpolation.
  - Aggregate intersections per bridge – if any intersection is "under," mark the bridge as "under."
  - Export bridges (with railway under) as GeoJSON.

For tunnels:
  - Extract ways with `tunnel=yes` and a `railway` tag (railway tunnels).
  - Buffer tunnels (5m) and spatially join with railways.
  - Aggregate intersections per tunnel and mark them as "tunnel."
  - Export tunnels as GeoJSON.

For multiple regions, the script processes each region (using its own PBF URL and file),
and then combines the outputs into one final combined GeoJSON file.
"""

import os
import time
import math
import requests
import logging
import osmium
import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from shapely.geometry import LineString, Point
from tqdm import tqdm
import threading

# --- PyQt5 Imports ---
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QListWidget, QPushButton, QTextEdit, QLabel, QMessageBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QMetaObject, Q_ARG

# Set up logging: remove default handlers so our custom handler is used.
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = []

############################################
# Default Parameters and URL Helpers
############################################

DEFAULT_PBF_URL_ALL = "https://download.geofabrik.de/europe/germany-latest.osm.pbf"
DEFAULT_PBF_FILE_ALL = "germany-latest.osm.pbf"


def get_bl_url(bl):
    return f"https://download.geofabrik.de/europe/germany/{bl}-latest.osm.pbf"


def get_bl_file(bl):
    return f"{bl}-latest.osm.pbf"


ANGLE_TOLERANCE = 10  # in degrees


############################################
# OSM Data Handler (No Cache)
############################################

class OSMHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.bridges = []
        self.railways = []
        self.tunnels = []  # Only railway tunnels
        self.way_count = 0

    def way(self, w):
        self.way_count += 1
        if self.way_count % 1_000_000 == 0:
            logging.info(f"Processed {self.way_count} ways...")
        if "bridge" in w.tags and w.tags.get("bridge") == "yes":
            coords = [(n.lon, n.lat) for n in w.nodes if n.location.valid()]
            if len(coords) > 1:
                try:
                    layer = int(w.tags.get("layer", "0"))
                except ValueError:
                    layer = 0
                self.bridges.append({"geometry": LineString(coords), "layer": layer})
        if "railway" in w.tags:
            coords = [(n.lon, n.lat) for n in w.nodes if n.location.valid()]
            if len(coords) > 1:
                try:
                    layer = int(w.tags.get("layer", "0"))
                except ValueError:
                    layer = 0
                self.railways.append({"geometry": LineString(coords), "layer": layer})
        if "tunnel" in w.tags and w.tags.get("tunnel") == "yes" and "railway" in w.tags:
            coords = [(n.lon, n.lat) for n in w.nodes if n.location.valid()]
            if len(coords) > 1:
                try:
                    layer = int(w.tags.get("layer", "0"))
                except ValueError:
                    layer = 0
                self.tunnels.append({"geometry": LineString(coords), "layer": layer})


############################################
# Download PBF File (No Cache)
############################################

def download_pbf(pbf_url, pbf_file):
    if os.path.exists(pbf_file):
        logging.info(f"File '{pbf_file}' already exists. Skipping download.")
        return
    logging.info(f"Downloading '{pbf_file}' from {pbf_url} ...")
    try:
        r = requests.get(pbf_url, stream=True)
        r.raise_for_status()
    except requests.RequestException as e:
        logging.error(f"Failed to download PBF file: {e}")
        raise
    total_size = int(r.headers.get("content-length", 0))
    chunk_size = 1024 * 1024
    with open(pbf_file, "wb") as f, tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024,
                                         desc="Downloading PBF") as bar:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
    logging.info(f"Download complete: {pbf_file}")


############################################
# Orientation Calculation Functions
############################################

def compute_local_orientation_spline(line, pt, delta_m=10, n_points=7):
    gdf_line = gpd.GeoDataFrame({'geometry': [line]}, crs="EPSG:4326")
    gdf_pt = gpd.GeoDataFrame({'geometry': [pt]}, crs="EPSG:4326")
    line_3857 = gdf_line.to_crs(epsg=3857).iloc[0].geometry
    pt_3857 = gdf_pt.to_crs(epsg=3857).iloc[0].geometry
    proj_dist = line_3857.project(pt_3857)
    start_dist = max(proj_dist - delta_m, 0)
    end_dist = min(proj_dist + delta_m, line_3857.length)
    if end_dist - start_dist < 1e-6:
        return compute_local_orientation(line, pt, delta_m)
    distances = np.linspace(start_dist, end_dist, n_points)
    x_vals = [line_3857.interpolate(d).x for d in distances]
    y_vals = [line_3857.interpolate(d).y for d in distances]
    spline_x = UnivariateSpline(distances, np.array(x_vals), k=3, s=0)
    spline_y = UnivariateSpline(distances, np.array(y_vals), k=3, s=0)
    central_d = np.clip(proj_dist, distances[0], distances[-1])
    dx = spline_x.derivative()(central_d)
    dy = spline_y.derivative()(central_d)
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 180
    elif angle_deg >= 180:
        angle_deg -= 180
    return angle_deg


def compute_local_orientation(line, pt, delta_m=10):
    gdf_line = gpd.GeoDataFrame({'geometry': [line]}, crs="EPSG:4326")
    gdf_pt = gpd.GeoDataFrame({'geometry': [pt]}, crs="EPSG:4326")
    line_3857 = gdf_line.to_crs(epsg=3857).iloc[0].geometry
    pt_3857 = gdf_pt.to_crs(epsg=3857).iloc[0].geometry
    proj_dist = line_3857.project(pt_3857)
    start_dist = max(proj_dist - delta_m, 0)
    end_dist = min(proj_dist + delta_m, line_3857.length)
    start_point = line_3857.interpolate(start_dist)
    end_point = line_3857.interpolate(end_dist)
    dx = end_point.x - start_point.x
    dy = end_point.y - start_point.y
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 180
    elif angle_deg >= 180:
        angle_deg -= 180
    return angle_deg


def get_representative_point(intersection_geom):
    if intersection_geom.is_empty:
        return None
    geom_type = intersection_geom.geom_type
    if geom_type == 'Point':
        return intersection_geom
    elif geom_type == 'LineString':
        return intersection_geom.interpolate(0.5, normalized=True)
    elif geom_type == 'MultiPoint':
        return intersection_geom.centroid
    elif geom_type == 'MultiLineString':
        from shapely.ops import linemerge
        merged = linemerge(intersection_geom)
        if merged.geom_type == 'LineString':
            return merged.interpolate(0.5, normalized=True)
        else:
            return list(intersection_geom.geoms)[0].interpolate(0.5, normalized=True)
    elif geom_type == 'GeometryCollection':
        for geom in intersection_geom.geoms:
            if not geom.is_empty:
                if geom.geom_type == 'Point':
                    return geom
                elif geom.geom_type == 'LineString':
                    return geom.interpolate(0.5, normalized=True)
        return intersection_geom.centroid
    else:
        return intersection_geom.centroid


def classify_by_parallelism(row):
    bridge_line = row["bridge_geom"]
    railway_line = row["railway_geom"]
    intersection = bridge_line.intersection(railway_line)
    rep_pt = get_representative_point(intersection)
    if rep_pt is None or not isinstance(rep_pt, Point):
        return "unknown"
    try:
        angle_bridge = compute_local_orientation_spline(bridge_line, rep_pt)
        angle_rail = compute_local_orientation_spline(railway_line, rep_pt)
    except Exception as e:
        logging.error(f"Error computing orientation: {e}")
        return "unknown"
    diff = abs(angle_bridge - angle_rail)
    diff = min(diff, 180 - diff)
    return "on" if diff <= ANGLE_TOLERANCE else "under"


############################################
# Processing Functions (No Cache)
############################################

def process_data(pbf_url, pbf_file, output_prefix):
    download_pbf(pbf_url, pbf_file)
    logging.info("Parsing PBF with Osmium...")
    start = time.time()
    handler = OSMHandler()
    handler.apply_file(pbf_file, locations=True)
    elapsed = time.time() - start
    logging.info(f"Processed {handler.way_count} ways in {elapsed:.2f} seconds.")
    bridges_list = handler.bridges
    railway_list = handler.railways
    logging.info("Building GeoDataFrames for bridges and railways...")
    bridges_gdf = gpd.GeoDataFrame(bridges_list, geometry="geometry", crs="EPSG:4326")
    railways_gdf = gpd.GeoDataFrame(railway_list, geometry="geometry", crs="EPSG:4326")
    logging.info(f"Bridges: {len(bridges_gdf)}, Railways: {len(railways_gdf)}")
    bridges_gdf["orig_geom"] = bridges_gdf.geometry.copy()
    railways_gdf["orig_geom"] = railways_gdf.geometry.copy()
    logging.info("Buffering bridges by 5m for spatial join...")
    bridges_3857 = bridges_gdf.to_crs(epsg=3857)
    bridges_3857["geometry"] = bridges_3857.geometry.buffer(5)
    bridges_buffered = bridges_3857.to_crs(epsg=4326)
    logging.info("Spatial join: Bridges <-> Railways...")
    joined = gpd.sjoin(bridges_buffered, railways_gdf, how="inner", predicate="intersects")
    logging.info(f"Found {len(joined)} bridge-railway intersections.")
    joined.reset_index(inplace=True)
    joined["bridge_geom"] = joined["index"].apply(lambda i: bridges_gdf.loc[i, "orig_geom"])
    joined["railway_geom"] = joined["index_right"].apply(lambda i: railways_gdf.loc[i, "orig_geom"])
    logging.info("Classifying intersections using spline-smoothed local orientation...")
    joined["position"] = joined.apply(classify_by_parallelism, axis=1)

    def aggregate_bridge_classification(group):
        overall_position = "under" if "under" in group["position"].values else "on"
        rep_bridge_geom = group["bridge_geom"].iloc[0]
        return pd.Series({"position": overall_position, "bridge_geom": rep_bridge_geom})

    aggregated = joined.groupby("index", group_keys=False).apply(aggregate_bridge_classification).reset_index()
    logging.info(f"Aggregated bridges: {len(aggregated)}")
    bridges_with_railway_under = aggregated[aggregated["position"] == "under"].reset_index(drop=True)
    bridges_with_railway_under = bridges_with_railway_under.set_geometry("bridge_geom").set_crs("EPSG:4326",
                                                                                                allow_override=True)
    from shapely.geometry.base import BaseGeometry
    for col in list(bridges_with_railway_under.columns):
        if col != "bridge_geom" and not bridges_with_railway_under[col].empty:
            sample = bridges_with_railway_under[col].dropna().iloc[0]
            if isinstance(sample, BaseGeometry):
                bridges_with_railway_under.drop(columns=[col], inplace=True)
    out_file = f"{output_prefix}_bridges_with_railway_under.geojson"
    bridges_with_railway_under.to_file(out_file, driver="GeoJSON")
    logging.info(f"Saved bridges with railway under to '{out_file}'.")


def process_tunnels(pbf_url, pbf_file, output_prefix):
    download_pbf(pbf_url, pbf_file)
    logging.info("Parsing PBF with Osmium for tunnels...")
    start = time.time()
    handler = OSMHandler()
    handler.apply_file(pbf_file, locations=True)
    elapsed = time.time() - start
    logging.info(f"Processed {handler.way_count} ways in {elapsed:.2f} seconds.")
    tunnels_list = handler.tunnels
    railway_list = handler.railways
    if not tunnels_list:
        logging.info("No railway tunnels found in data.")
        return
    tunnels_gdf = gpd.GeoDataFrame(tunnels_list, geometry="geometry", crs="EPSG:4326")
    railways_gdf = gpd.GeoDataFrame(railway_list, geometry="geometry", crs="EPSG:4326")
    logging.info(f"Tunnels: {len(tunnels_gdf)}, Railways: {len(railways_gdf)}")
    tunnels_gdf["orig_geom"] = tunnels_gdf.geometry.copy()
    logging.info("Buffering tunnels by 5m for spatial join...")
    tunnels_3857 = tunnels_gdf.to_crs(epsg=3857)
    tunnels_3857["geometry"] = tunnels_3857.geometry.buffer(5)
    tunnels_buffered = tunnels_3857.to_crs(epsg=4326)
    logging.info("Spatial join: Tunnels <-> Railways...")
    joined = gpd.sjoin(tunnels_buffered, railways_gdf, how="inner", predicate="intersects")
    logging.info(f"Found {len(joined)} tunnel-railway intersections.")
    joined.reset_index(inplace=True)
    joined["tunnel_geom"] = joined["index"].apply(lambda i: tunnels_gdf.loc[i, "orig_geom"])
    joined["position"] = "tunnel"

    def aggregate_tunnel_classification(group):
        rep_tunnel_geom = group["tunnel_geom"].iloc[0]
        return pd.Series({"position": "tunnel", "tunnel_geom": rep_tunnel_geom})

    aggregated = joined.groupby("index", group_keys=False).apply(aggregate_tunnel_classification).reset_index()
    logging.info(f"Aggregated tunnels: {len(aggregated)}")
    tunnels_in_railway = aggregated.copy().set_geometry("tunnel_geom").set_crs("EPSG:4326", allow_override=True)
    from shapely.geometry.base import BaseGeometry
    for col in list(tunnels_in_railway.columns):
        if col != "tunnel_geom" and not tunnels_in_railway[col].empty:
            sample = tunnels_in_railway[col].dropna().iloc[0]
            if isinstance(sample, BaseGeometry):
                tunnels_in_railway.drop(columns=[col], inplace=True)
    out_file = f"{output_prefix}_railways_in_tunnel.geojson"
    tunnels_in_railway.to_file(out_file, driver="GeoJSON")
    logging.info(f"Saved railways in tunnel to '{out_file}'.")


def process_combined(combined_files, output_prefix):
    combined_list = []
    for bridges_file, tunnels_file, bl in combined_files:
        try:
            bridges = gpd.read_file(bridges_file)
        except Exception:
            bridges = gpd.GeoDataFrame()
        try:
            tunnels = gpd.read_file(tunnels_file)
        except Exception:
            tunnels = gpd.GeoDataFrame()
        if not bridges.empty:
            bridges["infrastructure"] = "bridge"
            bridges["bundesland"] = bl
        if not tunnels.empty:
            tunnels["infrastructure"] = "tunnel"
            tunnels["bundesland"] = bl
        combined_list.append(bridges)
        combined_list.append(tunnels)
    if combined_list:
        combined = pd.concat(combined_list, ignore_index=True)
        out_file = f"{output_prefix}_combined_infrastructure.geojson"
        combined.to_file(out_file, driver="GeoJSON")
        logging.info(f"Saved combined infrastructure to '{out_file}'.")


############################################
# Worker Thread for Processing (PyQt)
############################################

class Worker(QThread):
    finished = pyqtSignal(list)

    def __init__(self, selections, mapping):
        super().__init__()
        self.selections = selections
        self.mapping = mapping

    def run(self):
        combined_outputs = []
        if "all" in [self.mapping[s] for s in self.selections]:
            pbf_url = DEFAULT_PBF_URL_ALL
            pbf_file = DEFAULT_PBF_FILE_ALL
            output_prefix = "germany"
            process_data(pbf_url, pbf_file, output_prefix)
            process_tunnels(pbf_url, pbf_file, output_prefix)
            combined_outputs.append((f"{output_prefix}_bridges_with_railway_under.geojson",
                                     f"{output_prefix}_railways_in_tunnel.geojson",
                                     "germany"))
        else:
            for s in self.selections:
                bl = self.mapping[s]
                pbf_url = get_bl_url(bl)
                pbf_file = get_bl_file(bl)
                output_prefix = bl
                logging.info(f"Processing Bundesland: {bl}")
                process_data(pbf_url, pbf_file, output_prefix)
                process_tunnels(pbf_url, pbf_file, output_prefix)
                combined_outputs.append((f"{output_prefix}_bridges_with_railway_under.geojson",
                                         f"{output_prefix}_railways_in_tunnel.geojson",
                                         bl))
            combined_prefix = "_".join([self.mapping[s] for s in self.selections])
            process_combined(combined_outputs, combined_prefix)
        self.finished.emit(combined_outputs)


############################################
# PyQt5 GUI Implementation
############################################

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bundesländer Infrastructure Processor")
        self.resize(600, 600)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout()
        central.setLayout(layout)

        title = QLabel("Select Bundesländer to Process")
        title.setStyleSheet("font-size: 16pt; font-weight: bold;")
        layout.addWidget(title)

        self.listWidget = QListWidget()
        self.listWidget.setSelectionMode(QListWidget.MultiSelection)
        regions = [
            "All (Germany)",
            "Baden-Württemberg",
            "Bayern",
            "Berlin",
            "Brandenburg",
            "Bremen",
            "Hamburg",
            "Hessen",
            "Mecklenburg-Vorpommern",
            "Niedersachsen",
            "Nordrhein-Westfalen",
            "Rheinland-Pfalz",
            "Saarland",
            "Sachsen",
            "Sachsen-Anhalt",
            "Schleswig-Holstein",
            "Thüringen"
        ]
        self.listWidget.addItems(regions)
        layout.addWidget(self.listWidget)

        btnLayout = QHBoxLayout()
        self.btnProcess = QPushButton("Process")
        self.btnProcess.clicked.connect(self.start_processing)
        btnLayout.addWidget(self.btnProcess)
        btnQuit = QPushButton("Quit")
        btnQuit.clicked.connect(self.close)
        btnLayout.addWidget(btnQuit)
        layout.addLayout(btnLayout)

        self.statusLabel = QLabel("Status: Ready")
        layout.addWidget(self.statusLabel)

        logLabel = QLabel("Log Output:")
        logLabel.setStyleSheet("font-weight: bold;")
        layout.addWidget(logLabel)

        self.logText = QTextEdit()
        self.logText.setReadOnly(True)
        layout.addWidget(self.logText)

        self.mapping = {
            "All (Germany)": "all",
            "Baden-Württemberg": "baden-wuerttemberg",
            "Bayern": "bayern",
            "Berlin": "berlin",
            "Brandenburg": "brandenburg",
            "Bremen": "bremen",
            "Hamburg": "hamburg",
            "Hessen": "hessen",
            "Mecklenburg-Vorpommern": "mecklenburg-vorpommern",
            "Niedersachsen": "niedersachsen",
            "Nordrhein-Westfalen": "nordrhein-westfalen",
            "Rheinland-Pfalz": "rheinland-pfalz",
            "Saarland": "saarland",
            "Sachsen": "sachsen",
            "Sachsen-Anhalt": "sachsen-anhalt",
            "Schleswig-Holstein": "schleswig-holstein",
            "Thüringen": "thueringen"
        }

        self.log_handler = QtHandler(self.logText)
        self.log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.handlers = []
        logger.addHandler(self.log_handler)

    def start_processing(self):
        selected = self.listWidget.selectedItems()
        if not selected:
            QMessageBox.critical(self, "Error", "Please select at least one region.")
            return
        self.btnProcess.setEnabled(False)
        self.statusLabel.setText("Status: Processing...")
        regions = [item.text() for item in selected]
        self.worker = Worker(regions, self.mapping)
        self.worker.finished.connect(self.processing_finished)
        self.worker.start()

    def processing_finished(self, combined_outputs):
        self.statusLabel.setText("Status: Processing complete!")
        self.btnProcess.setEnabled(True)
        QMessageBox.information(self, "Done",
                                "Processing complete! Check the output GeoJSON files in the working directory.")


############################################
# Custom Logging Handler for PyQt
############################################

class QtHandler(logging.Handler):
    """A logging handler that writes log messages to a QTextEdit widget in a thread-safe manner."""

    def __init__(self, text_edit):
        super().__init__()
        self.text_edit = text_edit

    def emit(self, record):
        msg = self.format(record)
        QMetaObject.invokeMethod(self.text_edit, "append", Qt.QueuedConnection, Q_ARG(str, msg))


############################################
# Main Block
############################################

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QTextEdit

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
