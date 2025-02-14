#!/usr/bin/env python3
"""
Extract railway segments that pass under/on bridges and identify railway tunnels.
Also, produce a combined output where bridges and tunnels are distinguished
by an "infrastructure" property.

Logic for bridges:
1. Extract all ways with `bridge=yes` and all ways with `railway=*` from OSM PBF.
2. Spatially join bridges (buffered by 10m) and railways.
3. For each intersection, retrieve the original (unbuffered) geometries.
4. Compute the local orientation (angle) at the intersection point for both geometries using spline interpolation.
5. If the angles are nearly parallel (within a tolerance), the railway is ON the bridge;
   otherwise, it is going UNDER the bridge.
6. Aggregate intersections per bridge – if any intersection shows "under", mark the bridge as "under".

Logic for tunnels:
1. Extract all ways that are both `tunnel=yes` and have a `railway` tag (i.e. railway tunnels).
2. Spatially join these railway tunnels (buffered by 5m) with railways.
3. Aggregate intersections per tunnel and mark them as "tunnel".
4. Export the result.

Finally, produce a combined output that includes both bridges (with railway under) and tunnels,
adding an "infrastructure" property to distinguish them.
"""

import os
import time
import pickle
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

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Constants
PBF_URL = "https://download.geofabrik.de/europe/germany/sachsen-anhalt-latest.osm.pbf"
PBF_FILE = "sachsen-anhalt-latest.osm.pbf"
CACHE_FILE = "bridges_railways.pkl"
ANGLE_TOLERANCE = 10  # Tolerance in degrees


class OSMHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.bridges = []
        self.railways = []
        self.tunnels = []  # Only railway tunnels will be collected
        self.way_count = 0

    def way(self, w):
        self.way_count += 1
        if self.way_count % 1_000_000 == 0:
            logging.info(f"Processed {self.way_count} ways...")

        # Collect bridges
        if "bridge" in w.tags and w.tags.get("bridge") == "yes":
            coords = [(n.lon, n.lat) for n in w.nodes if n.location.valid()]
            if len(coords) > 1:
                try:
                    layer = int(w.tags.get("layer", "0"))
                except ValueError:
                    layer = 0
                self.bridges.append({
                    "geometry": LineString(coords),
                    "layer": layer
                })

        # Collect railways
        if "railway" in w.tags:
            coords = [(n.lon, n.lat) for n in w.nodes if n.location.valid()]
            if len(coords) > 1:
                try:
                    layer = int(w.tags.get("layer", "0"))
                except ValueError:
                    layer = 0
                self.railways.append({
                    "geometry": LineString(coords),
                    "layer": layer
                })

        # Collect only railway tunnels
        if "tunnel" in w.tags and w.tags.get("tunnel") == "yes" and "railway" in w.tags:
            coords = [(n.lon, n.lat) for n in w.nodes if n.location.valid()]
            if len(coords) > 1:
                try:
                    layer = int(w.tags.get("layer", "0"))
                except ValueError:
                    layer = 0
                self.tunnels.append({
                    "geometry": LineString(coords),
                    "layer": layer
                })


def download_pbf():
    """Download the PBF file if not present."""
    if os.path.exists(PBF_FILE):
        logging.info(f"File '{PBF_FILE}' already exists. Skipping download.")
        return

    logging.info(f"Downloading '{PBF_FILE}' from Geofabrik...")
    try:
        r = requests.get(PBF_URL, stream=True)
        r.raise_for_status()
    except requests.RequestException as e:
        logging.error(f"Failed to download PBF file: {e}")
        raise

    total_size = int(r.headers.get("content-length", 0))
    chunk_size = 1024 * 1024

    with open(PBF_FILE, "wb") as f, tqdm(
            total=total_size, unit="B", unit_scale=True, unit_divisor=1024, desc="Downloading PBF"
    ) as bar:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
    logging.info(f"Download complete: {PBF_FILE}")


def compute_local_orientation_spline(line, pt, delta_m=10, n_points=7):
    """
    Compute the local orientation (in degrees) of a LineString at a point using spline interpolation.
    The line and point are assumed to be in EPSG:4326. For accurate measurements,
    we reproject them to EPSG:3857.

    We sample n_points equally spaced along the segment [proj_dist - delta_m, proj_dist + delta_m],
    fit splines for the x and y coordinates as functions of distance, and compute the derivative at
    the central point.
    """
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
    x_vals = []
    y_vals = []
    for d in distances:
        pt_d = line_3857.interpolate(d)
        x_vals.append(pt_d.x)
        y_vals.append(pt_d.y)
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)

    spline_x = UnivariateSpline(distances, x_vals, k=3, s=0)
    spline_y = UnivariateSpline(distances, y_vals, k=3, s=0)

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
    """
    Fallback: Compute the local orientation using a simple two-point method.
    """
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
    """
    Given an intersection geometry, return a representative point.
    Handles Point, LineString, MultiPoint, MultiLineString, and GeometryCollection.
    """
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
            if geom.is_empty:
                continue
            if geom.geom_type == 'Point':
                return geom
            elif geom.geom_type == 'LineString':
                return geom.interpolate(0.5, normalized=True)
            elif geom.geom_type == 'MultiLineString':
                from shapely.ops import linemerge
                merged = linemerge(geom)
                if merged.geom_type == 'LineString':
                    return merged.interpolate(0.5, normalized=True)
                else:
                    return list(geom.geoms)[0].interpolate(0.5, normalized=True)
        return intersection_geom.centroid
    else:
        return intersection_geom.centroid


def classify_by_parallelism(row):
    """
    Classify the relationship between a bridge and a railway based on their
    local orientations (using spline smoothing). Returns "on" or "under".
    """
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
    if diff <= ANGLE_TOLERANCE:
        return "on"
    else:
        return "under"


def process_data():
    """
    Process bridges and railways:
    - Parse the OSM PBF (or load from cache).
    - Build GeoDataFrames, buffer bridges, perform spatial join with railways.
    - Classify each intersection using the refined spline method.
    - Aggregate intersections per bridge: if any intersection is "under", mark the bridge as "under".
    - Export bridges (with railway under) as GeoJSON.
    """
    if os.path.exists(CACHE_FILE):
        logging.info(f"Loading cached data from '{CACHE_FILE}'...")
        with open(CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
        bridges_list = cache["bridges"]
        railway_list = cache["railways"]
    else:
        logging.info("Parsing PBF with Osmium...")
        start = time.time()
        handler = OSMHandler()
        handler.apply_file(PBF_FILE, locations=True)
        elapsed = time.time() - start
        logging.info(f"Processed {handler.way_count} ways in {elapsed:.2f} seconds.")
        bridges_list = handler.bridges
        railway_list = handler.railways
        with open(CACHE_FILE, "wb") as f:
            pickle.dump({"bridges": bridges_list, "railways": railway_list, "tunnels": handler.tunnels}, f)
        logging.info(f"Data cached in '{CACHE_FILE}'.")

    logging.info("Building GeoDataFrames for bridges and railways...")
    bridges_gdf = gpd.GeoDataFrame(bridges_list, geometry="geometry", crs="EPSG:4326")
    railways_gdf = gpd.GeoDataFrame(railway_list, geometry="geometry", crs="EPSG:4326")
    logging.info(f"Bridges: {len(bridges_gdf)}, Railways: {len(railways_gdf)}")

    # Save original geometries for later use
    bridges_gdf["orig_geom"] = bridges_gdf.geometry.copy()
    railways_gdf["orig_geom"] = railways_gdf.geometry.copy()

    logging.info("Buffering bridges by 10m in EPSG:3857 for spatial join...")
    bridges_3857 = bridges_gdf.to_crs(epsg=3857)
    bridges_3857["geometry"] = bridges_3857.geometry.buffer(10)
    bridges_buffered = bridges_3857.to_crs(epsg=4326)

    logging.info("Spatial join: Bridges <-> Railways...")
    joined = gpd.sjoin(bridges_buffered, railways_gdf, how="inner", predicate="intersects")
    logging.info(f"Found {len(joined)} bridge-railway intersections.")

    joined.reset_index(inplace=True)
    joined["bridge_geom"] = joined["index"].apply(lambda i: bridges_gdf.loc[i, "orig_geom"])
    joined["railway_geom"] = joined["index_right"].apply(lambda i: railways_gdf.loc[i, "orig_geom"])

    logging.info("Classifying intersections based on spline-smoothed local orientation...")
    joined["position"] = joined.apply(classify_by_parallelism, axis=1)

    # --- Aggregate intersections per bridge ---
    def aggregate_bridge_classification(group):
        overall_position = "under" if "under" in group["position"].values else "on"
        rep_bridge_geom = group["bridge_geom"].iloc[0]
        return pd.Series({
            "position": overall_position,
            "bridge_geom": rep_bridge_geom
        })

    aggregated = joined.groupby("index").apply(aggregate_bridge_classification)
    aggregated.reset_index(inplace=True)
    logging.info(f"Aggregated bridges: {len(aggregated)}")

    bridges_with_railway_under = aggregated[aggregated["position"] == "under"].reset_index(drop=True)

    # --- Prepare for export ---
    bridges_with_railway_under = bridges_with_railway_under.set_geometry("bridge_geom")
    bridges_with_railway_under = bridges_with_railway_under.set_crs("EPSG:4326", allow_override=True)
    from shapely.geometry.base import BaseGeometry
    geometry_cols = []
    for col in bridges_with_railway_under.columns:
        non_null = bridges_with_railway_under[col].dropna()
        if not non_null.empty:
            sample = non_null.iloc[0]
            if isinstance(sample, BaseGeometry):
                geometry_cols.append(col)
    logging.info(f"Found geometry columns: {geometry_cols}")
    for col in geometry_cols:
        if col != "bridge_geom":
            bridges_with_railway_under = bridges_with_railway_under.drop(columns=[col])
    output_file = "bridges_with_railway_under.geojson"
    bridges_with_railway_under.to_file(output_file, driver="GeoJSON")
    logging.info(f"Saved bridges with railway under to '{output_file}'.")


def process_tunnels():
    """
    Process tunnels and railways:
    - Load tunnels and railways from cache.
    - Build GeoDataFrames, buffer tunnels by 5m, perform spatial join with railways.
    - Aggregate intersections per tunnel and mark them as "tunnel".
    - Export the tunnels (railway tunnels) as GeoJSON.
    """
    if os.path.exists(CACHE_FILE):
        logging.info(f"Loading cached data from '{CACHE_FILE}' for tunnels...")
        with open(CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
        tunnels_list = cache.get("tunnels", [])
        railway_list = cache["railways"]
    else:
        logging.error("Cache file not found. Please run the script to generate cache first.")
        return

    if not tunnels_list:
        logging.info("No railway tunnels found in data.")
        return

    tunnels_gdf = gpd.GeoDataFrame(tunnels_list, geometry="geometry", crs="EPSG:4326")
    railways_gdf = gpd.GeoDataFrame(railway_list, geometry="geometry", crs="EPSG:4326")
    logging.info(f"Tunnels: {len(tunnels_gdf)}, Railways: {len(railways_gdf)}")

    tunnels_gdf["orig_geom"] = tunnels_gdf.geometry.copy()

    logging.info("Buffering tunnels by 5m in EPSG:3857 for spatial join...")
    tunnels_3857 = tunnels_gdf.to_crs(epsg=3857)
    tunnels_3857["geometry"] = tunnels_3857.geometry.buffer(5)
    tunnels_buffered = tunnels_3857.to_crs(epsg=4326)

    logging.info("Spatial join: Tunnels <-> Railways...")
    joined = gpd.sjoin(tunnels_buffered, railways_gdf, how="inner", predicate="intersects")
    logging.info(f"Found {len(joined)} tunnel-railway intersections.")

    joined.reset_index(inplace=True)
    joined["tunnel_geom"] = joined["index"].apply(lambda i: tunnels_gdf.loc[i, "orig_geom"])
    joined["position"] = "tunnel"  # Mark as tunnel

    def aggregate_tunnel_classification(group):
        rep_tunnel_geom = group["tunnel_geom"].iloc[0]
        return pd.Series({
            "position": "tunnel",
            "tunnel_geom": rep_tunnel_geom
        })

    aggregated = joined.groupby("index").apply(aggregate_tunnel_classification)
    aggregated.reset_index(inplace=True)
    logging.info(f"Aggregated tunnels: {len(aggregated)}")

    tunnels_in_railway = aggregated.copy()
    tunnels_in_railway = tunnels_in_railway.set_geometry("tunnel_geom")
    tunnels_in_railway = tunnels_in_railway.set_crs("EPSG:4326", allow_override=True)
    from shapely.geometry.base import BaseGeometry
    geometry_cols = []
    for col in tunnels_in_railway.columns:
        non_null = tunnels_in_railway[col].dropna()
        if not non_null.empty:
            sample = non_null.iloc[0]
            if isinstance(sample, BaseGeometry):
                geometry_cols.append(col)
    logging.info(f"Found geometry columns in tunnels: {geometry_cols}")
    for col in geometry_cols:
        if col != "tunnel_geom":
            tunnels_in_railway = tunnels_in_railway.drop(columns=[col])
    output_file = "railways_in_tunnel.geojson"
    tunnels_in_railway.to_file(output_file, driver="GeoJSON")
    logging.info(f"Saved railways in tunnel to '{output_file}'.")


def process_combined():
    """
    Combine the bridges (with railway under) and the railway tunnels into a single GeoDataFrame.
    Add a property "infrastructure" to distinguish them, then export as GeoJSON.
    """
    # Load the previously exported GeoJSON files
    bridges = gpd.read_file("bridges_with_railway_under.geojson")
    tunnels = gpd.read_file("railways_in_tunnel.geojson")

    # Add a column to indicate the type
    bridges["infrastructure"] = "bridge"
    tunnels["infrastructure"] = "tunnel"

    # Combine both GeoDataFrames
    combined = pd.concat([bridges, tunnels], ignore_index=True)

    # Export the combined GeoDataFrame as GeoJSON
    output_file = "combined_infrastructure.geojson"
    combined.to_file(output_file, driver="GeoJSON")
    logging.info(f"Saved combined infrastructure to '{output_file}'.")


if __name__ == "__main__":
    download_pbf()
    process_data()
    process_tunnels()
    process_combined()
