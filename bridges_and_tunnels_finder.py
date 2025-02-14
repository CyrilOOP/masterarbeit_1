#!/usr/bin/env python3
"""
Extract railway segments that pass *under* or *above* bridges based on OSM `layer` values.

Logic:
1. Extract all ways with `bridge=yes` and save `layer=*` (default: 0).
2. Extract all ways with `railway=*` and save `layer=*` (default: 0).
3. Spatially intersect bridges and railways.
4. Determine position:
   - If `layer(bridge) > layer(railway)`, then railway is *below*.
   - If `layer(bridge) < layer(railway)`, then railway is *above*.
"""

import os
import time
import pickle
import requests
import logging
import osmium
import geopandas as gpd
import pandas as pd

from shapely.geometry import LineString
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PBF_URL = "https://download.geofabrik.de/europe/gdermany-latest.osm.pbf"
PBF
PBF_FILE = "germany-latest.osm.pbf"
CACHE_FILE = "layer_bridge_rail.pkl"


class OSMHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.bridges = []
        self.railways = []
        self.way_count = 0

    def way(self, w):
        self.way_count += 1
        if self.way_count % 1_000_000 == 0:
            logging.info(f"Processed {self.way_count} ways...")

        # (1) All `bridge=yes` ways
        if "bridge" in w.tags and w.tags.get("bridge") == "yes":
            coords = [(n.lon, n.lat) for n in w.nodes if n.location.valid()]
            if len(coords) > 1:
                # Default layer = 0 if not present
                try:
                    layer = int(w.tags.get("layer", "0"))
                except ValueError:
                    layer = 0
                self.bridges.append({
                    "geometry": LineString(coords),
                    "layer": layer
                })

        # (2) All `railway=*` ways
        if "railway" in w.tags:
            coords = [(n.lon, n.lat) for n in w.nodes if n.location.valid()]
            if len(coords) > 1:
                # Default layer = 0 if not present
                try:
                    layer = int(w.tags.get("layer", "0"))
                except ValueError:
                    layer = 0
                self.railways.append({
                    "geometry": LineString(coords),
                    "layer": layer
                })


def download_pbf():
    """Download the Germany PBF if not present."""
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


def process_data():
    """
    - Parse the OSM PBF (or load from cache).
    - Build GeoDataFrames of bridges and railways with `layer` values.
    - Spatially intersect them to determine if railway is above or below.
    - Return only bridges that have a railway underneath.
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
            pickle.dump({"bridges": bridges_list, "railways": railway_list}, f)
        logging.info(f"Data cached in '{CACHE_FILE}'.")

    # Convert to GeoDataFrames
    logging.info("Building GeoDataFrames...")
    bridges_gdf = gpd.GeoDataFrame(bridges_list, geometry="geometry", crs="EPSG:4326")
    railways_gdf = gpd.GeoDataFrame(railway_list, geometry="geometry", crs="EPSG:4326")

    logging.info(f"Bridges: {len(bridges_gdf)}, Railways: {len(railways_gdf)}")

    # Buffer bridges for better intersection detection
    logging.info("Buffering bridges by 10m in EPSG:3857...")
    bridges_3857 = bridges_gdf.to_crs(epsg=3857)
    bridges_3857["geometry"] = bridges_3857.geometry.buffer(10)
    bridges_buffered = bridges_3857.to_crs(epsg=4326)

    logging.info("Spatial join: Bridges <-> Railways...")
    joined = gpd.sjoin(
        bridges_buffered,
        railways_gdf,
        how="inner",
        predicate="intersects"
    )
    logging.info(f"Found {len(joined)} bridge-railway intersections.")

    # Determine if railway is "above" or "below"
    def classify_position(row):
        # According to your logic:
        # If bridge layer > railway layer => railway is below.
        # If bridge layer < railway layer => railway is above.
        if row["layer_right"] < row["layer_left"]:
            return "below"
        elif row["layer_right"] > row["layer_left"]:
            return "above"
        else:
            return "equal"

    logging.info("Classifying positions...")
    joined["position"] = joined.apply(classify_position, axis=1)


    # Reset index so that the left index becomes a column
    joined.reset_index(inplace=True)

    # Filter: only keep bridges with a railway below
    bridges_with_railway_below = joined[joined["position"] == "below"]

    # Drop duplicates using the left index (now in the 'index' column)
    bridges_with_railway_below = bridges_with_railway_below.drop_duplicates(subset="index")
    logging.info(f"Bridges with railway below: {len(bridges_with_railway_below)}")

    # Save the results to a GeoJSON for mapping or further analysis
    output_file = "bridges_with_railway_below.geojson"
    bridges_with_railway_below.to_file(output_file, driver="GeoJSON")
    logging.info(f"Saved bridges with railway below to '{output_file}'.")



if __name__ == "__main__":
    download_pbf()
    process_data()
