import os
import requests
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
config = {
    "overpass_url": "https://overpass-api.de/api/interpreter",
    "bbox": [47.2, 5.9, 55.1, 15.0],  # [min_lat, min_lon, max_lat, max_lon] for Germany
    "threshold": 0.01  # not used in this script, but kept for consistency
}


# --- FUNCTION TO FETCH BRIDGE DATA FROM OSM ---
def fetch_railway_bridges(config):
    bbox = config.get('bbox')
    overpass_url = config.get('overpass_url')

    # Query for railway bridges (ways with railway=rail and bridge=yes)
    query = f"""
    [out:json];
    (
      way["railway"="rail"]["bridge"="yes"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
    );
    out body;
    >;
    out skel qt;
    """

    print("Fetching railway bridges from OpenStreetMap...")
    response = requests.get(overpass_url, params={'data': query})
    data = response.json()
    print("Data received!")

    # Extract ways (bridges) and nodes
    ways = {el["id"]: el["nodes"] for el in data["elements"] if el["type"] == "way"}
    nodes = {el["id"]: (el["lat"], el["lon"]) for el in data["elements"] if el["type"] == "node"}

    # Build a list of bridges with their coordinates
    bridge_list = []
    for way_id, node_ids in ways.items():
        coords = [nodes[nid] for nid in node_ids if nid in nodes]
        if len(coords) > 1:
            bridge_list.append({"bridge_id": way_id, "coordinates": coords})

    bridge_df = pd.DataFrame(bridge_list)
    return bridge_df


# --- FUNCTION TO GET BRIDGE DATA (Load from File if Available, Otherwise Download) ---
def get_bridges_data(config, file_path="bridges.csv"):
    if os.path.exists(file_path):
        print(f"Loading bridge data from {file_path} ...")
        # Since the 'coordinates' column contains lists, we use a converter.
        # Note: using eval is not always recommended for untrusted data.
        bridge_df = pd.read_csv(file_path, converters={"coordinates": eval})
        print("Bridge data loaded.")
    else:
        print("Bridge data file not found. Fetching from Overpass API...")
        bridge_df = fetch_railway_bridges(config)
        # Save the DataFrame to CSV.
        # (This will save the 'coordinates' column as a string representation of the list.)
        bridge_df.to_csv(file_path, index=False)
        print(f"Bridge data saved to {file_path}.")
    return bridge_df


# --- MAIN FUNCTION ---
def main():
    # Set the file path for the bridge data.
    bridge_file = "bridges.csv"

    # Get the bridge data (either by loading from file or fetching from the internet)
    bridges_df = get_bridges_data(config, file_path=bridge_file)

    # For debugging: print the first few rows and total count.
    print("Bridge data sample:")
    print(bridges_df.head())
    print(f"Total bridges fetched/loaded: {len(bridges_df)}")


if __name__ == "__main__":
    main()
