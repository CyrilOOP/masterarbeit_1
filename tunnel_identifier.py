import requests
import pandas as pd
from geopy.distance import geodesic
#import ace_tools as tools  # For displaying DataFrames in ChatGPT (optional)

# --- CONFIGURATION ---
config = {
    "overpass_url": "https://overpass-api.de/api/interpreter",
    "bbox": [47.2, 5.9, 55.1, 15.0],  # [min_lat, min_lon, max_lat, max_lon] for Germany
    "threshold": 0.5  # distance in km to consider "near" a tunnel
}


# --- FUNCTION TO FETCH TUNNEL DATA FROM OSM ---
def fetch_railway_tunnels(config):
    bbox = config.get('bbox')
    overpass_url = config.get('overpass_url')

    query = f"""
    [out:json];
    (
      way["railway"="rail"]["tunnel"="yes"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
    );
    out body;
    >;
    out skel qt;
    """

    print("Fetching railway tunnels from OpenStreetMap...")
    response = requests.get(overpass_url, params={'data': query})
    data = response.json()
    print("Data received!")

    # Extract ways (tunnels) and nodes
    ways = {el["id"]: el["nodes"] for el in data["elements"] if el["type"] == "way"}
    nodes = {el["id"]: (el["lat"], el["lon"]) for el in data["elements"] if el["type"] == "node"}

    # Build a list of tunnels with their coordinates
    tunnel_list = []
    for way_id, node_ids in ways.items():
        coords = [nodes[nid] for nid in node_ids if nid in nodes]
        if len(coords) > 1:
            tunnel_list.append({"tunnel_id": way_id, "coordinates": coords})

    tunnel_df = pd.DataFrame(tunnel_list)
    return tunnel_df


# --- FUNCTION TO ADD TUNNEL STATUS TO THE TRAIN DATAFRAME ---
def add_tunnel_status_to_train_df(train_df, tunnel_df, config):
    """

    For each row in train_df (which must contain 'latitude' and 'longitude'),
    determine if the train is within the given threshold (in km) of any tunnel.
    Adds a new column 'tunnel_status' to train_df.
    """
    threshold = config.get('threshold', 0.05)
    statuses = []

    # Iterate over train positions
    for idx, row in train_df.iterrows():
        train_gps = (row["GPS_lat"], row["GPS_lon"])
        status = "No nearby tunnel detected"

        # Check each tunnel's coordinates
        for _, tunnel in tunnel_df.iterrows():
            for point in tunnel["coordinates"]:
                if geodesic(train_gps, point).km < threshold:
                    status = f"Near tunnel {tunnel['tunnel_id']}"
                    break
            if status != "No nearby tunnel detected":
                break

        statuses.append(status)

    train_df["tunnel_status"] = statuses
    return train_df


# --- FUNCTION THAT PROCESSES THE TRAIN DATAFRAME ---
def process_train_data(train_df):
    """
    Processes the input train DataFrame by fetching railway tunnels and
    adding a tunnel proximity status column.

    Parameters:
        train_df (pd.DataFrame): A DataFrame with at least 'timestamp', 'latitude', and 'longitude' columns.

    Returns:
        pd.DataFrame: The updated DataFrame with an additional 'tunnel_status' column.
    """
    # 1. Fetch tunnel data from OSM
    tunnel_df = fetch_railway_tunnels(config)
    #tools.display_dataframe_to_user(name="Railway Tunnels in Germany", dataframe=tunnel_df)

    # 2. Add tunnel proximity status to the train DataFrame
    processed_train_df = add_tunnel_status_to_train_df(train_df, tunnel_df, config)
    #tools.display_dataframe_to_user(name="Train Data with Tunnel Status", dataframe=processed_train_df)

    return processed_train_df


# --- MAIN FUNCTION ---
def main():
    # Hardcoded file paths for input and output
    input_file = "subsets_by_date/2024-04-02/2024-04-02_rollingW_planar_time_headingDS_yawRate_curvature.csv"
    output_file = "processed_train_data.csv"

    # Load the CSV file into a DataFrame
    try:
        train_df = pd.read_csv(input_file)
        print(f"Loaded train data from {input_file}")
    except Exception as e:
        print(f"Error loading CSV file {input_file}: {e}")
        return

    # Process the train data to add tunnel status
    processed_df = process_train_data(train_df)

    # Save the processed DataFrame to the output CSV file
    try:
        processed_df.to_csv(output_file, index=False)
        print(f"\n✅ Processed data saved to {output_file}")
    except Exception as e:
        print(f"Error saving CSV file {output_file}: {e}")


if __name__ == "__main__":
    main()
