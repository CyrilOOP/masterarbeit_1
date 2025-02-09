import requests
import pandas as pd
import folium

# --- CONFIGURATION ---
config = {
    "overpass_url": "https://overpass-api.de/api/interpreter",
    "bbox": [47.2, 5.9, 55.1, 15.0],  # [min_lat, min_lon, max_lat, max_lon] for Germany
    "threshold": 0.5  # (Not used in this script; used for distance checks in other scripts)
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


# --- FUNCTION TO DISPLAY Tunnels on a Folium Map ---
def display_tunnels_on_map(tunnel_df, center=[51.1657, 10.4515], zoom_start=6):
    """
    Creates a Folium map centered over the given location and adds each tunnel as a blue polyline.

    Args:
        tunnel_df (pd.DataFrame): DataFrame containing tunnel_id and list of coordinates.
        center (list): [lat, lon] for the center of the map.
        zoom_start (int): Initial zoom level for the map.

    Returns:
        folium.Map: The generated map.
    """
    m = folium.Map(location=center, zoom_start=zoom_start, tiles="OpenStreetMap")

    for idx, row in tunnel_df.iterrows():
        coords = row["coordinates"]
        # Add the tunnel as a blue polyline with a popup showing the tunnel_id
        folium.PolyLine(
            locations=coords,
            color="blue",
            weight=3,
            opacity=0.7,
            popup=f"Tunnel {row['tunnel_id']}"
        ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save("tunnels_map.html")
    print("Map saved as 'tunnels_map.html'. Open it in your browser to view the tunnels.")
    return m


# --- MAIN FUNCTION ---
def main():
    tunnel_df = fetch_railway_tunnels(config)
    # (Optional) Print some debug information
    print("Sample of fetched tunnel data:")
    print(tunnel_df.head())
    print(f"Total tunnels fetched: {len(tunnel_df)}")

    # Create and display the map with tunnels
    display_tunnels_on_map(tunnel_df)


if __name__ == "__main__":
    main()

