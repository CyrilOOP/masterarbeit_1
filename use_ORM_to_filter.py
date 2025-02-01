import requests
import geopandas as gpd
from shapely.geometry import LineString
import time


def fetch_railway_tracks_near(lat, lon, radius_km=5):
    """
    Fetches railway tracks near a given GPS coordinate using OpenStreetMap Overpass API.

    Parameters:
    - lat, lon: GPS coordinates of the train
    - radius_km: Search radius in kilometers

    Returns:
    - GeoDataFrame with railway track geometries
    """
    print(f"Fetching railway data near {lat}, {lon} (radius {radius_km} km)...")

    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    (
      way["railway"="rail"](around:{radius_km * 1000},{lat},{lon});
    );
    out geom;
    """

    response = requests.get(overpass_url, params={"data": overpass_query})

    if response.status_code != 200:
        print("Error fetching railway data:", response.status_code)
        return gpd.GeoDataFrame(columns=["geometry"])

    data = response.json()

    # Convert OSM data to GeoDataFrame
    railway_lines = []
    for element in data["elements"]:
        if "geometry" in element and element["type"] == "way":
            coords = [(node["lon"], node["lat"]) for node in element["geometry"]]
            railway_lines.append(LineString(coords))

    railway_gdf = gpd.GeoDataFrame(geometry=railway_lines, crs="EPSG:4326")

    print(f"Fetched {len(railway_gdf)} railway segments.")
    return railway_gdf
