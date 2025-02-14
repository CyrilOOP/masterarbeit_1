import math
import pandas as pd
import geopandas as gpd
import folium
from shapely.geometry import Point

# -------------------------------
# Configuration
# -------------------------------

# File paths
INFRASTRUCTURE_FILE = "sachsen-anhalt_combined_infrastructure.geojson"  # Path to your GeoJSON file
TRAIN_GPS_FILE = "subsets_by_date/2024-04-02/2024-04-02.csv"  # Path to your CSV file

# CRS settings
ORIGINAL_CRS = "EPSG:4326"  # Input data is in WGS84 (lat/lon)
PROJECTED_CRS = "EPSG:32633"  # Example: UTM zone 33N (adjust as needed)

# Buffer distance (in meters) for checking proximity between train and infrastructure
BUFFER_DISTANCE = 10

# Column names in the train GPS CSV for latitude, longitude, and timestamp
LATITUDE_COLUMN = "GPS_lat"
LONGITUDE_COLUMN = "GPS_lon"
TIMESTAMP_COLUMN = "DatumZeit"

# Timestamp format in the CSV (e.g., "2024-04-02 17:44:00.100")
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

# -------------------------------
# Load Data
# -------------------------------

print("Loading infrastructure data...")
infrastructure_gdf = gpd.read_file(INFRASTRUCTURE_FILE)

print("Loading train GPS data...")
train_df = pd.read_csv(TRAIN_GPS_FILE)
train_df[TIMESTAMP_COLUMN] = pd.to_datetime(train_df[TIMESTAMP_COLUMN], format=TIMESTAMP_FORMAT)

print("Creating GeoDataFrame for train positions...")
train_gdf = gpd.GeoDataFrame(
    train_df,
    geometry=train_df.apply(lambda row: Point(row[LONGITUDE_COLUMN], row[LATITUDE_COLUMN]), axis=1),
    crs=ORIGINAL_CRS
)

print("Reprojecting data to projected CRS for distance calculations...")
infrastructure_gdf = infrastructure_gdf.to_crs(PROJECTED_CRS)
train_gdf = train_gdf.to_crs(PROJECTED_CRS)

# -------------------------------
# Flag Problematic Points
# -------------------------------
# For each train point, check if it falls within a buffer of any infrastructure feature.
print("Checking which train points are within the buffer distance of infrastructure...")
problematic_points = []  # Will store train points that are near infrastructure.

for idx, train_row in train_gdf.iterrows():
    train_point = train_row['geometry']
    # Create a buffer around the train point.
    train_buffer = train_point.buffer(BUFFER_DISTANCE)

    # Check if this buffer intersects any infrastructure feature.
    nearby_infra = infrastructure_gdf[infrastructure_gdf.intersects(train_buffer)]

    if not nearby_infra.empty:
        #print(f"Train point {idx} at {train_point} is near {len(nearby_infra)} infrastructure feature(s).")
        # Flag this point as problematic.
        problematic_points.append(train_row)

# Create a GeoDataFrame of the problematic points.
if problematic_points:
    problematic_gdf = gpd.GeoDataFrame(problematic_points, crs=PROJECTED_CRS)
else:
    problematic_gdf = gpd.GeoDataFrame(columns=train_gdf.columns, crs=PROJECTED_CRS)
print(f"Total problematic train points: {len(problematic_gdf)}")

# -------------------------------
# Create Interactive Map with Folium
# -------------------------------
print("Preparing map for display...")

# Convert to WGS84 for Folium.
problematic_gdf = problematic_gdf.to_crs("EPSG:4326")
infrastructure_gdf = infrastructure_gdf.to_crs("EPSG:4326")

# Choose a central location for the map (using the first problematic point as an example)
if not problematic_gdf.empty:
    center = [problematic_gdf.geometry.iloc[0].y, problematic_gdf.geometry.iloc[0].x]
else:
    center = [0, 0]

m = folium.Map(location=center, zoom_start=14)

# Add problematic train points to the map.
print("Adding problematic points to map...")
for idx, row in problematic_gdf.iterrows():
    lat = row.geometry.y
    lon = row.geometry.x
    popup_text = (f"Timestamp: {row[TIMESTAMP_COLUMN]}<br>"
                  f"Location: ({row.geometry.y:.5f}, {row.geometry.x:.5f})")
    folium.CircleMarker(
        location=[lat, lon],
        radius=5,
        popup=popup_text,
        color="red",
        fill=True,
        fill_color="red"
    ).add_to(m)


print("Saving map to train_infrastructure_map.html...")
m.save("train_infrastructure_map.html")
print("Map saved. Open 'train_infrastructure_map.html' to view the results.")
m
