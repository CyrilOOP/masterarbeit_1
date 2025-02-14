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
TRAIN_GPS_FILE = "subsets_by_date/2024-04-02/2024-04-02_rollingW_planar_time_headingDS_yawRate.csv"            # Path to your CSV file

# CRS settings
ORIGINAL_CRS = "EPSG:4326"   # Input data is in WGS84 (lat/lon)
PROJECTED_CRS = "EPSG:32633" # Example: UTM zone 33N (adjust as needed)

# Buffer distance (in meters) for checking proximity between train and infrastructure
BUFFER_DISTANCE = 10

# Angle threshold (in degrees) to decide if train and infrastructure are aligned.
# In the half-circle system (0–180°), a small difference (e.g., less than THRESHOLD) is considered aligned.
ANGLE_THRESHOLD = 5

# Column names in the train GPS CSV
LATITUDE_COLUMN = "GPS_lat"
LONGITUDE_COLUMN = "GPS_lon"
TIMESTAMP_COLUMN = "DatumZeit"
# Column that already contains the train heading (0–360° relative to North)
HEADING_COLUMN = "heading_deg_ds"

# Timestamp format in the CSV (e.g., "2024-04-02 17:44:00.100")
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

# -------------------------------
# Helper Functions
# -------------------------------

def get_line_orientation_at_point(line, pt, delta=1.0):
    """
    Calculate the orientation (tangent bearing) of a line at the location nearest to pt.
    The function projects the train point onto the line, then interpolates a point
    delta meters ahead (or behind if near the end) to compute the tangent direction.
    Returns a bearing (0–360°) relative to North.
    """
    d = line.project(pt)
    if d + delta <= line.length:
        p1 = line.interpolate(d)
        p2 = line.interpolate(d + delta)
    else:
        p1 = line.interpolate(d - delta)
        p2 = line.interpolate(d)
    # Calculate bearing using the standard formula:
    lat1, lon1 = math.radians(p1.y), math.radians(p1.x)
    lat2, lon2 = math.radians(p2.y), math.radians(p2.x)
    d_lon = lon2 - lon1
    x = math.sin(d_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(d_lon))
    bearing = math.degrees(math.atan2(x, y))
    result = (bearing + 360) % 360
    print(f"Line orientation at point {pt}: {result:.2f}° (using points {p1} and {p2})")
    return result

def angle_difference_half(train_angle, infra_angle):
    """
    Given a train bearing (0–360°) and an infrastructure bearing (0–180°),
    convert the train bearing into a half-circle value (0–180°) and compute
    the smallest difference between them.
    """
    # Convert train bearing (0–360°) into a half-circle value (0–180°)
    train_half = train_angle % 180
    diff = abs(train_half - infra_angle)
    if diff > 90:
        diff = 180 - diff
    return diff

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

# Use the pre-calculated train bearing (0–360°) from CSV.
print("Assigning pre-calculated train bearings from CSV...")
train_gdf["bearing"] = train_df[HEADING_COLUMN]

# -------------------------------
# Check Infrastructure Proximity and Direction
# -------------------------------

print("Checking infrastructure proximity and direction for each train point...")
results = []
for idx, train_row in train_gdf.iterrows():
    train_point = train_row['geometry']
    train_bearing = train_row['bearing']
    print(f"\nTrain point {idx} at {train_point} with bearing (0–360°): {train_bearing}")

    # Create a buffer around the train point for detecting nearby infrastructure.
    train_buffer = train_point.buffer(BUFFER_DISTANCE)

    # Find all infrastructure features that intersect this buffer.
    nearby_infra = infrastructure_gdf[infrastructure_gdf.intersects(train_buffer)]
    print(f"Found {len(nearby_infra)} nearby infrastructure features.")

    for infra_idx, infra_row in nearby_infra.iterrows():
        infra_geom = infra_row['geometry']
        # Calculate the orientation of the infrastructure at the point closest to the train.
        infra_bearing = get_line_orientation_at_point(infra_geom, train_point)
        if infra_bearing is None:
            continue

        # Compute the angle difference in the half-circle system.
        angle_diff = angle_difference_half(train_bearing, infra_bearing)
        print(f"Angle difference (half-circle) between train and infra: {angle_diff:.2f}°")
        if angle_diff < ANGLE_THRESHOLD:
            status = "On bridge (aligned direction)"
        else:
            status = "Under bridge (non-aligned direction)"
        print(f"Infra feature {infra_idx}: train_bearing={train_bearing}, infra_bearing={infra_bearing}, status={status}")

        results.append({
            "timestamp": train_row[TIMESTAMP_COLUMN],
            "train_longitude": train_row.geometry.x,
            "train_latitude": train_row.geometry.y,
            "train_bearing": train_bearing,
            "infra_id": infra_idx,
            "infra_bearing": infra_bearing,
            "angle_diff": angle_diff,
            "status": status,
            "geometry": train_row.geometry  # saving geometry for mapping
        })

results_df = pd.DataFrame(results)
print("Converting results to GeoDataFrame...")
results_gdf = gpd.GeoDataFrame(results_df, geometry="geometry", crs=PROJECTED_CRS)

# -------------------------------
# Create Interactive Map with Folium
# -------------------------------

print("Preparing map for display...")
results_gdf = results_gdf.to_crs("EPSG:4326")
infrastructure_gdf = infrastructure_gdf.to_crs("EPSG:4326")

# Filter for only points where the train is under a bridge (non-aligned direction)
under_bridge_gdf = results_gdf[results_gdf['status'] == "Under bridge (non-aligned direction)"]
print(f"Number of under-bridge points: {len(under_bridge_gdf)}")

if not under_bridge_gdf.empty:
    center = [under_bridge_gdf.geometry.iloc[0].y, under_bridge_gdf.geometry.iloc[0].x]
else:
    center = [0, 0]

m = folium.Map(location=center, zoom_start=14)
print("Adding under-bridge points to map...")
for idx, row in under_bridge_gdf.iterrows():
    lat = row.geometry.y
    lon = row.geometry.x
    popup_text = (f"Timestamp: {row.timestamp}<br>"
                  f"Train Bearing (0–360°): {row.train_bearing}<br>"
                  f"Infra Bearing (0–180°): {row.infra_bearing}<br>"
                  f"Angle Diff (mod 180): {row.angle_diff}<br>"
                  f"Status: {row.status}")
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
