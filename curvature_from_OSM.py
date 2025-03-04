import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import math
from pyproj import Transformer

########################
# 1) Load Your Data
########################
# Example: A CSV file with columns [latitude, longitude], or similar.
df_train = pd.read_csv("subsets_by_date/2024-04-02/2024-04-02_time_rollingW_planar_distance_headingDX_headingDS_yawRate_radius_übogen.csv")

# Convert to a GeoDataFrame
gdf_train = gpd.GeoDataFrame(
    df_train,
    geometry=gpd.points_from_xy(df_train["GPS_lon_smoothed_rolling_windows"], df_train["GPS_lat_smoothed_rolling_windows"]),
    crs="EPSG:4326"
)

# Load the railway lines (extracted from Script #1)
gdf_rail = gpd.read_file("railways.shp")  # This should have CRS = EPSG:4326 if unchanged.

# To accurately compute distances, reproject both to a projected CRS (e.g. EPSG:3857).
gdf_train = gdf_train.to_crs("EPSG:32633")
gdf_rail = gdf_rail.to_crs("EPSG:32633")


########################
# 2) Map Matching: Find Nearest Railway for Each GPS Point
########################
# A simple method: for each point, find the closest line in the dataset.
# (For large datasets, spatial indexing is recommended to speed this up.)

def get_nearest_railway(point, rail_gdf):
    """
    Returns the geometry of the closest railway line to 'point'.
    """
    # Compute distances from this point to each railway line
    distances = rail_gdf.geometry.distance(point)
    nearest_idx = distances.idxmin()
    return rail_gdf.loc[nearest_idx, 'geometry']


gdf_train["matched_rail"] = gdf_train["geometry"].apply(
    lambda p: get_nearest_railway(p, gdf_rail)
)


########################
# 3) Define a Function to Compute Radius from Three Points
########################
# We'll assume input points are (lat, lon) in EPSG:4326, but since we are
# already in EPSG:3857, we'll adapt accordingly.
#
# If you prefer to do the circle fit in lat/lon, you can transform
# back to EPSG:4326, but typically a projected CRS is better for geometry.

def circle_radius_from_three_points(x1, y1, x2, y2, x3, y3):
    """
    Given three points in a projected coordinate system (x, y),
    compute the radius of the circle passing through them.
    Return float('inf') if collinear.
    """

    def dist(ax, ay, bx, by):
        return math.hypot(ax - bx, ay - by)

    a = dist(x2, y2, x3, y3)
    b = dist(x1, y1, x3, y3)
    c = dist(x1, y1, x2, y2)

    s = (a + b + c) / 2.0
    area = math.sqrt(max(s * (s - a) * (s - b) * (s - c), 0))

    if area == 0:
        return float('inf')  # collinear
    return (a * b * c) / (4.0 * area)


########################
# 4) Compute the Map-Based Radius at Each GPS Point
########################
def compute_map_radius(gps_point, matched_line):
    """
    Given a GPS point (Point) and the matched railway line (LineString),
    find the nearest vertex in the line, gather that vertex and its neighbors,
    and compute the circle radius.
    """
    if not isinstance(matched_line, LineString):
        return None  # If it's MultiLineString or something else, handle accordingly

    # Find the nearest point on the matched line
    nearest_point_on_line = matched_line.interpolate(matched_line.project(gps_point))

    # Convert the line's coords to a list of (x, y)
    coords = list(matched_line.coords)
    # We'll find whichever vertex is closest to our 'nearest_point_on_line'
    nearest_idx = min(range(len(coords)), key=lambda i: Point(coords[i]).distance(nearest_point_on_line))

    # We need 3 consecutive points to compute curvature
    if 0 < nearest_idx < len(coords) - 1:
        (x1, y1) = coords[nearest_idx - 1]
        (x2, y2) = coords[nearest_idx]
        (x3, y3) = coords[nearest_idx + 1]

        # Compute the circle radius
        return circle_radius_from_three_points(x1, y1, x2, y2, x3, y3)
    else:
        # at the ends, we can't get 3 consecutive points
        return None


# Apply the function for each row
gdf_train["map_radius"] = gdf_train.apply(
    lambda row: compute_map_radius(row["geometry"], row["matched_rail"]),
    axis=1
)

########################
# 5) Save or Inspect the Results
########################
# Now you have a new column "map_radius" with the computed curvature radius at each GPS point
gdf_train.to_csv("train_gps_with_map_radius.csv", index=False)
print(gdf_train[["geometry", "map_radius"]].head())
