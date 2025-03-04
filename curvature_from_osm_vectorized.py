import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import math
from tqdm import tqdm  # optional progress bar

############################
# 1) LOAD TRAIN DATA (LAT/LON IN EPSG:4326)
############################
csv_file = "subsets_by_date/2024-04-02/2024-04-02_time_rollingW_planar_distance_headingDX_headingDS_yawRate_radius_übogen.csv"  # <-- your CSV with columns 'GPS_lat' & 'GPS_lon'
print(f"Loading train data from '{csv_file}'...")
df_train = pd.read_csv(csv_file)
print(f" -> Loaded {df_train.shape[0]} rows.")

# Create a GeoDataFrame using 'geometry' as the default column name,
# with lat/lon in EPSG:4326 (WGS 84).
gdf_train = gpd.GeoDataFrame(
    df_train,
    geometry=gpd.points_from_xy(df_train["GPS_lon"], df_train["GPS_lat"]),
    crs="EPSG:4326"
)
print(f" -> gdf_train has CRS={gdf_train.crs} and {gdf_train.shape[0]} features.")

# Rename 'geometry' to 'geom_train' and set it as the active geometry
gdf_train.rename(columns={"geometry": "geom_train"}, inplace=True)
gdf_train.set_geometry("geom_train", inplace=True)

############################
# 2) LOAD THE RAILWAY LINES (ALSO IN EPSG:4326)
############################
shp_file = "railways.shp"  # <-- your rail shapefile in lat/lon
print(f"Loading railway lines from '{shp_file}'...")
gdf_rail = gpd.read_file(shp_file)
print(f" -> Loaded {gdf_rail.shape[0]} lines with CRS={gdf_rail.crs}")

# If gdf_rail is not EPSG:4326, reproject it:
# e.g. if gdf_rail.crs is None or different, do:
# gdf_rail = gdf_rail.set_crs("EPSG:4326")  # or to_crs if needed
if gdf_rail.crs != "EPSG:4326":
    print(f"Reprojecting gdf_rail from {gdf_rail.crs} to EPSG:4326...")
    gdf_rail = gdf_rail.to_crs("EPSG:4326")

# Rename its geometry column to 'geom_rail' and set as active
gdf_rail.rename(columns={"geometry": "geom_rail"}, inplace=True)
gdf_rail.set_geometry("geom_rail", inplace=True)

############################
# 3) SPATIAL JOIN (NEAREST)
############################
# Because GeoPandas 1.0.1 won't automatically merge the right geometry,
# we'll only keep the left geometry and store 'index_right'.
print("Performing sjoin_nearest to match each train point to the closest railway line...")
gdf_joined = gpd.sjoin_nearest(
    gdf_train,
    gdf_rail,
    how="left",
    distance_col="dist2rail"  # distance in degrees, typically, if both are lat/lon
)
print(f" -> sjoin_nearest done. Columns: {gdf_joined.columns}")
print(f" -> gdf_joined has {gdf_joined.shape[0]} rows.")

############################
# 4) MERGE THE RAIL GEOMETRY MANUALLY
############################
# 'index_right' is the row index in gdf_rail. We'll join to retrieve 'geom_rail'.
gdf_matched = gdf_joined.join(
    gdf_rail[["geom_rail"]],  # just the geometry from the rail
    on="index_right",
    how="left"
)
# Now gdf_matched should have: 'geom_train' (train pts), 'geom_rail' (rail lines).

# For clarity, rename 'geom_rail' to 'matched_line'
gdf_matched["matched_line"] = gdf_matched["geom_rail"]


############################
# 5) DEFINE A CIRCLE-FITTING FUNCTION (IN LAT/LON!!)
############################
def circle_radius_from_three_points(lat1, lon1, lat2, lon2, lat3, lon3):
    """
    Compute the circle radius through three points given in lat/lon (degrees).
    **WARNING**: This is a naive Euclidean approach on lat/lon, which is
    not really correct for large distances. For more accurate results,
    reproject your data to a planar CRS first!

    Returns float('inf') if collinear or degenerate.
    """
    # We'll treat (lon, lat) as if they're planar x,y -> This is rough for lat/lon
    import math
    def dist(ax, ay, bx, by):
        return math.hypot(ax - bx, ay - by)

    # Let's treat (lon, lat) as (x, y).
    a = dist(lon2, lat2, lon3, lat3)
    b = dist(lon1, lat1, lon3, lat3)
    c = dist(lon1, lat1, lon2, lat2)

    s = (a + b + c) / 2.0
    area = math.sqrt(max(s * (s - a) * (s - b) * (s - c), 0))
    if area == 0:
        return float('inf')
    return (a * b * c) / (4.0 * area)


def compute_map_radius_for_point(pt, line):
    """
    For a lat/lon point (pt) and matched line (line),
    find the nearest vertex and compute curvature.

    Because we're in lat/lon, all distances are naive "degree distances".
    This is only a rough approximation.
    """
    if line is None or not isinstance(line, LineString):
        return None

    # nearest point on 'line'
    nearest_pt_on_line = line.interpolate(line.project(pt))

    coords = list(line.coords)  # lat/lon coords in shapely
    # find index of the nearest vertex
    nearest_idx = min(
        range(len(coords)),
        key=lambda i: Point(coords[i]).distance(nearest_pt_on_line)
    )

    if 0 < nearest_idx < len(coords) - 1:
        # coords[i] is (lon, lat), shapely typically stores as (x, y)
        lon1, lat1 = coords[nearest_idx - 1]
        lon2, lat2 = coords[nearest_idx]
        lon3, lat3 = coords[nearest_idx + 1]
        return circle_radius_from_three_points(lat1, lon1, lat2, lon2, lat3, lon3)
    else:
        return None


############################
# 6) COMPUTE CURVATURE FOR EACH TRAIN POINT
############################
print("Computing naive lat/lon-based radius for each point...")
radii = []
for idx, row in tqdm(gdf_matched.iterrows(), total=gdf_matched.shape[0]):
    pt = row["geom_train"]  # The train point (lat/lon)
    line_geom = row["matched_line"]  # The matched rail line
    r = compute_map_radius_for_point(pt, line_geom)
    radii.append(r)

gdf_matched["map_radius"] = radii
print(" -> Done computing radii.")

############################
# 7) SAVE/INSPECT RESULTS
############################
output_csv = "train_gps_with_map_radius.csv"
print(f"Saving final data to '{output_csv}'...")
gdf_matched.to_csv(output_csv, index=False)

print("Sample output:")
print(gdf_matched[["GPS_lat", "GPS_lon", "dist2rail", "map_radius"]].head(10))
print("Finished!")
