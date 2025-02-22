import pandas as pd
import folium
import matplotlib.colors as mcolors

# === Configuration Variables ===
CSV_INPUT_FILE = 'output_with_elevation.csv'  # CSV file with GPS & elevation data
LAT_COLUMN = 'GPS_lat'  # Column name for latitude
LON_COLUMN = 'GPS_lon'  # Column name for longitude
ELEVATION_COLUMN = 'elevation'  # Column name for elevation
OUTPUT_MAP_FILE = 'gps_elevation_map.html'  # Output HTML file for interactive map

# === Read CSV File ===
df = pd.read_csv(CSV_INPUT_FILE)

# === Create a Base Map (Centered on Average Coordinates) ===
map_center = [df[LAT_COLUMN].mean(), df[LON_COLUMN].mean()]
map = folium.Map(location=map_center, zoom_start=10, tiles="OpenStreetMap")

# === Function to Assign Color Based on Elevation ===
def get_color(elevation, min_elev, max_elev):
    norm = (elevation - min_elev) / (max_elev - min_elev) if max_elev != min_elev else 0.5
    cmap = mcolors.LinearSegmentedColormap.from_list("elev_cmap", ["blue", "green", "yellow", "red"])
    return mcolors.to_hex(cmap(norm))

# Get min/max elevation to normalize colors
min_elev, max_elev = df[ELEVATION_COLUMN].min(), df[ELEVATION_COLUMN].max()

# === Add Markers to the Map ===
for _, row in df.iterrows():
    color = get_color(row[ELEVATION_COLUMN], min_elev, max_elev)
    folium.CircleMarker(
        location=[row[LAT_COLUMN], row[LON_COLUMN]],
        radius=4,  # Marker size
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=f"Elevation: {row[ELEVATION_COLUMN]:.2f} m"
    ).add_to(map)

# === Save and Display the Map ===
map.save(OUTPUT_MAP_FILE)
print(f"\n✅ Map saved as {OUTPUT_MAP_FILE}. Open it in a web browser to view.")

