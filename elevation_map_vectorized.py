import pandas as pd
import folium
import matplotlib.colors as mcolors
import numpy as np

# === Configuration Variables ===
CSV_INPUT_FILE = 'output_with_elevation.csv'  # CSV file with GPS & elevation data
LAT_COLUMN = 'latitude'  # Column name for latitude
LON_COLUMN = 'longitude'  # Column name for longitude
ELEVATION_COLUMN = 'elevation'  # Column name for elevation
OUTPUT_MAP_FILE = 'gps_elevation_map.html'  # Output HTML file for interactive map

print("📥 Loading CSV file...")
df = pd.read_csv(CSV_INPUT_FILE)
print(f"✅ Loaded {len(df)} rows from {CSV_INPUT_FILE}")

# === Create a Base Map ===
print("🗺️ Creating base map...")
map_center = [df[LAT_COLUMN].mean(), df[LON_COLUMN].mean()]
map = folium.Map(location=map_center, zoom_start=10, tiles="OpenStreetMap")
print("✅ Base map created!")

# === Normalize Elevation to Colors (Vectorized) ===
print("🎨 Mapping elevation to colors...")
min_elev, max_elev = df[ELEVATION_COLUMN].min(), df[ELEVATION_COLUMN].max()
norm_elev = (df[ELEVATION_COLUMN] - min_elev) / (max_elev - min_elev + 1e-9)  # Avoid div by zero

# Create colormap
cmap = mcolors.LinearSegmentedColormap.from_list("elev_cmap", ["blue", "green", "yellow", "red"])
df["color"] = [mcolors.to_hex(cmap(norm)) for norm in norm_elev]
print("✅ Elevation colors assigned!")

# === Add All Points in One Go (Vectorized) ===
print("📍 Adding points to the map...")
locations = df[[LAT_COLUMN, LON_COLUMN]].values.tolist()
colors = df["color"].values
popups = [f"Elevation: {elev:.2f} m" for elev in df[ELEVATION_COLUMN]]

for idx, (loc, color, popup) in enumerate(zip(locations, colors, popups)):
    folium.CircleMarker(
        location=loc,
        radius=4,  # Marker size
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=popup
    ).add_to(map)

    # Print progress every 10,000 points
    if (idx + 1) % 10000 == 0:
        print(f"📊 Processed {idx + 1}/{len(df)} points...")

print("✅ All points added to the map!")

# === Save and Display the Map ===
map.save(OUTPUT_MAP_FILE)
print(f"\n✅ Map saved as {OUTPUT_MAP_FILE}. Open it in a web browser to view.")
