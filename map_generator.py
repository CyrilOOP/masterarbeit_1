import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import folium
from folium.plugins import TimestampedGeoJson
from matplotlib import colormaps, colors
from branca.colormap import LinearColormap
import os

from csv_tools import csv_load, csv_select_gps_columns


def generate_map_from_csv(subset_full_path: str) -> None:
    """
    Generates a Folium map from CSV data using GPS_lat and GPS_lon exclusively.
    If 'Geschwindigkeit in m/s' is present, it converts to Speed_kmh for coloring.
    If 'Gier' or a yaw_rate column (e.g., 'yaw_rate_deg_s') are present, it draws
    additional color-coded paths.

    The map starts with no base tiles. All tile layers (OpenStreetMap, OpenRailwayMap, etc.)
    are added as togglable overlays. If all are turned off, the map background is empty.

    Args:
        subset_full_path: Path to the CSV file containing the data.

    Raises:
        ValueError: If the CSV file is empty or contains invalid data.
    """
    # -------------------------------------------------------------------------
    # 1. Load CSV
    # -------------------------------------------------------------------------
    base_dir = os.path.dirname(subset_full_path)
    df = pd.read_csv(subset_full_path, parse_dates=["DatumZeit"])

    if df.empty:
        raise ValueError("No data found in the CSV. Cannot generate map.")

    # -------------------------------------------------------------------------
    # 2. Extract Lat/Lon from GPS Columns
    # -------------------------------------------------------------------------
    if "selected_smoothing_method" in df.columns:
        selected_method = df["selected_smoothing_method"].iloc[0]
        if selected_method == "none":
            lat_vals = df["GPS_lat"]
            lon_vals = df["GPS_lon"]
            print("Using default columns: GPS_lat and GPS_lon")
        else:
            lat_col = f"GPS_lat_smoothed_{selected_method}"
            lon_col = f"GPS_lon_smoothed_{selected_method}"
            if lat_col in df.columns and lon_col in df.columns:
                lat_vals = df[lat_col]
                lon_vals = df[lon_col]
                print(f"Using smoothed columns: {lat_col} and {lon_col}")
            else:
                raise ValueError(
                    f"Columns {lat_col} and {lon_col} not found in the DataFrame. Check your smoothing configuration."
                )
    else:
        lat_vals = df["GPS_lat"]
        lon_vals = df["GPS_lon"]
        print("The 'selected_smoothing_method' column is missing. Using default columns: GPS_lat and GPS_lon")

    if "Geschwindigkeit in m/s" in df.columns:
        df["Speed_kmh"] = df["Geschwindigkeit in m/s"] * 3.6

    has_speed = "Speed_kmh" in df.columns and df["Speed_kmh"].notna().any()

    day_display = df["DatumZeit"].iloc[0].date()

    # -------------------------------------------------------------------------
    # 3. Build GeoDataFrame in EPSG:4326
    # -------------------------------------------------------------------------
    geometry = [
        Point(lon, lat) if pd.notna(lon) and pd.notna(lat) else None
        for lat, lon in zip(lat_vals, lon_vals)
    ]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    gdf.dropna(subset=["geometry"], inplace=True)
    if gdf.empty:
        raise ValueError("All geometry points were NaN. Cannot generate map.")

    def get_lat_lon(row):
        return (row.geometry.y, row.geometry.x)

    # -------------------------------------------------------------------------
    # 4. Initialize Folium Map with no base tiles
    # -------------------------------------------------------------------------
    start_lat, start_lon = get_lat_lon(gdf.iloc[0])
    m = folium.Map(location=[start_lat, start_lon], zoom_start=14, tiles=None)

    # -------------------------------------------------------------------------
    # 5. Add Overlay Tile Layers (toggleable)
    # -------------------------------------------------------------------------
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        name="OpenStreetMap",
        attr="&copy; OpenStreetMap contributors",
        overlay=True,
        opacity=0.7,
        show=True
    ).add_to(m)

    orm_layers = {
        "ORM - Standard": {"url": "https://{s}.tiles.openrailwaymap.org/standard/{z}/{x}/{y}.png", "show": True},
        "ORM - Electrified": {"url": "https://{s}.tiles.openrailwaymap.org/electrified/{z}/{x}/{y}.png", "show": False},
        "ORM - Signals": {"url": "https://{s}.tiles.openrailwaymap.org/signals/{z}/{x}/{y}.png", "show": False},
    }
    for layer_name, layer_info in orm_layers.items():
        folium.TileLayer(
            tiles=layer_info["url"],
            attr="&copy; OpenRailwayMap contributors",
            name=layer_name,
            overlay=True,
            show=layer_info["show"]
        ).add_to(m)

    # -------------------------------------------------------------------------
    # 6. GPS Path either from the rows or from the selected smoothed if exist
    # -------------------------------------------------------------------------
    uniform_path_fg = folium.FeatureGroup(name="Uniform Path", show=True)
    for i in range(len(gdf) - 1):
        lat1, lon1 = get_lat_lon(gdf.iloc[i])
        lat2, lon2 = get_lat_lon(gdf.iloc[i + 1])
        folium.PolyLine(
            [(lat1, lon1), (lat2, lon2)],
            color="blue",
            weight=8,
            opacity=1
        ).add_to(uniform_path_fg)
    uniform_path_fg.add_to(m)

    # -------------------------------------------------------------------------
    # 6.1. Additional GPS path if needed
    # -------------------------------------------------------------------------
    # Extract the base filename (e.g., '2025-02-02_something.csv')
    filename = os.path.basename(subset_full_path)
    # Extract the date part from the filename (assumes the filename starts with 'YYYY-MM-DD')
    date_extracted = filename.split('_')[0]
    # Get the directory where the original file is located
    directory = os.path.dirname(subset_full_path)
    # Construct the new filename and path (e.g., '2025-02-02.csv')
    raw_file = os.path.join(directory, f"{date_extracted}.csv")

    print(f'The raw file path is {raw_file}')

    # Load the CSV using your existing csv_load function
    df_raw = csv_load(raw_file)
    # Determine which columns contain the GPS coordinates (latitude and longitude)
    lat_column, lon_column = csv_select_gps_columns(
        df_raw,
        title="GPS data to compare",
        prompt="GPS data to compare"
    )
    # Filter out rows where either the latitude or longitude is missing.
    df_valid = df_raw[pd.notna(df_raw[lat_column]) & pd.notna(df_raw[lon_column])]
    # Create a list of coordinates in (latitude, longitude) format.
    coordinates = df_valid.apply(lambda row: (row[lat_column], row[lon_column]), axis=1).tolist()
    if len(coordinates) < 2:
        raise ValueError("Not enough valid GPS coordinates to draw a path.")
    # Create a FeatureGroup for the GPS coordinates path.
    gps_path_fg = folium.FeatureGroup(name="GPS Coordinates Path", show=False)
    # Draw a polyline connecting the GPS points in green.
    folium.PolyLine(
        locations=coordinates,
        color="green",
        weight=8,
        opacity=1
    ).add_to(gps_path_fg)
    gps_path_fg.add_to(m)

    # -------------------------------------------------------------------------
    # 7. Speed Path (Optional)
    # -------------------------------------------------------------------------
    if has_speed:
        speed_path_fg = folium.FeatureGroup(name="Speed Path", show=False)
        speed_min, speed_max = gdf["Speed_kmh"].min(), gdf["Speed_kmh"].max()
        norm_speed = colors.Normalize(vmin=speed_min, vmax=speed_max)
        cmap_speed = colormaps.get_cmap("turbo")

        for i in range(len(gdf) - 1):
            lat1, lon1 = get_lat_lon(gdf.iloc[i])
            lat2, lon2 = get_lat_lon(gdf.iloc[i + 1])
            speed_val = gdf.iloc[i]["Speed_kmh"]
            speed_color = colors.to_hex(cmap_speed(norm_speed(speed_val)))

            folium.PolyLine(
                [(lat1, lon1), (lat2, lon2)],
                color=speed_color,
                weight=10,
                opacity=1
            ).add_to(speed_path_fg)

        speed_path_fg.add_to(m)

        color_steps = range(int(speed_min), int(speed_max) + 1, 10)
        color_list = [colors.to_hex(cmap_speed(norm_speed(v))) for v in color_steps]
        speed_colormap = LinearColormap(
            colors=color_list,
            vmin=speed_min,
            vmax=speed_max,
            caption="Speed (km/h)"
        )
        speed_colormap.add_to(m)

    # -------------------------------------------------------------------------
    # 8. Gier Path (Optional)
    # -------------------------------------------------------------------------
    if "Gier" in gdf.columns and gdf["Gier"].notna().any():
        gier_path_fg = folium.FeatureGroup(name="Yaw Rate Path (Gier)", show=False)
        gier_min, gier_max = gdf["Gier"].min(), gdf["Gier"].max()
        gier_abs_max = max(abs(gier_min), abs(gier_max))
        gier_min, gier_max = -gier_abs_max, gier_abs_max
        gier_norm = colors.Normalize(vmin=gier_min, vmax=gier_max)
        gier_cmap = colormaps.get_cmap("coolwarm")

        for i in range(len(gdf) - 1):
            lat1, lon1 = get_lat_lon(gdf.iloc[i])
            lat2, lon2 = get_lat_lon(gdf.iloc[i + 1])
            gier_val = gdf.iloc[i]["Gier"]
            gier_color = colors.to_hex(gier_cmap(gier_norm(gier_val)))

            folium.PolyLine(
                [(lat1, lon1), (lat2, lon2)],
                color=gier_color,
                weight=10,
                opacity=1
            ).add_to(gier_path_fg)

        gier_path_fg.add_to(m)

        gier_color_steps = np.linspace(gier_min, gier_max, num=100)
        gier_color_list = [colors.to_hex(gier_cmap(gier_norm(v))) for v in gier_color_steps]
        gier_colormap = LinearColormap(
            colors=gier_color_list,
            vmin=gier_min,
            vmax=gier_max,
            caption="Yaw Rate (Gier)"
        )
        gier_colormap.add_to(m)

    # -------------------------------------------------------------------------
    # 9. Yaw Rate Path (Optional)
    # -------------------------------------------------------------------------
    yaw_rate_col = "yaw_rate_deg_s"
    if yaw_rate_col in gdf.columns and gdf[yaw_rate_col].notna().any():
        yaw_path_fg = folium.FeatureGroup(name="Yaw Rate (from heading)", show=False)
        yaw_min, yaw_max = gdf[yaw_rate_col].min(), gdf[yaw_rate_col].max()
        yaw_abs_max = max(abs(yaw_min), abs(yaw_max))
        yaw_min, yaw_max = -yaw_abs_max, yaw_abs_max
        yaw_norm = colors.Normalize(vmin=yaw_min, vmax=yaw_max)
        yaw_cmap = colormaps.get_cmap("coolwarm")

        for i in range(len(gdf) - 1):
            lat1, lon1 = get_lat_lon(gdf.iloc[i])
            lat2, lon2 = get_lat_lon(gdf.iloc[i + 1])
            yaw_val = gdf.iloc[i][yaw_rate_col]
            yaw_color = colors.to_hex(yaw_cmap(yaw_norm(yaw_val)))

            folium.PolyLine(
                [(lat1, lon1), (lat2, lon2)],
                color=yaw_color,
                weight=10,
                opacity=1
            ).add_to(yaw_path_fg)

        yaw_path_fg.add_to(m)

        yaw_color_steps = np.linspace(yaw_min, yaw_max, num=100)
        yaw_color_list = [colors.to_hex(yaw_cmap(yaw_norm(v))) for v in yaw_color_steps]
        yaw_colormap = LinearColormap(
            colors=yaw_color_list,
            vmin=yaw_min,
            vmax=yaw_max,
            caption="Yaw Rate (deg/s)"
        )
        yaw_colormap.add_to(m)

    # -------------------------------------------------------------------------
    # 10. Start & End Markers
    # -------------------------------------------------------------------------
    start_lat, start_lon = get_lat_lon(gdf.iloc[0])
    end_lat, end_lon = get_lat_lon(gdf.iloc[-1])

    folium.Marker(
        location=(start_lat, start_lon),
        popup=f"Start Point<br>Date: {day_display}",
        icon=folium.Icon(color="green")
    ).add_to(m)

    folium.Marker(
        location=(end_lat, end_lon),
        popup=f"End Point<br>Date: {day_display}",
        icon=folium.Icon(color="red")
    ).add_to(m)

    # -------------------------------------------------------------------------
    # 11. Title Box
    # -------------------------------------------------------------------------
    if "selected_smoothing_method" in df.columns:
        smoothing_method = df["selected_smoothing_method"].iloc[0]
    else:
        smoothing_method = "None"

    if "min_distance" in df.columns:
        min_distance = df["min_distance"].iloc[0]
    else:
        min_distance = "not applied"

    title_html = f"""
        <div style="position: fixed; bottom: 10px; right: 10px; width: 160px; 
                    background-color: white; z-index: 9999; font-size: 16px; 
                    border: 2px solid black; padding: 10px;">
            <b>Date:</b> {day_display}<br>
            <b>Smoothing:</b> {smoothing_method}<br>
            <b>Minimal Distance:</b> {min_distance}
        </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # -------------------------------------------------------------------------
    # 12. Time-Animated Marker (Optional)
    # -------------------------------------------------------------------------
    features = []
    for _, row in gdf.iterrows():
        lat, lon = row.geometry.y, row.geometry.x
        time_str = row["DatumZeit"].isoformat()
        speed_val = row["Speed_kmh"] if has_speed else 0.0
        popup_text = (f"<b>Time:</b> {row['DatumZeit']}<br>"
                      f"<b>Speed:</b> {speed_val:.2f} km/h")

        feature = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "time": time_str,
                "popup": popup_text,
                "style": {"color": "black", "fillColor": "black"},
                "icon": "circle"
            }
        }
        features.append(feature)

    if features:
        geojson_data = {"type": "FeatureCollection", "features": features}
        animated_marker = TimestampedGeoJson(
            data=geojson_data,
            transition_time=500,
            loop=False,
            auto_play=False,
            add_last_point=True,
            period="PT10S",
            duration="PT1S"
        )
        animated_marker.add_to(m)

    # -------------------------------------------------------------------------
    # 13. Layer Control
    # -------------------------------------------------------------------------
    folium.LayerControl(collapsed=False).add_to(m)

    # -------------------------------------------------------------------------
    # 14. Save the Map
    # -------------------------------------------------------------------------
    output_file = os.path.join(
        base_dir,
        f"map_{day_display}_smoothing_{smoothing_method}.html"
    )
    m.save(output_file)
    print(f"Map saved as '{output_file}'. Open it in your browser to view!")
