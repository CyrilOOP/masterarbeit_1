import os
import numpy as np
import pandas as pd
import folium
from folium.plugins import TimestampedGeoJson
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from branca.colormap import LinearColormap
from csv_tools import csv_load
import tkinter as tk
from tkinter import messagebox

def generate_map_from_csv(subset_full_path: str) -> None:
    """
    Generates an interactive Folium map from CSV data containing GPS information.

    The function creates a map with multiple layers:
      - A uniform blue polyline showing the processed GPS path.
      - An additional raw GPS path with segments colored by "Gier" (yaw rate).
      - (Optional) Speed-based colored paths if speed data is available.
      - (Optional) Yaw Rate (Gier) and Yaw Rate (from heading) colored paths.
      - Start and end markers, a title box, and a time-animated marker.
      - Toggleable overlay tile layers.

    Args:
        subset_full_path (str): Path to the processed CSV file.

    Raises:
        ValueError: If required data is missing or if no valid GPS coordinates are found.
    """
    # =========================================================================
    # 1. Load Processed CSV Data
    # =========================================================================
    base_dir = os.path.dirname(subset_full_path)
    df = pd.read_csv(subset_full_path, parse_dates=["DatumZeit"])
    if df.empty:
        raise ValueError("No data found in the CSV. Cannot generate map.")

    # =========================================================================
    # 2. Extract GPS Coordinates (with optional smoothing)
    # =========================================================================
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
                    f"Columns {lat_col} and {lon_col} not found. Check your smoothing configuration."
                )
    else:
        lat_vals = df["GPS_lat"]
        lon_vals = df["GPS_lon"]
        print("Column 'selected_smoothing_method' missing. Using default columns: GPS_lat and GPS_lon")

    # Ensure we have valid lat/lon (drop rows with NaN)
    df = df.dropna(subset=[lat_vals.name, lon_vals.name])
    if df.empty:
        raise ValueError("All GPS points are NaN. Cannot generate map.")

    # Use these updated lat/lon Series (after dropna)
    lat_vals = df[lat_vals.name]
    lon_vals = df[lon_vals.name]

    # For labeling date in markers/title, etc.
    day_display = df["DatumZeit"].iloc[0].date()

    # =========================================================================
    # 4. Initialize Folium Map with No Base Tiles
    # =========================================================================
    start_lat = lat_vals.iloc[0]
    start_lon = lon_vals.iloc[0]
    m = folium.Map(location=[start_lat, start_lon], zoom_start=14, tiles=None)

    # =========================================================================
    # 5. Add Overlay Tile Layers (Toggleable)
    # =========================================================================
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

#=================================================================================
#Ask if raw orignal path is wanted
#==================================================================================
    # Create a hidden root window
    root = tk.Tk()
    root.withdraw()

    # Ask the user
    run_raw = messagebox.askyesno(
        title="Include Raw Original GPS?",
        message="Do you want to include the raw original GPS path (with Gier coloring)?\nMoving marker won't be available)"

    )

    # Destroy the hidden root window once we have the answer
    root.destroy()
    color_scheme_gier = {}

    if run_raw:
        # =================================================================================
        # 7. Draw Raw untouched original GPS Path with "Gier" (Yaw Rate) Coloring if wanted
        # =================================================================================
        raw_filename = os.path.basename(subset_full_path)
        raw_date = raw_filename.split('_')[0]  # Expects filename starting with YYYY-MM-DD
        raw_directory = os.path.dirname(subset_full_path)
        raw_csv_file_path = os.path.join(raw_directory, f"{raw_date}.csv")
        print(f"Raw CSV file path: {raw_csv_file_path}")

        # --- Load Raw Data and Validate Required Columns ---
        raw_df = csv_load(raw_csv_file_path)

        # Ensure the raw data contains the necessary columns
        for col in ["GPS_lat", "GPS_lon", "Gier"]:
            if col not in raw_df.columns:
                raise ValueError(f"Raw data is missing the required column: {col}")

        # Drop rows where any of the required values are missing
        raw_df = raw_df.dropna(subset=["GPS_lat", "GPS_lon", "Gier"])
        if len(raw_df) < 2:
            raise ValueError("Not enough valid raw GPS data to draw a path.")

        # Convert the "Gier" column to numeric
        raw_df["Gier"] = pd.to_numeric(raw_df["Gier"], errors="coerce")
        raw_df = raw_df.dropna(subset=["Gier"])

        # --- Set Up a Centered Colormap for "Gier" ---
        gier_min_raw = raw_df["Gier"].min()
        gier_max_raw = raw_df["Gier"].max()
        max_abs_raw = max(abs(gier_min_raw), abs(gier_max_raw))
        gier_min_raw, gier_max_raw = -max_abs_raw, max_abs_raw

        norm_gier_raw = mcolors.Normalize(vmin=gier_min_raw, vmax=gier_max_raw)
        gier_cmap = plt.get_cmap("RdBu")

        gier_color_steps = np.linspace(gier_min_raw, gier_max_raw, num=100)
        gier_color_list = [
            mcolors.to_hex(gier_cmap(norm_gier_raw(val)))
            for val in gier_color_steps
        ]
        gier_colormap = LinearColormap(
            colors=gier_color_list,
            vmin=gier_min_raw,
            vmax=gier_max_raw,
            caption="Yaw Rate (Gier)"
        )
        gier_colormap.add_to(m)

        # --- Store the Gier colormap settings ---
        color_scheme_gier = {
            "norm": norm_gier_raw,
            "cmap": gier_cmap,
            "vmin": gier_min_raw,
            "vmax": gier_max_raw
        }

        # --- Draw the Raw GPS Path Colored by "Gier" ---
        raw_path_fg = folium.FeatureGroup(name="Raw Original GPS with Gier as Color", show=False)
        for i in range(len(raw_df) - 1):
            lat1 = raw_df.iloc[i]["GPS_lat"]
            lon1 = raw_df.iloc[i]["GPS_lon"]
            lat2 = raw_df.iloc[i + 1]["GPS_lat"]
            lon2 = raw_df.iloc[i + 1]["GPS_lon"]
            gier_value = raw_df.iloc[i]["Gier"]
            segment_color = mcolors.to_hex(gier_cmap(norm_gier_raw(gier_value)))
            folium.PolyLine(
                locations=[(lat1, lon1), (lat2, lon2)],
                color=segment_color,
                weight=24,
                opacity=1
            ).add_to(raw_path_fg)
        raw_path_fg.add_to(m)

        # =========================================================================
        # 13. Add Time-Animated Marker (Using Raw Data)
        # =========================================================================
        # Ensure that 'DatumZeit' is parsed as datetime in the raw data.
        if "DatumZeit" in raw_df.columns:
            raw_df["DatumZeit"] = pd.to_datetime(raw_df["DatumZeit"], errors="coerce")
        else:
            raise ValueError("Raw data is missing the required 'DatumZeit' column for animation.")

        # Remove rows with invalid or missing time data
        raw_df = raw_df.dropna(subset=["DatumZeit"])

        features = []
        for _, row in raw_df.iterrows():
            lat = row["GPS_lat"]
            lon = row["GPS_lon"]
            time_val = row["DatumZeit"]
            time_str = time_val.isoformat()

            popup_text = (
                f"<b>Time:</b> {time_val}<br>"
                f"<b>Gier:</b> {row['Gier']}"
            )

            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "time": time_str,
                    "popup": popup_text,
                    "style": {"color": "black", "fillColor": "black"},
                    "icon": "circle"
                }
            })

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

    else:
        print("Skipping the raw original GPS path to save time...")
    # =========================================================================
    # 8. Draw Speed Path (if Speed data is available)
    # =========================================================================
    # Convert speed from m/s to km/h if available
    if "Geschwindigkeit in m/s" in df.columns:
        df["Speed_kmh"] = df["Geschwindigkeit in m/s"] * 3.6
    has_speed = "Speed_kmh" in df.columns and df["Speed_kmh"].notna().any()

    if has_speed:
        speed_path_fg = folium.FeatureGroup(name="Speed Path", show=False)
        speed_min = df["Speed_kmh"].min()
        speed_max = df["Speed_kmh"].max()
        norm_speed = mcolors.Normalize(vmin=speed_min, vmax=speed_max)
        cmap_speed = plt.get_cmap("turbo")

        for i in range(len(df) - 1):
            # Use the lat/lon from our selected columns
            lat1 = lat_vals.iloc[i]
            lon1 = lon_vals.iloc[i]
            lat2 = lat_vals.iloc[i + 1]
            lon2 = lon_vals.iloc[i + 1]
            speed_val = df.iloc[i]["Speed_kmh"]
            speed_color = mcolors.to_hex(cmap_speed(norm_speed(speed_val)))
            folium.PolyLine(
                locations=[(lat1, lon1), (lat2, lon2)],
                color=speed_color,
                weight=10,
                opacity=1
            ).add_to(speed_path_fg)
        speed_path_fg.add_to(m)

        # Create and add a speed colormap legend
        color_steps = range(int(speed_min), int(speed_max) + 1, 10)
        color_list = [mcolors.to_hex(cmap_speed(norm_speed(v))) for v in color_steps]
        speed_colormap = LinearColormap(
            colors=color_list,
            vmin=speed_min,
            vmax=speed_max,
            caption="Speed (km/h)"
        )
        speed_colormap.add_to(m)

    # # =========================================================================
    # # 9. Draw Gier Path (Processed Data) if Available
    # # =========================================================================
    # if "Gier" in df.columns and df["Gier"].notna().any():
    #     gier_path_fg = folium.FeatureGroup(name="Yaw Rate Path (Gier)", show=False)
    #     gier_min_proc = df["Gier"].min()
    #     gier_max_proc = df["Gier"].max()
    #     gier_abs_max_proc = max(abs(gier_min_proc), abs(gier_max_proc))
    #     proc_gier_min, proc_gier_max = -gier_abs_max_proc, gier_abs_max_proc
    #     proc_gier_norm = mcolors.Normalize(vmin=proc_gier_min, vmax=proc_gier_max)
    #     proc_gier_cmap = plt.get_cmap("bwr")
    #
    #     for i in range(len(df) - 1):
    #         lat1 = lat_vals.iloc[i]
    #         lon1 = lon_vals.iloc[i]
    #         lat2 = lat_vals.iloc[i + 1]
    #         lon2 = lon_vals.iloc[i + 1]
    #         gier_val = df.iloc[i]["Gier"]
    #         gier_color = mcolors.to_hex(proc_gier_cmap(proc_gier_norm(gier_val)))
    #         folium.PolyLine(
    #             locations=[(lat1, lon1), (lat2, lon2)],
    #             color=gier_color,
    #             weight=10,
    #             opacity=1
    #         ).add_to(gier_path_fg)
    #     gier_path_fg.add_to(m)
    #
    #     proc_gier_color_steps = np.linspace(proc_gier_min, proc_gier_max, num=100)
    #     proc_gier_color_list = [mcolors.to_hex(proc_gier_cmap(proc_gier_norm(v))) for v in proc_gier_color_steps]
    #     proc_gier_colormap = LinearColormap(
    #         colors=proc_gier_color_list,
    #         vmin=proc_gier_min,
    #         vmax=proc_gier_max,
    #         caption="Yaw Rate (Gier)"
    #     )
    #     proc_gier_colormap.add_to(m)

    # =================================================================================
    # 10. Draw Yaw Rate Path (From Heading) if Available
    # =================================================================================
    yaw_rate_col = "yaw_rate_deg_s"
    if yaw_rate_col in df.columns and df[yaw_rate_col].notna().any():
        print("doing it")
        yaw_path_fg = folium.FeatureGroup(name="Yaw Rate from GPS", show=False)

        if color_scheme_gier:
            # Use Gier's color scheme
            yaw_norm = color_scheme_gier["norm"]
            yaw_cmap = color_scheme_gier["cmap"]
            yaw_vmin = color_scheme_gier["vmin"]
            yaw_vmax = color_scheme_gier["vmax"]
        else:
            # Compute Yaw Rate scale separately
            yaw_min = df[yaw_rate_col].min()
            yaw_max = df[yaw_rate_col].max()
            yaw_abs_max = max(abs(yaw_min), abs(yaw_max))
            yaw_vmin, yaw_vmax = -yaw_abs_max, yaw_abs_max
            yaw_norm = mcolors.Normalize(vmin=yaw_vmin, vmax=yaw_vmax)
            yaw_cmap = plt.get_cmap("RdBu")

        for i in range(len(df) - 1):
            lat1 = df.iloc[i]["GPS_lat"]
            lon1 = df.iloc[i]["GPS_lon"]
            lat2 = df.iloc[i + 1]["GPS_lat"]
            lon2 = df.iloc[i + 1]["GPS_lon"]
            yaw_value = df.iloc[i][yaw_rate_col]

            # --- Assign black color if the value is outside the defined range ---
            if yaw_value < yaw_vmin or yaw_value > yaw_vmax:
                segment_color = "#FFFF00"  # Black for out-of-range values
            else:
                segment_color = mcolors.to_hex(yaw_cmap(yaw_norm(yaw_value)))

            folium.PolyLine(
                locations=[(lat1, lon1), (lat2, lon2)],
                color=segment_color,
                weight=10,
                opacity=1
            ).add_to(yaw_path_fg)

        yaw_path_fg.add_to(m)



    # =========================================================================
    # 11. Add Start & End Markers
    # =========================================================================
    start_lat = lat_vals.iloc[0]
    start_lon = lon_vals.iloc[0]
    end_lat = lat_vals.iloc[-1]
    end_lon = lon_vals.iloc[-1]
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

    # =========================================================================
    # 12. Add Title Box Overlay
    # =========================================================================
    smoothing_method = df["selected_smoothing_method"].iloc[0] if "selected_smoothing_method" in df.columns else "None"
    min_distance = df["min_distance"].iloc[0] if "min_distance" in df.columns else "not applied"
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



    # =========================================================================
    # 14. Add Layer Control and Save the Map
    # =========================================================================
    folium.LayerControl(collapsed=False).add_to(m)
    output_file = os.path.join(
        base_dir,
        f"map_{day_display}_smoothing_{smoothing_method}.html"
    )
    m.save(output_file)
    print(f"Map saved as '{output_file}'. Open it in your browser to view!")
