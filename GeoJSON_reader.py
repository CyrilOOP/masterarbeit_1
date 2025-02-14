import os
import folium
import geopandas as gpd
import tkinter as tk
from tkinter import filedialog


def list_geojson_files():
    """Lists all GeoJSON files in the project directory."""
    files = [f for f in os.listdir() if f.endswith('.geojson')]
    return files


def select_geojson_file():
    """Opens a dialog to select a GeoJSON file from the project folder."""
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title="Select a GeoJSON File", filetypes=[("GeoJSON files", "*.geojson")])
    return file_path

def create_map(geojson_path):
    """Creates an interactive map from a GeoJSON file."""
    if not geojson_path:
        print("No file selected. Exiting.")
        return

    print(f"Loading: {geojson_path}")

    # Load GeoJSON data using GeoPandas
    gdf = gpd.read_file(geojson_path)

    # Reproject to a projected CRS for accurate centroid calculation.
    # Here, EPSG:3857 (Web Mercator) is used for demonstration,
    # but you might choose a projection that better fits your data's region.
    gdf_projected = gdf.to_crs(epsg=3857)

    # Compute centroid in the projected CRS
    centroid_projected = gdf_projected.geometry.centroid

    # To use the centroid for Folium (which requires lat/lon), reproject back to EPSG:4326.
    centroid = centroid_projected.to_crs(epsg=4326)

    # Compute the mean of the centroids' coordinates to set the map center.
    map_center = [centroid.y.mean(), centroid.x.mean()]

    # Create Folium map with the calculated center.
    m = folium.Map(location=map_center, zoom_start=12, tiles="OpenStreetMap")

    # Add GeoJSON layer to the map (Folium expects data in EPSG:4326)
    folium.GeoJson(geojson_path, name="GeoJSON Layer").add_to(m)

    # Save and open the map
    output_html = "map.html"
    m.save(output_html)
    print(f"Map saved as {output_html}. Open it in a browser.")

    # Open the map in the default web browser
    os.system(f"start {output_html}" if os.name == "nt" else f"xdg-open {output_html}")


if __name__ == "__main__":
    # Get available GeoJSON files
    available_files = list_geojson_files()

    if not available_files:
        print("No GeoJSON files found in the project directory.")
    else:
        # Ask user to select a file
        selected_file = select_geojson_file()

        # Create and display the map
        create_map(selected_file)
