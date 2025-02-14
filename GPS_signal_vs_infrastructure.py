import pandas as pd
import geopandas as gpd

def detect_gps_blockage(
    train_csv="train_positions.csv",
    blocking_geojson="my_infrastructure.geojson",
    output_csv="train_positions_with_blockage.csv",
    buffer_dist=10
):
    # 1) Read train positions -> GeoDataFrame
    df = pd.read_csv(train_csv)
    train_gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["GPS_lon"], df["GPS_lat"]),
        crs="EPSG:4326"
    )

    # 2) Read blocking infrastructure
    blocking_gdf = gpd.read_file(blocking_geojson)

    # Optional: if blocking is lines, buffer them by `buffer_dist` meters in a projected CRS
    #   (If the file is polygons or already buffered, you can skip this.)
    blocking_3857 = blocking_gdf.to_crs(epsg=3857)
    blocking_3857["geometry"] = blocking_3857.geometry.buffer(buffer_dist)
    blocking_gdf = blocking_3857.to_crs(epsg=4326)

    # 3) Drop any "index_right" column that might conflict with sjoin
    if "index_right" in train_gdf.columns:
        train_gdf.drop(columns=["index_right"], inplace=True)
    if "index_right" in blocking_gdf.columns:
        blocking_gdf.drop(columns=["index_right"], inplace=True)

    # 4) Spatial Join to see which points intersect blocking geometry
    joined_gdf = gpd.sjoin(train_gdf, blocking_gdf, how="left", predicate="intersects")

    # 5) Mark whether blocked or not
    joined_gdf["gps_blocked"] = ~joined_gdf["index_right"].isna()

    # 6) Save results to CSV (dropping geometry, index_right, etc.)
    # Keep your original columns plus "gps_blocked"
    # (Adjust as needed for your own column names.)
    output_cols = list(df.columns) + ["gps_blocked"]
    out_df = joined_gdf[output_cols]
    out_df.to_csv(output_csv, index=False)

    print(f"Output with GPS block status written to '{output_csv}'.")


if __name__ == "__main__":
    detect_gps_blockage(
        train_csv="subsets_by_date/2024-04-02/2024-04-02.csv",
        blocking_geojson="sachsen-anhalt_combined_infrastructure.geojson",
        output_csv="train_positions_with_blockage.csv",
        buffer_dist=10  # meters
    )
