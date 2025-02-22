import pyproj

from csv_tools import csv_select_gps_columns


def data_convert_to_planar(df: pd.DataFrame, config: Dict[str, str]) -> pd.DataFrame:
    """
    Convert latitude and longitude to planar coordinates (UTM).
    Uses the helper to select which GPS columns to use.

    Args:
        df: The input DataFrame containing GPS data.
        config: A configuration dictionary.

    Returns:
        The DataFrame with added planar coordinates (x, y) and a column 'selected_smoothing_method'.
    """
    # Use helper to select GPS columns.
    lat_input, lon_input = csv_select_gps_columns(
        df,
        title="Select GPS Data for Planar Conversion",
        prompt="Select the GPS data to use for planar conversion:"
    )
    print(f"Using input columns: {lat_input} and {lon_input}")

    # **Dynamically determine UTM zone based on longitude**
    utm_zone = int((df[lon_input].mean() + 180) / 6) + 1
    is_northern = df[lat_input].mean() >= 0  # True if in northern hemisphere
    epsg_code = f"EPSG:{32600 + utm_zone if is_northern else 32700 + utm_zone}"

    # **Use pyproj Transformer**
    transformer = pyproj.Transformer.from_crs("EPSG:4326", epsg_code, always_xy=True)

    # Convert coordinates
    df["x"], df["y"] = transformer.transform(df[lon_input].values, df[lat_input].values)

    return df

