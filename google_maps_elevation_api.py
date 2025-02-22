import requests
import pandas as pd
import time
import concurrent.futures  # For parallel processing




def data_get_elevation(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    # === Configuration Variables ===
    API_KEY = config['API_KEY']
    ELEVATION_COLUMN = config['ELEVATIONCOLUMN']
    API_URL = config['API_URL']
    BATCH_SIZE = config["BATCH_SIZE"]
    THREADS = config["THREADS"]

    # 3. Identify lat/lon columns (stubbed function you'd have in your code)
    lat_col, lon_col = csv_select_gps_columns(
        df,
        title="Select GPS Data for elevation",
        prompt="Select the GPS data to use as input for elevation:"
    )
    print(f"Using GPS columns: {lat_col} and {lon_col}")

    coords = list(zip(df[lat_col], df[lon_col]))
    total_rows = len(coords)


    # === Process in Batches with Multi-Threading ===
    elevations = [None] * total_rows  # Placeholder for elevation values
    batch_indices = [list(range(i, min(i + BATCH_SIZE, total_rows))) for i in range(0, total_rows, BATCH_SIZE)]
    batches = [coords[i[0]:i[-1] + 1] for i in batch_indices]  # Split into chunks

    print(f"🚀 Processing {len(batches)} batches with {THREADS} parallel threads...")

    # === Function to Get Elevation for a Batch of Coordinates ===
    def get_elevation_batch(coords_batch):
        locations_str = "|".join([f"{lat},{lon}" for lat, lon in coords_batch])
        url = f"{API_URL}?locations={locations_str}&key={API_KEY}"

        try:
            response = requests.get(url).json()
            if response.get('status') == 'OK':
                return [result["elevation"] for result in response["results"]]
            else:
                print(f"❌ API Error: {response.get('status')}")
                return [None] * len(coords_batch)
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return [None] * len(coords_batch)

    # Multithreaded execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as executor:
        results = list(executor.map(get_elevation_batch, batches))

    # Merge results back into the elevation list
    for batch_idx, batch_elevations in enumerate(results):
        for i, elevation in zip(batch_indices[batch_idx], batch_elevations):
            elevations[i] = elevation

    # === Add Elevation to DataFrame ===
    df[ELEVATION_COLUMN] = elevations
    return df

