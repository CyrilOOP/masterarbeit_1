import pandas as pd
from pykalman import KalmanFilter

def smooth_gps_yaw_rate(df: pd.DataFrame, config: Dict[str, str]) -> pd.DataFrame:
    """
    Apply a Kalman filter to smooth yaw rate values computed from GPS heading.

    Args:
        df: The DataFrame containing GPS-derived yaw rate.
        config: Configuration dictionary with column names.

    Returns:
        The modified DataFrame with an additional column 'yaw_rate_smooth_gps'.
    """
    yaw_col = config.get("yaw_col_for_kalman")  # GPS-based yaw rate column

    if yaw_col not in df.columns:
        raise ValueError(f"Required column '{yaw_col}' not found in DataFrame.")

    # Extract noisy yaw rate from GPS
    measurements = df[yaw_col].values

    # Kalman Filter Setup (assuming constant yaw rate model)
    kf = KalmanFilter(
        initial_state_mean=[measurements[0]],  # Start with first yaw rate value
        initial_state_covariance=[1],  # Initial uncertainty
        transition_matrices=[1],  # Yaw rate follows smooth transitions
        observation_matrices=[1],  # Direct observation model
        transition_covariance=[0.1],  # Process noise (adjustable)
        observation_covariance=[2]  # Measurement noise (adjustable)
    )

    # Apply Kalman filter
    smoothed_state_means, _ = kf.filter(measurements.reshape(-1, 1))

    # Store smoothed yaw rate in DataFrame
    df["yaw_rate_smooth_gps"] = smoothed_state_means[:, 0]

    return df
