import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter


def moving_average(data, window_size):
    """ Simple moving average for smoothing GPS positions when stationary."""
    return np.mean(data[-window_size:], axis=0)


def initialize_kalman(dt, process_noise, measurement_noise):
    """ Initialize the Kalman filter."""
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, dt, 0],  # x position update
                     [0, 1, 0, dt],  # y position update
                     [0, 0, 1, 0],  # x velocity
                     [0, 0, 0, 1]])  # y velocity
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    kf.Q = np.eye(4) * process_noise  # Process noise covariance
    kf.R = np.eye(2) * measurement_noise  # Measurement noise covariance
    kf.P *= 100  # Large initial uncertainty
    return kf


def reduce_and_smooth_gps(df, config):
    """ Process GPS data to remove stops and smooth with Kalman filtering using hysteresis."""
    SPEED_MOVE = config['speed_move']  # Speed to consider as moving
    SPEED_STOP = config['speed_stop']  # Speed to consider as stopped
    TIME_WINDOW = config['time_window']
    DT = config['dt']
    PROCESS_NOISE = config['process_noise']
    MEASUREMENT_NOISE = config['measurement_noise']
    MOVE_DURATION = config['move_duration']  # Time to confirm movement
    STOP_DURATION = config['stop_duration']  # Time to confirm stop

    kf = initialize_kalman(DT, PROCESS_NOISE, MEASUREMENT_NOISE)
    is_moving = False
    move_timer, stop_timer = 0, 0
    filtered_positions = []

    for i, row in df.iterrows():
        current_pos = np.array([row['lat'], row['lon']])

        if i == 0:
            kf.x = np.array([row['lat'], row['lon'], 0, 0])
            filtered_positions.append(current_pos)
            continue

        # Hysteresis for movement detection
        if row['speed'] > SPEED_MOVE:
            move_timer += 1
            stop_timer = 0
        elif row['speed'] < SPEED_STOP:
            stop_timer += 1
            move_timer = 0

        # Confirm moving or stopping state after sustained duration
        if move_timer >= MOVE_DURATION:
            is_moving = True
        elif stop_timer >= STOP_DURATION:
            is_moving = False

        if is_moving:
            kf.predict()
            kf.update(current_pos)
            filtered_pos = kf.x[:2]  # Extract (x, y) position
        else:
            filtered_pos = moving_average(df[['lat', 'lon']].values[:i + 1], window_size=TIME_WINDOW)

        filtered_positions.append(filtered_pos)

    filtered_df = pd.DataFrame(filtered_positions, columns=['filtered_lat', 'filtered_lon'])
    df = df[df['speed'] > SPEED_STOP].reset_index(drop=True)
    df = pd.concat([df, filtered_df], axis=1)

    return df
