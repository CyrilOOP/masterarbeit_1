#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1) CSV einlesen
    csv_file = "subsets_by_date/2024-04-02/2024-04-02_time_rollingW_planar_distance_headingDX_headingDS_yawRate_radius.csv"
    df = pd.read_csv(csv_file)

    # 2) Die Zeitspalte in richtige Datetime-Objekte umwandeln
    df["DatumZeit"] = pd.to_datetime(df["DatumZeit"])

    # 3) Spalten, die wir verwenden:
    x_col = "cumulative_distance"   # X-Achse (unten)
    y_col = "curvature"             # Y-Achse
    group_col = "steady_group"
    radius_mean_col = "steady_mean_radius"

    # --- Figure und zwei X-Achsen anlegen ---
    fig, ax_dist = plt.subplots(figsize=(220, 5))
    ax_time = ax_dist.twiny()  # Zweite X-Achse oben

    # --- Plot: Distance vs. Curvature auf der unteren Achse ---
    ax_dist.plot(df[x_col], df[y_col], label="Curvature", color="blue")
    ax_dist.set_xlabel("Distance [m]")
    ax_dist.set_ylabel("Curvature [1/m]")
    ax_dist.set_ylim(-0.005, 0.005)   # Feste Y-Grenzen
    ax_dist.grid(True)

    # --- Marker und Annotationen für steady-Gruppen ---
    steady_df = df[df[group_col] > 0]
    grouped = steady_df.groupby(group_col)
    for grp_id, grp_data in grouped:
        mid_idx = grp_data.index[len(grp_data) // 2]
        mid_row = grp_data.loc[mid_idx]
        mid_x = mid_row[x_col]
        mid_y = mid_row[y_col]
        radius_value = mid_row[radius_mean_col]
        ax_dist.plot(mid_x, mid_y, marker='o', color='red')
        if not np.isinf(radius_value):
            ax_dist.annotate(
                f"R={radius_value:.1f} m",
                xy=(mid_x, mid_y),
                xytext=(0, 15),
                textcoords="offset points",
                ha="center",
                color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=0.5)
            )

    # --- DENSE GRID: Konfiguration der oberen ZEIT-ACHSE mit dichten Tick-Markierungen ---
    # 1) Verwende dieselben x-Limits für beide Achsen.
    ax_time.set_xlim(ax_dist.get_xlim())
    x_min, x_max = ax_dist.get_xlim()

    # 2) Erstelle einen dichten Satz von Tick-Positionen (z.B. 60 Ticks)
    dense_ticks = np.linspace(x_min, x_max, num=60)
    # Setze diese Ticks auf die untere Achse (dies fügt auch vertikale Grid-Linien hinzu)
    ax_dist.set_xticks(dense_ticks)
    ax_dist.grid(which='major', axis='x', linestyle='--', color='gray')

    # 3) Interpolieren Sie für jeden dense Tick den entsprechenden Zeitwert.
    # Konvertiere DatumZeit in numerische Sekunden (als float)
    x_data = df[x_col].values
    t_numeric = df["DatumZeit"].astype(np.int64) / 1e9  # Sekunden seit Epoch

    dense_time_labels = []
    for tick in dense_ticks:
        t_val_numeric = np.interp(tick, x_data, t_numeric)
        t_val = pd.to_datetime(t_val_numeric, unit='s')
        label_str = t_val.strftime("%H:%M:%S.%f")[:-3]  # HH:MM:SS.mmm
        dense_time_labels.append(label_str)

    # 4) Setze die dichten Ticks und Labels auf die obere Achse.
    ax_time.set_xticks(dense_ticks)
    ax_time.set_xticklabels(dense_time_labels, rotation=40, ha='left')
    ax_time.set_xlabel("Time (HH:MM:SS.mmm)")

    # --- Titel, Legende und Layout ---
    ax_dist.set_title("Distance (bottom) vs. Time (top)")
    ax_dist.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
