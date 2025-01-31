#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pandas as pd
import numpy as np

# scikit-learn für DBSCAN
from sklearn.cluster import DBSCAN
# Für geographische Distanzberechnung
from haversine import haversine
from sklearn.metrics import pairwise_distances


def main(input_csv, output_csv,
         speed_col="Geschwindigkeit in m/s",
         lat_col="GPS_lat",
         lon_col="GPS_lon",
         speed_threshold=0.5,
         eps_meters=50.0,
         min_samples=5):
    """
    Liest eine CSV ein und schreibt eine CSV ohne Stopps heraus.

    Parameter:
    -----------
    input_csv : str
        Pfad zur Eingabedatei.
    output_csv : str
        Pfad zur Ausgabedatei.
    speed_col : str
        Spaltenname für die Geschwindigkeit in m/s.
    lat_col : str
        Spaltenname für die GPS-Breitengrad.
    lon_col : str
        Spaltenname für die GPS-Längengrad.
    speed_threshold : float
        Grenzwert (m/s), unterhalb dessen wir von "Halt" bzw. "sehr langsam" ausgehen.
    eps_meters : float
        Radius in Metern für das Clustering (DBSCAN).
    min_samples : int
        Minimale Punktanzahl für einen Cluster in DBSCAN.
    """

    # 1) CSV einlesen
    df = pd.read_csv(input_csv, delimiter=',', encoding='utf-8')

    # 2) Identifiziere potenzielle Haltepunkte anhand der Geschwindigkeit
    #    (Optional: Du könntest stattdessen direkt *alle* Punkte clustern,
    #     aber hier filtern wir zuerst auf "nahe 0" Geschwindigkeit.)
    stops_df = df[df[speed_col] < speed_threshold].copy()

    if stops_df.empty:
        print("Keine Stopps gefunden (alle Geschwindigkeiten > {} m/s).".format(speed_threshold))
        # Dann schreiben wir einfach alle Punkte raus (nichts zu entfernen).
        df.to_csv(output_csv, index=False, encoding='utf-8')
        return

    # 3) DBSCAN-Clustering auf die Stopppunkte anwenden
    #    a) Extrahiere die Koordinaten (lat, lon)
    coords = stops_df[[lat_col, lon_col]].values

    #    b) DBSCAN mit Haversine-Distanz:
    #       1 Grad Lat ~ 111 km, wir arbeiten aber lieber mit exakten
    #       Entfernungen über "haversine". DBSCAN braucht dazu eine
    #       Distanz-Matrix (metric='precomputed') oder wir übergeben
    #       metric='haversine' (geht in neueren sklearn-Versionen).
    #       Hier machen wir es mit einer Distanzmatrix.

    # Umwandlung von "eps_meters" in "eps" in Kilometern
    eps_km = eps_meters / 1000.0

    # Erstelle Distanzmatrix (paarweise Haversine-Distanzen)
    dist_matrix = pairwise_distances(coords, metric=lambda x, y: haversine(x, y))

    # Führe DBSCAN durch
    db = DBSCAN(eps=eps_km, min_samples=min_samples, metric='precomputed')
    labels = db.fit_predict(dist_matrix)


    stops_df['cluster_label'] = labels  # hänge Cluster-Label an

    # 4) Welche Stopppunkte sind in "gültigen" Clustern (Label >= 0)?
    #    -> Label = -1 bedeutet "Noise"
    valid_stop_points = stops_df[stops_df['cluster_label'] != -1]

    # 5) Diese "validen Stopppunkte" möchten wir AUS dem Haupt-df entfernen,
    #    weil wir ein CSV *ohne* diese Stopps haben wollen.
    #    Wir identifizieren sie anhand eines eindeutigen Schlüssels (z.B. Index).
    #    Dazu: wir können den DataFrame-Index verwenden, wenn er eindeutig ist.
    #    Falls dein CSV aber keinen expliziten Index hat, könnte man auch
    #    Zeitstempel + Lat/Lon als "Schlüssel" nehmen (Vorsicht bei Duplikaten).

    # Hier nehmen wir an, dass der Index eindeutig ist
    stop_indices = valid_stop_points.index
    df_no_stops = df.drop(stop_indices)

    # 6) Schreibe die "ohne Stopps"-Daten zurück in eine neue CSV
    df_no_stops.to_csv(output_csv, index=False, encoding='utf-8')

    print(f"Fertig. Anzahl Punkte gesamt: {len(df)}, "
          f"entfernte Stopppunkte: {len(stop_indices)}, "
          f"verbleibend: {len(df_no_stops)}.")
    print(f"Neue CSV (ohne Stopps) gespeichert in: {output_csv}")


if __name__ == "__main__":
    input_csv_file = "2024-04-02.csv"  # Your file in the project folder
    output_csv_file = "2024-04-02_no_stops.csv"  # Output file (you don't care about the name)

    main(

        input_csv=input_csv_file,
        output_csv=output_csv_file,
        speed_col="Geschwindigkeit in m/s",
        lat_col="GPS_lat",
        lon_col="GPS_lon",
        speed_threshold=0.5,
        eps_meters=50.0,
        min_samples=5
    )
