import requests
import json
import os

# List of German Bundesländer (states)
bundeslaender = [
    "Baden-Württemberg",
    "Bayern",
    "Berlin",
    "Brandenburg",
    "Bremen",
    "Hamburg",
    "Hessen",
    "Niedersachsen",
    "Nordrhein-Westfalen",
    "Rheinland-Pfalz",
    "Saarland",
    "Sachsen",
    "Sachsen-Anhalt",
    "Schleswig-Holstein",
    "Thüringen"
]

# Overpass query template: adjust the buffer if needed.
query_template = """
[out:json];
area["name"="{state}"]->.a;
way(area.a)["railway"="rail"]->.railways;
(
  way(around.railways:1)["bridge"="yes"];
  relation(around.railways:1)["bridge"="yes"];
);
(._;>;);
out geom;
"""

overpass_url = "https://overpass-api.de/api/interpreter"
output_dir = "bundeslaender_bridges"
os.makedirs(output_dir, exist_ok=True)

for state in bundeslaender:
    print(f"Processing {state}...")
    query = query_template.format(state=state)
    response = requests.get(overpass_url, params={"data": query})
    if response.status_code == 200:
        data = response.json()
        filename = os.path.join(output_dir, f"{state.replace(' ', '_')}_bridges_over_rail.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {state} data to {filename}")
    else:
        print(f"Error fetching data for {state}: HTTP {response.status_code}")
