import osmium as osm
import shapely.geometry as geom
import shapely.wkb as wkblib
import geopandas as gpd

# A global WKB factory to convert OSM data to shapely geometries
wkb_factory = osm.geom.WKBFactory()


class RailwayHandler(osm.SimpleHandler):
    """
    A custom handler to parse the OSM PBF file and extract railway ways.
    """

    def __init__(self):
        super(RailwayHandler, self).__init__()
        self.lines = []  # will store tuples of (way_id, shapely LineString, tags)

    def way(self, w):
        if 'railway' in w.tags:
            # Attempt to create a linestring geometry (some ways might be areas or incomplete)
            try:
                wkb = wkb_factory.create_linestring(w)
                if wkb is not None:
                    linestring = wkblib.loads(wkb, hex=True)
                    self.lines.append({
                        'id': w.id,
                        'geometry': linestring,
                        'railway_type': w.tags.get('railway', None)  # e.g. "rail", "subway", etc.
                    })
            except osm.InvalidLocationError:
                # Happens if node locations are missing
                pass


def main():
    # 1) Read/parse OSM PBF
    pbf_file = "germany-latest.osm.pbf"
    handler = RailwayHandler()
    handler.apply_file(pbf_file, locations=True)

    # 2) Create a GeoDataFrame from the extracted lines
    gdf = gpd.GeoDataFrame(handler.lines,
                           geometry='geometry',
                           crs="EPSG:4326")  # raw OSM data is typically EPSG:4326 (lat/lon)

    # 3) (Optional) Filter out only certain railway types
    #    e.g. keep only "rail", "subway", "light_rail", etc.
    #    If you want all railway lines, skip the filter below.
    valid_rail_types = ["rail", "light_rail"]
    gdf = gdf[gdf["railway_type"].isin(valid_rail_types)]

    # 4) Save as a Shapefile (or GeoPackage, etc.)
    out_file = "railways.shp"
    gdf.to_file(out_file)
    print(f"Saved {len(gdf)} railway lines to {out_file}")


if __name__ == "__main__":
    main()
