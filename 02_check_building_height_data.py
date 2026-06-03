import geopandas as gpd

f = "output/01_HoaLac_studies_area/openbuildingmap/clipped/obm_buildings_hoalac_clipped.gpkg"
gdf = gpd.read_file(f)

print("Columns:")
print(gdf.columns.tolist())

print("\nFirst rows:")
print(gdf.head())

for col in gdf.columns:
    if any(k in col.lower() for k in ["height", "level", "floor", "elev"]):
        print("\nPossible height column:", col)
        print(gdf[col].describe())
        print(gdf[col].dropna().head(20))