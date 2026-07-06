import geopandas as gpd
import pandas as pd
import json

taluk_gdf = gpd.read_file("Karnataka Taluk Boundary.shp")

dist_rows  = taluk_gdf[taluk_gdf["KGISDist_1"].notna()].copy().reset_index(drop=True)
taluk_rows = taluk_gdf[taluk_gdf["KGISTalukN"].notna()].copy().reset_index(drop=True)

dist_rows["dist_code"] = (dist_rows.index + 1).astype(str).str.zfill(2)
code_to_name = dict(zip(dist_rows["dist_code"], dist_rows["KGISDist_1"]))

DISTRICT_FIX = {
    "Bagalkote":       "Bagalkot",
    "Kalaburgi":       "Kalaburagi",
    "Kolara":          "Kolar",
    "Chamarajanagara": "Chamarajanagar",
    "Bengaluru South": "Ramanagara",
}

TALUK_FIX = {
    "Alnavar":            "Alnavara",
    "Anekal":             ">nekal",
    "Babaleshwar":        "Babaleshwara",
    "Bagalkote":          "Bagalkot",
    "Bangalore-East":     "Bengaluru-East",
    "Bangalore-North":    "Bengaluru-North",
    "Bangalore-South":    "Bengaluru-South",
    "Bangarpet":          "Bangarapete",
    "Basavanabagewadi":   "Basavana Bagevadi",
    "Bilagi":             "Bilgi",
    "Brahmavara":         "Bramhavara",
    "Chadchan":           "Chadachana",
    "Chamarajanagara":    "Chamarajanagar",
    "Chikkamagaluru":     "Chikkmagaluru",
    "Chitguppa":          "Chittaguppa",
    "Chittapur":          "Chitapur",
    "Dandeli":            "Dandelli",
    "Doddaballapura":     "Dod Ballapur",
    "Guledgudda":         "Guledagudda",
    "Gurmitakal":         "Gurumithakala",
    "Hadagali":           "Huvina Hadagali",
    "Hubballi Urban":     "Hubballi",
    "Hulsoor":            "Hulasur",
    "Hunasagi":           "Hunisigi",
    "Jamakhandi":         "Jamkhandi",
    "Jewargi":            "Jevargi",
    "Joida":              "Supa",
    "Kagwad":             "Kagavada",
    "Kalaburgi":          "Kalaburagi",
    "Kanakpura":          "Kanakapura",
    "Kolar Gold Field":   "K.G.F",
    "Kolara":             "Kolar",
    "Kolhar":             "Kolhara",
    "Kollegal":           "Kollegala",
    "Kotturu":            "Kutturu",
    "Krishnarajpet":      "Krishnarajapete",
    "Kukanoor":           "Kukanuru",
    "Kushalnagar":        "Kushalanagara",
    "Lingasuguru":        "Lingasugur",
    "Mangaluru":          "Mangalore",
    "Moodubidire":        "Mudabidri",
    "Mudalgi":            "Mudalagi",
    "Muddebihala":        "Muddebihal",
    "Navalagund":         "Navalgund",
    "Pandavpura":         "Pandavapura",
    "Puttur":             "Putturu",
    "Rabakavi-Banahatti": "Rabkavi Banhatti",
    "Ramadurg":           "Ramadurga",
    "Rattihalli":         "Ratteehalli",
    "Shahpur":            "Shahapur",
    "Sindhanuru":         "Sindhanur",
    "Siruguppa":          "Siraguppa",
    "Sirwar":             "Sirivara",
    "Somavarapete":       "Somvarpet",
    "Sonduru":            "Sanduru",
    "Srinivaspura":       "Sr\\nivasapura",
    "Srirangapatna":      "Sr\\rangapatna",
    "Sullia":             "Sulya",
    "T.Narasipura":       "T.Naras\\pura",
    "Talikoti":           "Talikote",
    "Thirthahalli":       "Th\\rthahalli",
    "Tumakuru":           "Tumkuru",
    "Wadagera":           "Vadagera",
    "Yalandur":           "Yelanduru",
}

taluk_rows["dist_code"] = taluk_rows["KGISTalukC"].astype(str).str[:2]
taluk_rows["DISTRICT"]  = taluk_rows["dist_code"].map(code_to_name).replace(DISTRICT_FIX)
taluk_rows["SUB_DIST"]  = taluk_rows["KGISTalukN"].astype(str).str.strip().replace(TALUK_FIX)
dist_rows["DISTRICT"]   = dist_rows["KGISDist_1"].replace(DISTRICT_FIX)

dist_rows  = dist_rows.to_crs(epsg=4326)
taluk_rows = taluk_rows.to_crs(epsg=4326)

district_out = dist_rows[["DISTRICT", "geometry"]].dissolve(by="DISTRICT").reset_index()
district_out["geometry"] = district_out["geometry"].simplify(0.001, preserve_topology=True)
district_out.to_file("district_boundaries.geojson", driver="GeoJSON")
print(f"Districts: {len(district_out)}")

taluk_out = taluk_rows[["DISTRICT", "SUB_DIST", "geometry"]].copy()
taluk_out = taluk_out[taluk_out["DISTRICT"].notna()]
taluk_out["geometry"] = taluk_out["geometry"].simplify(0.001, preserve_topology=True)
taluk_out.to_file("taluk_boundaries.geojson", driver="GeoJSON")
print(f"Taluks: {len(taluk_out)}")

village_df = pd.read_excel("combined_village_data.xlsx", engine="openpyxl")
village_df["SUB_DIST"] = village_df["SUB_DIST"].astype(str).str.strip()

with open("taluk_boundaries.geojson") as f:
    t = json.load(f)

geojson_taluks = set(f["properties"]["SUB_DIST"] for f in t["features"])
excel_taluks   = set(village_df["SUB_DIST"].unique())

print("\nGeoJSON not in Excel:", sorted(geojson_taluks - excel_taluks))
print("Excel not in GeoJSON:", sorted(excel_taluks - geojson_taluks))
print("Matched:", len(geojson_taluks & excel_taluks))