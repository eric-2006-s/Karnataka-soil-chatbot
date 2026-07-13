import asyncio
import sys
if sys.platform == "win32" and sys.version_info >= (3, 14):
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

import os
import json
import hashlib
import requests
import streamlit as st
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from groq import Groq
import edge_tts
from deep_translator import GoogleTranslator
from streamlit_mic_recorder import mic_recorder
import tempfile
import io
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm

# Map imports
import folium
from streamlit_folium import st_folium
import matplotlib as mpl
import matplotlib.colors as mcolors

# ── API key (from Streamlit secrets, never hardcoded) ──────────
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ── Paths ────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
CSV_URL = "https://huggingface.co/datasets/ricu9656/karnataka-soil-data/resolve/main/Export_Output.csv"
CSV_PATH = os.path.join(BASE, "Export_Output.csv")
VILLAGE_GEOJSON_PATH = os.path.join(BASE, "village_boundaries_simplified_v3.geojson")
DISTRICT_GEOJSON_PATH = os.path.join(BASE, "district_boundaries.geojson")
TALUK_GEOJSON_PATH = os.path.join(BASE, "taluk_boundaries.geojson")

# ── Data loading ──────────────────────────────────────────────
@st.cache_data
def load_village_data():
    df = pd.read_excel(os.path.join(BASE, "combined_village_data.xlsx"), engine="openpyxl")
    df = df[df['KGISVill_2'].notna() & (df['KGISVill_2'].str.strip() != '')]
    df = df[df['latitude'].notna() & df['longitude'].notna()]
    for col in ["DISTRICT", "SUB_DIST", "KGISVill_2"]:
        df[col] = df[col].astype(str).str.strip()

    # ── memory: DISTRICT/SUB_DIST repeat heavily across 45k rows → category
    # dtype stores each unique string once instead of once per row. KGISVill_2
    # is left as-is: too high-cardinality (near-unique) for category to help.
    for col in ["DISTRICT", "SUB_DIST"]:
        df[col] = df[col].astype("category")
    # latitude/longitude stay float64 on purpose: numpy.float64 subclasses
    # Python's built-in float so json.dumps handles it fine, but float32 does
    # NOT — and lat/lon end up JSON-serialized via st.map/deck.gl. Downcasting
    # them crashes st.map with "Object of type float32 is not JSON serializable".
    float_cols = [c for c in df.select_dtypes(include="float64").columns
                  if c not in ("latitude", "longitude")]
    df[float_cols] = df[float_cols].astype("float32")

    return df

@st.cache_data
def load_csv_data():
    if not os.path.exists(CSV_PATH):
        with st.spinner("Downloading soil dataset (first run only)..."):
            r = requests.get(CSV_URL)
            r.raise_for_status()
            with open(CSV_PATH, "wb") as f:
                f.write(r.content)

    df = pd.read_csv(CSV_PATH)
    df = df.rename(columns={
        "Depth":     "DEPTH",
        "pH":        "PH",
        "Texture":   "TEXTURE",
        "Longitude": "longitude",
    })
    df = df[df['latitude'].notna() & df['longitude'].notna()]

    # ── memory: this is the dense HuggingFace raster-style CSV — float64 by
    # default. float32 halves its footprint with no meaningful precision loss
    # for IDW interpolation over soil readings. latitude/longitude excluded —
    # same reason as load_village_data: float32 isn't JSON-serializable,
    # unlike float64, and coordinate columns are the ones that tend to end up
    # passed to st.map/folium/JSON paths.
    float_cols = [c for c in df.select_dtypes(include="float64").columns
                  if c not in ("latitude", "longitude")]
    df[float_cols] = df[float_cols].astype("float32")

    return df

@st.cache_data
def load_village_boundaries():
    """Village polygon boundaries, joined to DISTRICT/SUB_DIST via merger.py output.

    Some districts (e.g. Uttara Kannada) ship with a blank KGISVill_2 in the source
    geojson for every polygon — the original merge never attached village names for
    that district. We patch those in here: for each polygon with a missing name, take
    its centroid and match it to the nearest point in combined_village_data.xlsx
    within the same DISTRICT + SUB_DIST (which does have correct names).
    """
    gdf = gpd.read_file(VILLAGE_GEOJSON_PATH)
    gdf = gdf[gdf["DISTRICT"].notna()]  # drop unjoinable shapefile rows — no soil data possible
    gdf["DISTRICT"] = gdf["DISTRICT"].astype(str).str.strip()
    gdf["SUB_DIST"] = gdf["SUB_DIST"].astype(str).str.strip()
    gdf["KGISVill_2"] = gdf["KGISVill_2"].astype(str).str.strip()
    gdf.loc[gdf["KGISVill_2"].isin(["None", "nan", ""]), "KGISVill_2"] = None

    missing_mask = gdf["KGISVill_2"].isna()
    if missing_mask.any():
        vdf = load_village_data()  # has correct names + lat/lon, cached
        filled_names = []
        for idx in gdf.loc[missing_mask].index:
            row = gdf.loc[idx]
            centroid = row.geometry.centroid
            candidates = vdf[
                (vdf["DISTRICT"] == row["DISTRICT"]) &
                (vdf["SUB_DIST"] == row["SUB_DIST"])
            ]
            if candidates.empty:
                filled_names.append(None)
                continue
            dists = (candidates["latitude"] - centroid.y) ** 2 + (candidates["longitude"] - centroid.x) ** 2
            nearest_name = candidates.loc[dists.idxmin(), "KGISVill_2"]
            filled_names.append(nearest_name)
        gdf.loc[missing_mask, "KGISVill_2"] = filled_names
        gdf["KGISVill_2"] = gdf["KGISVill_2"].fillna("Unknown")

    return gdf

@st.cache_data
def get_district_boundaries():
    gdf = gpd.read_file(DISTRICT_GEOJSON_PATH)
    gdf["DISTRICT"] = gdf["DISTRICT"].astype(str).str.strip()
    return gdf

@st.cache_data
def get_taluk_boundaries(district):
    gdf = gpd.read_file(TALUK_GEOJSON_PATH)
    gdf["DISTRICT"] = gdf["DISTRICT"].astype(str).str.strip()
    gdf["SUB_DIST"] = gdf["SUB_DIST"].astype(str).str.strip()
    return gdf[gdf["DISTRICT"] == district]

@st.cache_resource
def get_village_tree(_df):
    return cKDTree(_df[["latitude", "longitude"]].values)

@st.cache_resource
def get_csv_tree(_df):
    return cKDTree(_df[["latitude", "longitude"]].values)

village_df = load_village_data()
csv_df     = load_csv_data()
village_tree = get_village_tree(village_df)
csv_tree     = get_csv_tree(csv_df)
# village_boundaries_gdf is NOT loaded here — it's the heaviest asset (full-res
# village polygons for the whole state) and most sessions never touch Map mode's
# village level. Loaded lazily below, right where it's first used. Still
# @st.cache_data, so this only costs time once per server process, not once per
# session — every session after the first hits the cache instantly.

def idw_estimate(tree, df, lat, lon, columns, k=4, power=2, max_dist_deg=0.01):
    distances, indices = tree.query([[lat, lon]], k=k)
    distances = distances[0]
    indices = indices[0]

    if distances[0] > max_dist_deg:
        return None

    if distances[0] < 1e-12:
        return df.iloc[indices[0]][columns].astype(float)

    weights = 1.0 / (distances ** power)
    weights /= weights.sum()

    result = {}
    for col in columns:
        values = df.iloc[indices][col].astype(float).values
        result[col] = float((values * weights).sum())

    return pd.Series(result)

# ── Interpretation helpers ─────────────────────────────────────
def soc_interp(v):
    v = float(v)
    if v < 0.5:    return f"{v:.2f}% — Low. Needs organic amendments."
    elif v < 0.75: return f"{v:.2f}% — Moderate. Moderately fertile."
    else:           return f"{v:.2f}% — High. Good fertility."

def depth_interp(v):
    v = round(float(v))
    if v < 25:   return f"{v} cm — Very shallow. Limited crop options."
    elif v < 50: return f"{v} cm — Shallow. Short-rooted crops only."
    elif v < 75: return f"{v} cm — Moderate. Suitable for most crops."
    else:         return f"{v} cm — Deep. Suitable for all crops."

TEXTURE_BANDS = [
    (1.45, "Clay"),
    (1.48, "Sandy Clay"),
    (1.49, "Clay Loam"),
    (1.54, "Loam"),
    (1.56, "Sandy Clay Loam"),
    (1.63, "Sandy Loam"),
    (1.65, "Loamy Sand"),
    (1.69, "Sand"),
]

def texture_soil_type(v):
    v = float(v)
    for upper, label in TEXTURE_BANDS:
        if v <= upper:
            return label
    return "Sand"

TEXTURE_NOTES = {
    "Clay":            "Heavy, high water retention, poor drainage.",
    "Sandy Clay":      "Retains water well, drains slowly.",
    "Clay Loam":       "Balanced retention, moderate drainage.",
    "Loam":            "Ideal for most crops.",
    "Sandy Clay Loam": "Good balance of drainage and retention.",
    "Sandy Loam":      "Good drainage, decent retention.",
    "Loamy Sand":      "Fast drainage, low retention.",
    "Sand":            "Very fast drainage, low water retention.",
}

def texture_interp(v):
    soil_type = texture_soil_type(v)
    return f"{soil_type} — {TEXTURE_NOTES.get(soil_type, '')}"

# Fixed category colors for texture soil types — dark/heavy (clay) to
# light/coarse (sand), so the map reads intuitively even without the legend.
TEXTURE_TYPE_COLORS = {
    "Clay":            "#5b3a29",
    "Sandy Clay":      "#8a5a3c",
    "Clay Loam":       "#a97450",
    "Loam":            "#6b8e23",
    "Sandy Clay Loam": "#c9a227",
    "Sandy Loam":      "#e0b96b",
    "Loamy Sand":      "#f0d18c",
    "Sand":            "#f5e6b8",
}

def ph_interp(v):
    v = float(v)
    if v < 5.5:   return f"{v:.2f} — Strongly acidic. Lime needed."
    elif v < 6.5: return f"{v:.2f} — Slightly acidic. Good for most crops."
    elif v < 7.5: return f"{v:.2f} — Neutral. Ideal."
    elif v < 8.5: return f"{v:.2f} — Slightly alkaline. Suitable for many crops."
    else:          return f"{v:.2f} — Strongly alkaline. Needs amendment."

def fertility_score(record):
    score = 0
    try:
        if float(record['SOC']) >= 0.75: score += 1
    except: pass
    try:
        if float(record['DEPTH']) >= 75: score += 1
    except: pass
    try:
        if 1.2 <= float(record['TEXTURE']) <= 1.6: score += 1
    except: pass
    try:
        if 6.0 <= float(record['PH']) <= 7.5: score += 1
    except: pass
    return score

def haversine_km(lat1, lon1, lat2, lon2):
    import math
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ── Weather (Open-Meteo, free, no API key needed) ──────────────
@st.cache_data(ttl=1800)
def fetch_weather(lat, lon):
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,relative_humidity_2m,precipitation"
            "&daily=precipitation_sum,temperature_2m_max,temperature_2m_min"
            "&timezone=auto&forecast_days=5"
        )
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def rainfall_advice(daily_precip_sum):
    total = sum(daily_precip_sum) if daily_precip_sum else 0
    if total < 5:
        return f"{total:.1f} mm expected over 5 days — Dry spell. Irrigation likely needed."
    elif total < 25:
        return f"{total:.1f} mm expected over 5 days — Moderate rainfall. Monitor soil moisture."
    else:
        return f"{total:.1f} mm expected over 5 days — Heavy rainfall expected. Watch for waterlogging."

# ── Keyword response ───────────────────────────────────────────
def keyword_response(query, record, location_name="Selected location"):
    q = query.lower()
    name = location_name

    if any(w in q for w in ["soc", "organic carbon", "carbon"]):
        return f"**SOC — {name}:** {soc_interp(record['SOC'])}"
    if "depth" in q:
        return f"**Depth — {name}:** {depth_interp(record['DEPTH'])}"
    if "texture" in q or "soil type" in q:
        return f"**Texture — {name}:** {texture_interp(record['TEXTURE'])}"
    if any(w in q for w in ["ph", "acidity", "acidic", "alkaline"]):
        return f"**pH — {name}:** {ph_interp(record['PH'])}"
    if any(w in q for w in ["summary", "all", "profile", "details", "overview"]):
        return (
            f"**Soil Summary — {name}**\n\n"
            f"- SOC: {soc_interp(record['SOC'])}\n"
            f"- Depth: {depth_interp(record['DEPTH'])}\n"
            f"- Texture: {texture_interp(record['TEXTURE'])}\n"
            f"- pH: {ph_interp(record['PH'])}"
        )
    if any(w in q for w in ["fertile", "fertility", "soil quality"]):
        score = fertility_score(record)
        rating = ["Poor", "Low", "Moderate", "Good", "Excellent"][score]
        return f"**Fertility — {name}:** {rating} ({score}/4)"
    return None

# ── Village compare helper (used by both Compare mode UI and chat) ─
def compare_villages(rec_a, rec_b):
    """Returns a DataFrame comparing two soil records. rec_a/rec_b are
    pandas Series with SOC/DEPTH/TEXTURE/PH/KGISVill_2 fields."""
    rows = [
        ("SOC (%)",     f"{float(rec_a['SOC']):.2f}",           f"{float(rec_b['SOC']):.2f}",           float(rec_a['SOC']) - float(rec_b['SOC'])),
        ("Depth (cm)",  f"{round(float(rec_a['DEPTH']))}",      f"{round(float(rec_b['DEPTH']))}",      float(rec_a['DEPTH']) - float(rec_b['DEPTH'])),
        ("Texture",     texture_soil_type(rec_a['TEXTURE']),    texture_soil_type(rec_b['TEXTURE']),    None),
        ("pH",          f"{float(rec_a['PH']):.2f}",            f"{float(rec_b['PH']):.2f}",            float(rec_a['PH']) - float(rec_b['PH'])),
    ]
    score_a, score_b = fertility_score(rec_a), fertility_score(rec_b)
    rating = ["Poor", "Low", "Moderate", "Good", "Excellent"]
    rows.append(("Fertility", f"{rating[score_a]} ({score_a}/4)", f"{rating[score_b]} ({score_b}/4)", score_a - score_b))

    df = pd.DataFrame(rows, columns=["Parameter", str(rec_a['KGISVill_2']), str(rec_b['KGISVill_2']), "Diff (A − B)"])
    return df, score_a, score_b

# ── Ranking queries: "highest SOC villages in Belagavi", "lowest pH", etc ──
PARAM_KEYWORDS = [
    # order matters: more specific phrases first
    (["organic carbon", "soc"], "SOC"),
    (["ph", "acidity", "acidic", "alkaline"], "PH"),
    (["depth"], "DEPTH"),
    (["texture"], "TEXTURE"),
    (["fertility", "fertile"], "FERTILITY"),
]

HIGH_WORDS = ["highest", "high ", "most", "top", "maximum", "max "]
LOW_WORDS  = ["lowest", "low ", "least", "minimum", "min "]

PARAM_LABELS = {
    "SOC": "SOC (%)", "PH": "pH", "DEPTH": "Depth (cm)",
    "TEXTURE": "Texture (bulk density)", "FERTILITY": "Fertility score",
}

# Physically plausible ranges — anything outside these is a sentinel/garbage
# value (e.g. -9999 for "no data") from the source shapefile/CSV, not a real
# reading. Filtered out before any average or ranking is computed.
PARAM_VALID_RANGES = {
    "SOC": (0.0, 20.0),      # % organic carbon
    "PH": (2.0, 12.0),
    "DEPTH": (0.0, 300.0),   # cm
    "TEXTURE": (1.0, 2.2),   # g/cm3 bulk density — see TEXTURE_BANDS above
}

def filter_valid_range(df, param_col):
    """Drop rows where param_col falls outside a physically plausible range.
    Guards against sentinel/placeholder values (e.g. -9999) in the source
    data that aren't NaN but aren't real readings either."""
    if param_col not in PARAM_VALID_RANGES:
        return df
    lo, hi = PARAM_VALID_RANGES[param_col]
    return df[(df[param_col] >= lo) & (df[param_col] <= hi)]

def detect_ranking_query(q):
    """Returns (param_col, order, district_filter, subdist_filter) if q is a
    ranking-style query like 'areas with high SOC' or 'lowest pH villages in
    Mysuru', else None.

    If the query doesn't name a district/sub-district explicitly, falls back
    to whatever scope is currently selected via the dropdown/map picker
    (st.session_state['current_district'] / ['current_subdist'])."""
    ql = q.lower()

    order = None
    if any(w in ql for w in HIGH_WORDS):
        order = "high"
    elif any(w in ql for w in LOW_WORDS):
        order = "low"
    if order is None:
        return None

    param_col = None
    for keywords, col in PARAM_KEYWORDS:
        if any(k in ql for k in keywords):
            param_col = col
            break
    if param_col is None:
        return None

    district_filter = None
    for d in village_df["DISTRICT"].dropna().astype(str).unique():
        if d.lower() in ql:
            district_filter = d
            break

    subdist_filter = None
    sub_scope = village_df
    if district_filter:
        sub_scope = village_df[village_df["DISTRICT"].astype(str) == district_filter]
    for s in sub_scope["SUB_DIST"].dropna().astype(str).unique():
        if s.lower() in ql:
            subdist_filter = s
            break

    # No explicit place named in the query — fall back to whatever's
    # currently selected via the dropdown/map picker.
    if district_filter is None and subdist_filter is None:
        district_filter = st.session_state.get("current_district")
        subdist_filter = st.session_state.get("current_subdist")

    return param_col, order, district_filter, subdist_filter

def rank_villages(param_col, order="high", district_filter=None, subdist_filter=None, n=10):
    df = village_df.copy()
    if district_filter:
        df = df[df["DISTRICT"].astype(str) == district_filter]
    if subdist_filter:
        df = df[df["SUB_DIST"].astype(str) == subdist_filter]

    if param_col == "FERTILITY":
        for col in ["SOC", "DEPTH", "TEXTURE", "PH"]:
            df = filter_valid_range(df.dropna(subset=[col]), col)
        if df.empty:
            return df
        df = df.copy()
        df["FERTILITY"] = df.apply(fertility_score, axis=1)
        sort_col = "FERTILITY"
    else:
        df = filter_valid_range(df[df[param_col].notna()], param_col)
        sort_col = param_col

    ascending = (order == "low")
    return df.sort_values(sort_col, ascending=ascending).head(n)

def _scope_label(district_filter, subdist_filter):
    if subdist_filter and district_filter:
        return f"{subdist_filter}, {district_filter}"
    elif district_filter:
        return district_filter
    else:
        return "Karnataka"

def format_ranking_answer(param_col, order, district_filter, subdist_filter, df_result):
    scope = _scope_label(district_filter, subdist_filter)
    direction = "Highest" if order == "high" else "Lowest"
    label = PARAM_LABELS.get(param_col, param_col)
    lines = [f"**{direction} {label} — {scope}**\n"]
    for _, row in df_result.iterrows():
        if param_col == "TEXTURE":
            val_str = texture_soil_type(row["TEXTURE"])
        elif param_col == "DEPTH":
            val_str = f"{round(float(row['DEPTH']))} cm"
        elif param_col == "FERTILITY":
            rating = ["Poor", "Low", "Moderate", "Good", "Excellent"][int(row["FERTILITY"])]
            val_str = f"{rating} ({int(row['FERTILITY'])}/4)"
        else:
            val_str = f"{float(row[param_col]):.2f}"
        lines.append(f"- {row['KGISVill_2']} ({row['SUB_DIST']}, {row['DISTRICT']}): {val_str}")
    return "\n".join(lines)

def generate_ranking_pdf(df_result, param_col, order, district_filter, subdist_filter=None):
    """PDF export of a ranking result table (top N villages by a parameter)."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm,
                            leftMargin=2*cm, rightMargin=2*cm)
    styles = getSampleStyleSheet()
    title_style   = ParagraphStyle('title',   parent=styles['Title'],   fontSize=18, textColor=colors.HexColor('#1a4d1a'), spaceAfter=6)
    heading_style = ParagraphStyle('heading', parent=styles['Heading2'], fontSize=13, textColor=colors.HexColor('#2e7d32'), spaceAfter=4)

    scope = _scope_label(district_filter, subdist_filter)
    direction = "Highest" if order == "high" else "Lowest"
    label = PARAM_LABELS.get(param_col, param_col)

    story = [
        Paragraph("Karnataka Soil Report — Ranking", title_style),
        Paragraph(f"{direction} {label} — {scope}", heading_style),
        Spacer(1, 0.3*cm),
    ]

    rows = [["Village", "Sub-district", "District", label]]
    for _, row in df_result.iterrows():
        if param_col == "TEXTURE":
            val_str = texture_soil_type(row["TEXTURE"])
        elif param_col == "DEPTH":
            val_str = f"{round(float(row['DEPTH']))}"
        elif param_col == "FERTILITY":
            val_str = ["Poor", "Low", "Moderate", "Good", "Excellent"][int(row["FERTILITY"])]
        else:
            val_str = f"{float(row[param_col]):.2f}"
        rows.append([row["KGISVill_2"], row["SUB_DIST"], row["DISTRICT"], val_str])

    table = Table(rows, colWidths=[4.5*cm, 4*cm, 4*cm, 4*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,0), colors.HexColor('#2e7d32')),
        ('TEXTCOLOR',  (0,0),(-1,0), colors.white),
        ('FONTNAME',   (0,0),(-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0),(-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1),(-1,-1), [colors.white, colors.HexColor('#f5f7f0')]),
        ('GRID',       (0,0),(-1,-1), 0.5, colors.HexColor('#c8e6c9')),
        ('PADDING',    (0,0),(-1,-1), 6),
        ('TEXTCOLOR',  (0,1),(-1,-1), colors.HexColor('#1a3a1a')),
    ]))
    story.append(table)
    doc.build(story)
    buffer.seek(0)
    return buffer

def ranking_result_to_csv_bytes(df_result, param_col):
    cols = ["KGISVill_2", "SUB_DIST", "DISTRICT"]
    if param_col == "FERTILITY":
        cols.append("FERTILITY")
    else:
        cols.append(param_col)
    return df_result[cols].to_csv(index=False).encode("utf-8")

# ── Nearest-value queries: "SOC nearest to 6.2", "pH close to 7 in Mysuru" ──
def detect_nearest_query(q):
    """Returns (param_col, target_value, district_filter) for queries like
    'SOC nearest to 6.2', 'villages with pH close to 7 in Mysuru',
    'depth around 50', else None."""
    ql = q.lower()

    trigger_words = ["nearest to", "closest to", "close to", "near ", "around ", "approx"]
    if not any(w in ql for w in trigger_words):
        return None

    param_col = None
    for keywords, col in PARAM_KEYWORDS:
        if col == "FERTILITY":
            continue  # fertility is a 0-4 score, not a value to target
        if any(k in ql for k in keywords):
            param_col = col
            break
    if param_col is None:
        return None

    import re
    match = re.search(r'(\d+\.?\d*)', ql)
    if not match:
        return None
    target_value = float(match.group(1))

    district_filter = None
    for d in village_df["DISTRICT"].dropna().astype(str).unique():
        if d.lower() in ql:
            district_filter = d
            break

    return param_col, target_value, district_filter

def rank_villages_nearest(param_col, target_value, district_filter=None, n=10):
    df = village_df.copy()
    if district_filter:
        df = df[df["DISTRICT"].astype(str) == district_filter]

    df = filter_valid_range(df[df[param_col].notna()], param_col)
    if df.empty:
        return df

    df = df.copy()
    df["_diff"] = (df[param_col].astype(float) - target_value).abs()
    return df.sort_values("_diff").head(n)

def format_nearest_answer(param_col, target_value, district_filter, df_result):
    scope = district_filter if district_filter else "Karnataka"
    label = PARAM_LABELS.get(param_col, param_col)
    lines = [f"**{label} closest to {target_value:g} — {scope}**\n"]
    for _, row in df_result.iterrows():
        if param_col == "TEXTURE":
            val_str = texture_soil_type(row["TEXTURE"])
        elif param_col == "DEPTH":
            val_str = f"{round(float(row['DEPTH']))} cm"
        else:
            val_str = f"{float(row[param_col]):.2f}"
        lines.append(f"- {row['KGISVill_2']} ({row['SUB_DIST']}, {row['DISTRICT']}): {val_str} (Δ {row['_diff']:.2f})")
    return "\n".join(lines)

def nearest_result_to_csv_bytes(df_result, param_col):
    cols = ["KGISVill_2", "SUB_DIST", "DISTRICT", param_col, "_diff"]
    return df_result[cols].rename(columns={"_diff": "Diff_from_target"}).to_csv(index=False).encode("utf-8")

def generate_nearest_pdf(df_result, param_col, target_value, district_filter):
    """PDF export of a nearest-value result table (N villages closest to a target)."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm,
                            leftMargin=2*cm, rightMargin=2*cm)
    styles = getSampleStyleSheet()
    title_style   = ParagraphStyle('title',   parent=styles['Title'],   fontSize=18, textColor=colors.HexColor('#1a4d1a'), spaceAfter=6)
    heading_style = ParagraphStyle('heading', parent=styles['Heading2'], fontSize=13, textColor=colors.HexColor('#2e7d32'), spaceAfter=4)

    scope = district_filter if district_filter else "Karnataka"
    label = PARAM_LABELS.get(param_col, param_col)

    story = [
        Paragraph("Karnataka Soil Report — Nearest Value", title_style),
        Paragraph(f"{label} closest to {target_value:g} — {scope}", heading_style),
        Spacer(1, 0.3*cm),
    ]

    rows = [["Village", "Sub-district", "District", label, "Δ from target"]]
    for _, row in df_result.iterrows():
        if param_col == "TEXTURE":
            val_str = texture_soil_type(row["TEXTURE"])
        elif param_col == "DEPTH":
            val_str = f"{round(float(row['DEPTH']))}"
        else:
            val_str = f"{float(row[param_col]):.2f}"
        rows.append([row["KGISVill_2"], row["SUB_DIST"], row["DISTRICT"], val_str, f"{row['_diff']:.2f}"])

    table = Table(rows, colWidths=[4*cm, 3.5*cm, 3.5*cm, 3*cm, 3*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,0), colors.HexColor('#2e7d32')),
        ('TEXTCOLOR',  (0,0),(-1,0), colors.white),
        ('FONTNAME',   (0,0),(-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0),(-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1),(-1,-1), [colors.white, colors.HexColor('#f5f7f0')]),
        ('GRID',       (0,0),(-1,-1), 0.5, colors.HexColor('#c8e6c9')),
        ('PADDING',    (0,0),(-1,-1), 6),
        ('TEXTCOLOR',  (0,1),(-1,-1), colors.HexColor('#1a3a1a')),
    ]))
    story.append(table)
    doc.build(story)
    buffer.seek(0)
    return buffer

# ── Range queries: "pH between 6 and 7", "SOC 0.5 to 0.75 in Mysuru" ──────
def detect_range_query(q):
    """Returns (param_col, lo, hi, district_filter, subdist_filter) for queries
    like 'pH between 6 and 7', 'SOC from 0.5 to 0.75 in Mysuru',
    'depth 40-60 in Belagavi', else None.

    Falls back to the currently selected dropdown/map scope when no place is
    named, same as detect_ranking_query / detect_nearest_query."""
    ql = q.lower()

    if not any(w in ql for w in ["between", " to ", "-", " and "]):
        return None

    param_col = None
    for keywords, col in PARAM_KEYWORDS:
        if col == "FERTILITY":
            continue  # range doesn't make sense for a 0-4 score
        if any(k in ql for k in keywords):
            param_col = col
            break
    if param_col is None:
        return None

    import re
    # matches "between 6 and 7", "6 to 7", "6-7", "6 and 7"
    match = re.search(
        r'(\d+\.?\d*)\s*(?:-|to|and)\s*(\d+\.?\d*)', ql
    )
    if not match:
        return None

    v1, v2 = float(match.group(1)), float(match.group(2))
    lo, hi = min(v1, v2), max(v1, v2)
    if lo == hi:
        return None  # not actually a range, let detect_nearest_query handle it

    district_filter = None
    for d in village_df["DISTRICT"].dropna().astype(str).unique():
        if d.lower() in ql:
            district_filter = d
            break

    subdist_filter = None
    sub_scope = village_df
    if district_filter:
        sub_scope = village_df[village_df["DISTRICT"].astype(str) == district_filter]
    for s in sub_scope["SUB_DIST"].dropna().astype(str).unique():
        if s.lower() in ql:
            subdist_filter = s
            break

    if district_filter is None and subdist_filter is None:
        district_filter = st.session_state.get("current_district")
        subdist_filter = st.session_state.get("current_subdist")

    return param_col, lo, hi, district_filter, subdist_filter

def rank_villages_range(param_col, lo, hi, district_filter=None, subdist_filter=None, n=50):
    df = village_df.copy()
    if district_filter:
        df = df[df["DISTRICT"].astype(str) == district_filter]
    if subdist_filter:
        df = df[df["SUB_DIST"].astype(str) == subdist_filter]

    df = filter_valid_range(df[df[param_col].notna()], param_col)
    if df.empty:
        return df

    mask = (df[param_col].astype(float) >= lo) & (df[param_col].astype(float) <= hi)
    return df[mask].sort_values(param_col).head(n)

def format_range_answer(param_col, lo, hi, district_filter, subdist_filter, df_result):
    scope = _scope_label(district_filter, subdist_filter)
    label = PARAM_LABELS.get(param_col, param_col)
    lines = [f"**{label} between {lo:g} and {hi:g} — {scope}** ({len(df_result)} matches)\n"]
    for _, row in df_result.head(20).iterrows():
        if param_col == "TEXTURE":
            val_str = texture_soil_type(row["TEXTURE"])
        elif param_col == "DEPTH":
            val_str = f"{round(float(row['DEPTH']))} cm"
        else:
            val_str = f"{float(row[param_col]):.2f}"
        lines.append(f"- {row['KGISVill_2']} ({row['SUB_DIST']}, {row['DISTRICT']}): {val_str}")
    if len(df_result) > 20:
        lines.append(f"\n_...and {len(df_result) - 20} more. Download CSV/PDF for the full list._")
    return "\n".join(lines)

def range_result_to_csv_bytes(df_result, param_col):
    cols = ["KGISVill_2", "SUB_DIST", "DISTRICT", param_col]
    return df_result[cols].to_csv(index=False).encode("utf-8")

def generate_range_pdf(df_result, param_col, lo, hi, district_filter, subdist_filter=None):
    """PDF export of a range-query result table (villages within [lo, hi])."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm,
                            leftMargin=2*cm, rightMargin=2*cm)
    styles = getSampleStyleSheet()
    title_style   = ParagraphStyle('title',   parent=styles['Title'],   fontSize=18, textColor=colors.HexColor('#1a4d1a'), spaceAfter=6)
    heading_style = ParagraphStyle('heading', parent=styles['Heading2'], fontSize=13, textColor=colors.HexColor('#2e7d32'), spaceAfter=4)

    scope = _scope_label(district_filter, subdist_filter)
    label = PARAM_LABELS.get(param_col, param_col)

    story = [
        Paragraph("Karnataka Soil Report — Range Query", title_style),
        Paragraph(f"{label} between {lo:g} and {hi:g} — {scope}", heading_style),
        Spacer(1, 0.3*cm),
    ]

    rows = [["Village", "Sub-district", "District", label]]
    for _, row in df_result.iterrows():
        if param_col == "TEXTURE":
            val_str = texture_soil_type(row["TEXTURE"])
        elif param_col == "DEPTH":
            val_str = f"{round(float(row['DEPTH']))}"
        else:
            val_str = f"{float(row[param_col]):.2f}"
        rows.append([row["KGISVill_2"], row["SUB_DIST"], row["DISTRICT"], val_str])

    table = Table(rows, colWidths=[4.5*cm, 4*cm, 4*cm, 4*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,0), colors.HexColor('#2e7d32')),
        ('TEXTCOLOR',  (0,0),(-1,0), colors.white),
        ('FONTNAME',   (0,0),(-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0),(-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1),(-1,-1), [colors.white, colors.HexColor('#f5f7f0')]),
        ('GRID',       (0,0),(-1,-1), 0.5, colors.HexColor('#c8e6c9')),
        ('PADDING',    (0,0),(-1,-1), 6),
        ('TEXTCOLOR',  (0,1),(-1,-1), colors.HexColor('#1a3a1a')),
    ]))
    story.append(table)
    doc.build(story)
    buffer.seek(0)
    return buffer

# ── Compare-intent parsing for chat ("compare X and Y", "X vs Y") ──
def detect_compare_query(q):
    """Returns (village_name_a, village_name_b) if q looks like a compare
    request and both names can be matched against known villages, else None."""
    ql = q.lower()
    if not any(w in ql for w in ["compare", " vs ", " vs.", "versus"]):
        return None

    import re
    q_clean = re.sub(r'^.*?compare\s+', '', ql) if "compare" in ql else ql
    q_clean = re.sub(r'\bsoil\b|\bfertility\b|\bdata\b', '', q_clean)
    parts = re.split(r'\s+(?:vs\.?|versus|and)\s+|,', q_clean)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) < 2:
        return None

    all_villages = village_df["KGISVill_2"].dropna().astype(str).unique()
    matched = []
    for part in parts[:2]:
        hit = next((v for v in all_villages if part == v.lower()), None)
        if hit is None:
            hit = next((v for v in all_villages if part in v.lower() or v.lower() in part), None)
        if hit:
            matched.append(hit)

    if len(matched) < 2 or matched[0] == matched[1]:
        return None
    return matched[0], matched[1]

# ── PDF report ─────────────────────────────────────────────────
def generate_pdf_report(record, village_name=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm,
                            leftMargin=2*cm, rightMargin=2*cm)
    styles = getSampleStyleSheet()
    title_style   = ParagraphStyle('title',   parent=styles['Title'],   fontSize=18, textColor=colors.HexColor('#1a4d1a'), spaceAfter=6)
    heading_style = ParagraphStyle('heading', parent=styles['Heading2'], fontSize=13, textColor=colors.HexColor('#2e7d32'), spaceAfter=4)
    normal_style  = ParagraphStyle('normal',  parent=styles['Normal'],  fontSize=11, textColor=colors.HexColor('#1a3a1a'), spaceAfter=4)

    story = []
    story.append(Paragraph("Karnataka Soil Report", title_style))
    label = village_name or "Custom location"
    story.append(Paragraph(f"Location: {label}", heading_style))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Soil Parameters", heading_style))
    soil_data = [
        ["Parameter", "Value", "Interpretation"],
        ["SOC (%)", f"{float(record['SOC']):.2f}", soc_interp(record['SOC'])],
        ["Depth (cm)", str(round(float(record['DEPTH']))), depth_interp(record['DEPTH'])],
        ["Texture",    texture_soil_type(record['TEXTURE']), texture_interp(record['TEXTURE'])],
        ["pH",         f"{float(record['PH']):.2f}", ph_interp(record['PH'])],
    ]
    soil_table = Table(soil_data, colWidths=[4*cm, 3*cm, 10*cm])
    soil_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,0), colors.HexColor('#2e7d32')),
        ('TEXTCOLOR',  (0,0),(-1,0), colors.white),
        ('FONTNAME',   (0,0),(-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0),(-1,-1), 10),
        ('ROWBACKGROUNDS', (0,1),(-1,-1), [colors.white, colors.HexColor('#f5f7f0')]),
        ('GRID',       (0,0),(-1,-1), 0.5, colors.HexColor('#c8e6c9')),
        ('PADDING',    (0,0),(-1,-1), 8),
        ('TEXTCOLOR',  (0,1),(-1,-1), colors.HexColor('#1a3a1a')),
    ]))
    story.append(soil_table)
    story.append(Spacer(1, 0.5*cm))

    score  = fertility_score(record)
    rating = ["Poor", "Low", "Moderate", "Good", "Excellent"][score]
    story.append(Paragraph("Overall Fertility", heading_style))
    story.append(Paragraph(f"Rating: <b>{rating}</b> ({score}/4 parameters optimal)", normal_style))
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("Generated by Karnataka Soil Chatbot",
                            ParagraphStyle('footer', parent=styles['Normal'], fontSize=9, textColor=colors.grey)))
    doc.build(story)
    buffer.seek(0)
    return buffer

async def _speak_text_async(text, voice="en-IN-NeerjaNeural"):
    communicate = edge_tts.Communicate(str(text), voice=voice)
    path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    await communicate.save(path)
    return path

def speak_text(text):
    try:
        return asyncio.run(_speak_text_async(text))
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════
# Map drill-down helpers (boundary-based)
# ══════════════════════════════════════════════════════════════
def gdf_to_geojson(gdf):
    return json.loads(gdf.to_json())

def feature_bounds_from_geom(geom):
    minx, miny, maxx, maxy = geom.bounds
    return [[miny, minx], [maxy, maxx]]

# ── Per-feature color palette (used for district-level coloring) ──
DISTRICT_PALETTE = [
    "#e57373", "#64b5f6", "#81c784", "#ffb74d", "#ba68c8",
    "#4db6ac", "#f06292", "#a1887f", "#90a4ae", "#dce775",
    "#4fc3f7", "#ff8a65", "#9575cd", "#aed581", "#7986cb",
    "#fff176", "#4dd0e1", "#f8bbd0", "#c5e1a5", "#ffd54f",
    "#ef9a9a", "#80cbc4", "#ce93d8", "#fff59d", "#b0bec5",
    "#c5cae9", "#ffcc80", "#a5d6a7", "#f48fb1", "#bcaaa4",
]

def color_for_name(name):
    """Stable hash → palette index. Uses md5 instead of built-in hash()
    because str hash() is randomized per-process (PYTHONHASHSEED),
    which would reshuffle district colors on every restart."""
    h = int(hashlib.md5(str(name).encode()).hexdigest(), 16)
    return DISTRICT_PALETTE[h % len(DISTRICT_PALETTE)]

def add_boundary_layer(m, geojson_data, name_key, fill_color="#4caf50",
                        border_color="#1a4d1a", color_by_feature=False,
                        value_color_map=None, value_labels=None, default_color="#cccccc"):
    """value_color_map: optional {name: hex_color} to color features by a
    continuous value (choropleth) instead of a categorical hash.
    value_labels: optional {name: "display value string"} shown in the tooltip."""
    # Fill in missing/blank names so the tooltip never renders as an empty pill
    for feature in geojson_data.get("features", []):
        props = feature.get("properties", {})
        val = props.get(name_key)
        if val is None or str(val).strip() == "":
            props[name_key] = "Unknown"
        if value_labels is not None:
            props["_value_label"] = value_labels.get(props[name_key], "No data")

    def style_fn(f):
        name = f["properties"].get(name_key, "")
        if value_color_map is not None:
            c = value_color_map.get(name, default_color)
            return {"fillColor": c, "color": "#333333", "weight": 1.2, "fillOpacity": 0.65}
        if color_by_feature:
            c = color_for_name(name)
            return {"fillColor": c, "color": "#333333", "weight": 1.2, "fillOpacity": 0.45}
        return {"fillColor": fill_color, "color": border_color, "weight": 1.5, "fillOpacity": 0.25}

    tooltip_fields = [name_key] + (["_value_label"] if value_labels is not None else [])
    tooltip_labels = value_labels is not None  # show field name only when there are 2 fields

    folium.GeoJson(
        geojson_data,
        style_function=style_fn,
        highlight_function=lambda f: {"fillColor": "#ffb300", "color": "#e65100", "weight": 2.5, "fillOpacity": 0.55},
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            labels=tooltip_labels,
            sticky=True,       # tooltip follows the cursor and appears reliably on hover
            style="""
                background-color: #ffffff !important;
                color: #1a3a1a !important;
                font-weight: 600 !important;
                font-size: 13px !important;
                padding: 4px 8px !important;
                border: 1px solid #a5d6a7 !important;
                border-radius: 4px !important;
                white-space: nowrap !important;
            """,
        ),
    ).add_to(m)

def compute_texture_choropleth(df_source, group_col):
    """Dominant soil-type category per group_col value (by count, not average
    bulk density — texture is categorical, not a continuous high/low scale).
    Returns {name: hex_color}, {name: soil_type_label}, (None, None)."""
    df = filter_valid_range(df_source.dropna(subset=["TEXTURE"]), "TEXTURE")
    if df.empty:
        return {}, {}, (None, None)

    df = df.copy()
    df["_soil_type"] = df["TEXTURE"].apply(texture_soil_type)
    dominant = df.groupby(group_col)["_soil_type"].agg(lambda s: s.value_counts().idxmax())

    color_map = {name: TEXTURE_TYPE_COLORS.get(t, "#cccccc") for name, t in dominant.items()}
    label_map = {name: t for name, t in dominant.items()}
    return color_map, label_map, (None, None)

def compute_choropleth(df_source, group_col, param_col, cmap_name="RdYlGn_r"):
    """Average param_col per group_col value → {name: hex_color}, {name: label}, (vmin, vmax).
    TEXTURE is categorical, so it's routed to compute_texture_choropleth instead."""
    if param_col == "TEXTURE":
        return compute_texture_choropleth(df_source, group_col)

    df = filter_valid_range(df_source.dropna(subset=[param_col]), param_col)
    avg = df.groupby(group_col)[param_col].mean()
    if avg.empty:
        return {}, {}, (None, None)

    vmin, vmax = float(avg.min()), float(avg.max())
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax) if vmax > vmin else mcolors.Normalize(vmin=vmin - 1, vmax=vmax + 1)
    cmap = mpl.colormaps[cmap_name]

    color_map, label_map = {}, {}
    for name, value in avg.items():
        color_map[name] = mcolors.rgb2hex(cmap(norm(value)))
        label_map[name] = f"{value:.2f}"
    return color_map, label_map, (vmin, vmax)

def render_choropleth_legend(param_col, metric_display_name, label_map, vmin, vmax, group_word):
    """Shows a numeric gradient caption for continuous params, or a colored
    swatch legend of soil-type categories present when param_col is TEXTURE."""
    if param_col == "TEXTURE":
        if not label_map:
            return
        types_present = sorted(set(label_map.values()), key=lambda t: [b[1] for b in TEXTURE_BANDS].index(t) if t in [b[1] for b in TEXTURE_BANDS] else 99)
        swatches = " &nbsp; ".join(
            f'<span style="display:inline-block;width:12px;height:12px;background:{TEXTURE_TYPE_COLORS.get(t, "#ccc")};'
            f'border-radius:2px;margin-right:4px;vertical-align:middle;"></span>{t}'
            for t in types_present
        )
        st.markdown(f"🎨 Dominant soil type per {group_word}: {swatches}", unsafe_allow_html=True)
    elif vmin is not None:
        st.caption(f"🎨 Colour scale (green → red): {vmin:.2f} → {vmax:.2f} average {metric_display_name} per {group_word}")

CHOROPLETH_METRIC_OPTIONS = ["Default (by name)", "SOC (avg)", "pH (avg)", "Depth (avg)", "Texture (soil type)"]
CHOROPLETH_METRIC_COL_MAP = {"SOC (avg)": "SOC", "pH (avg)": "PH", "Depth (avg)": "DEPTH", "Texture (soil type)": "TEXTURE"}

def init_map_state():
    defaults = {
        "map_level": "district",
        "sel_district": None,
        "sel_subdist": None,
        "sel_village": None,
        "zoom_bounds": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

def set_current_scope(district=None, subdist=None):
    """Tracks the district/sub-district currently selected via the
    dropdown or map picker, so chat ranking queries that don't name a
    place explicitly can fall back to this scope."""
    st.session_state["current_district"] = district
    st.session_state["current_subdist"] = subdist

def reset_map_selection():
    st.session_state.map_level = "district"
    st.session_state.sel_district = None
    st.session_state.sel_subdist = None
    st.session_state.sel_village = None
    st.session_state.zoom_bounds = None
    set_current_scope(None, None)

# ══════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════
st.set_page_config(page_title="Karnataka Soil Chatbot", page_icon="🌱", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; color: #1a3a1a !important; }
.stApp { background-color: #f5f7f0 !important; }
p, div, span, label, li, td, th, a { color: #1a3a1a !important; }
h1 { color: #1a4d1a !important; font-weight: 700 !important; font-size: 1.2rem !important; border-bottom: 3px solid #4caf50; padding-bottom: 10px; margin-bottom: 20px !important; }
h2, h3 { color: #1a4d1a !important; font-weight: 700 !important; }
section[data-testid="stSidebar"] { background-color: #2d5a2d !important; }
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label { color: #d4edda !important; }
.stSelectbox > div > div { border: 1.5px solid #a5d6a7 !important; border-radius: 8px !important; background-color: #ffffff !important; }
.stTextInput > div > div > input { border: 1.5px solid #a5d6a7 !important; border-radius: 8px !important; background-color: #ffffff !important; color: #1a3a1a !important; }
.stButton > button { background-color: #2e7d32 !important; color: white !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; }
.stButton > button:hover { background-color: #1b5e20 !important; }
.stButton > button p { color: white !important; }
[data-testid="stChatMessage"] { border-radius: 12px !important; padding: 12px 16px !important; margin-bottom: 8px !important; border: 1px solid #c8e6c9 !important; background-color: #ffffff !important; }
.stSuccess { background-color: #e8f5e9 !important; border-left: 4px solid #4caf50 !important; border-radius: 6px !important; }
[data-testid="stMetricLabel"] { font-size: 11px !important; color: #2e6b2e !important; }
[data-testid="stMetricValue"] { font-size: 18px !important; color: #1a3a1a !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🌱 Karnataka Soil Chatbot</h1>", unsafe_allow_html=True)

search_mode = st.radio(
    "Search by",
    ["District / Sub-district / Village (dropdown)", "Map (click to select)",
     "Latitude & Longitude", "Compare Two Villages"]
)

record           = None
nearest_village  = None
input_lat = input_lon = None
location_label   = "Selected location"

# ── Mode 1: Dropdown picker ──────────────────────────────────
if search_mode == "District / Sub-district / Village (dropdown)":
    selected_district = st.selectbox("District", sorted(village_df["DISTRICT"].dropna().astype(str).unique()))
    sub_df = village_df[village_df["DISTRICT"].astype(str) == selected_district]
    selected_subdist = st.selectbox("Sub-district", sorted(sub_df["SUB_DIST"].dropna().astype(str).unique()))
    vill_df = sub_df[sub_df["SUB_DIST"].astype(str) == selected_subdist]
    selected_village = st.selectbox("Village", sorted(vill_df["KGISVill_2"].dropna().astype(str).unique()))
    record = vill_df[vill_df["KGISVill_2"].astype(str) == selected_village].iloc[0]
    location_label = str(record["KGISVill_2"])
    set_current_scope(selected_district, selected_subdist)

# ── Mode 2: Map-based drill-down picker (boundary polygons) ─────
elif search_mode == "Map (click to select)":
    init_map_state()

    st.markdown("### 🗺️ Click the map: District → Sub-district → Village")

    b1, b2, b3, _ = st.columns([1, 1, 1, 3])
    if b1.button("⟲ Reset", key="btn_reset_map"):
        reset_map_selection()
        st.rerun()
    if st.session_state.sel_district and b2.button(f"◀ {st.session_state.sel_district}", key="btn_back_district"):
        st.session_state.map_level = "district"
        st.session_state.sel_district = None
        st.session_state.sel_subdist = None
        st.session_state.sel_village = None
        st.session_state.zoom_bounds = None
        set_current_scope(None, None)
        st.rerun()
    if st.session_state.sel_subdist and b3.button(f"◀ {st.session_state.sel_subdist}", key="btn_back_subdist"):
        st.session_state.map_level = "subdistrict"
        st.session_state.sel_subdist = None
        st.session_state.sel_village = None
        set_current_scope(st.session_state.sel_district, None)
        st.rerun()

    # ---- LEVEL 1: districts ----
    if st.session_state.map_level == "district":
        district_gdf = get_district_boundaries()

        color_metric = st.selectbox(
            "Color districts by", CHOROPLETH_METRIC_OPTIONS, key="district_color_metric",
        )

        m = folium.Map(location=[15.3, 75.7], zoom_start=7, tiles="CartoDB positron")

        if color_metric in CHOROPLETH_METRIC_COL_MAP:
            param_col = CHOROPLETH_METRIC_COL_MAP[color_metric]
            color_map, label_map, (vmin, vmax) = compute_choropleth(village_df, "DISTRICT", param_col)
            add_boundary_layer(m, gdf_to_geojson(district_gdf), "DISTRICT",
                                value_color_map=color_map, value_labels=label_map)
            render_choropleth_legend(param_col, color_metric.split(' ')[0], label_map, vmin, vmax, "district")
        else:
            add_boundary_layer(m, gdf_to_geojson(district_gdf), "DISTRICT", color_by_feature=True)

        map_data = st_folium(m, height=520, width=None, key="district_map")
        clicked = map_data.get("last_active_drawing")
        if clicked:
            name = clicked["properties"]["DISTRICT"].strip()
            if name != st.session_state.sel_district:
                row = district_gdf[district_gdf["DISTRICT"] == name].iloc[0]
                st.session_state.sel_district = name
                st.session_state.sel_subdist = None
                st.session_state.sel_village = None
                st.session_state.map_level = "subdistrict"
                st.session_state.zoom_bounds = feature_bounds_from_geom(row.geometry)
                set_current_scope(name, None)
                st.rerun()

    # ---- LEVEL 2: sub-districts ----
    elif st.session_state.map_level == "subdistrict":
        taluk_gdf = get_taluk_boundaries(st.session_state.sel_district)
        if taluk_gdf.empty:
            st.error(f"No boundary data found for district '{st.session_state.sel_district}'. Click Reset.")
        else:
            color_metric = st.selectbox(
                "Color sub-districts by", CHOROPLETH_METRIC_OPTIONS, key="subdist_color_metric",
            )

            m = folium.Map(tiles="CartoDB positron")
            m.fit_bounds(st.session_state.zoom_bounds)

            district_villages = village_df[village_df["DISTRICT"].astype(str) == st.session_state.sel_district]

            if color_metric in CHOROPLETH_METRIC_COL_MAP:
                param_col = CHOROPLETH_METRIC_COL_MAP[color_metric]
                color_map, label_map, (vmin, vmax) = compute_choropleth(district_villages, "SUB_DIST", param_col)
                add_boundary_layer(m, gdf_to_geojson(taluk_gdf), "SUB_DIST",
                                    value_color_map=color_map, value_labels=label_map)
                render_choropleth_legend(param_col, color_metric.split(' ')[0], label_map, vmin, vmax, "sub-district")
            else:
                add_boundary_layer(m, gdf_to_geojson(taluk_gdf), "SUB_DIST", fill_color="#4caf50", border_color="#1a4d1a")

            map_data = st_folium(m, height=520, width=None, key="subdist_map")
            clicked = map_data.get("last_active_drawing")
            if clicked:
                name = clicked["properties"]["SUB_DIST"].strip()
                if name != st.session_state.sel_subdist:
                    row = taluk_gdf[taluk_gdf["SUB_DIST"] == name].iloc[0]
                    st.session_state.sel_subdist = name
                    st.session_state.sel_village = None
                    st.session_state.map_level = "village"
                    st.session_state.zoom_bounds = feature_bounds_from_geom(row.geometry)
                    set_current_scope(st.session_state.sel_district, name)
                    st.rerun()

    # ---- LEVEL 3: villages ----
    elif st.session_state.map_level == "village":
        village_boundaries_gdf = load_village_boundaries()  # lazy — cached after first call
        vill_gdf = village_boundaries_gdf[
            (village_boundaries_gdf["DISTRICT"] == st.session_state.sel_district) &
            (village_boundaries_gdf["SUB_DIST"] == st.session_state.sel_subdist)
        ]
        if vill_gdf.empty:
            st.error(
                f"No village boundaries found for sub-district '{st.session_state.sel_subdist}' in "
                f"'{st.session_state.sel_district}'. Click Reset and try again."
            )
        else:
            color_metric = st.selectbox(
                "Color villages by", CHOROPLETH_METRIC_OPTIONS, key="village_color_metric",
            )

            m = folium.Map(tiles="CartoDB positron")
            m.fit_bounds(st.session_state.zoom_bounds)

            subdist_villages = village_df[
                (village_df["DISTRICT"].astype(str) == st.session_state.sel_district) &
                (village_df["SUB_DIST"].astype(str) == st.session_state.sel_subdist)
            ]

            if color_metric in CHOROPLETH_METRIC_COL_MAP:
                param_col = CHOROPLETH_METRIC_COL_MAP[color_metric]
                color_map, label_map, (vmin, vmax) = compute_choropleth(subdist_villages, "KGISVill_2", param_col)
                add_boundary_layer(m, gdf_to_geojson(vill_gdf), "KGISVill_2",
                                    value_color_map=color_map, value_labels=label_map)
                render_choropleth_legend(param_col, color_metric.split(' ')[0], label_map, vmin, vmax, "village")
            else:
                add_boundary_layer(m, gdf_to_geojson(vill_gdf), "KGISVill_2", fill_color="#ff9800", border_color="#e65100")

            map_data = st_folium(m, height=520, width=None, key="village_map")
            clicked = map_data.get("last_active_drawing")
            if clicked:
                name = clicked["properties"]["KGISVill_2"].strip()
                if name != st.session_state.sel_village:
                    st.session_state.sel_village = name
                    st.rerun()

    if st.session_state.sel_village:
        target_village = st.session_state.sel_village.strip()
        matches = village_df[village_df["KGISVill_2"].astype(str).str.strip() == target_village]
        if st.session_state.sel_subdist:
            matches = matches[matches["SUB_DIST"].astype(str).str.strip() == st.session_state.sel_subdist.strip()]
        if st.session_state.sel_district:
            matches = matches[matches["DISTRICT"].astype(str).str.strip() == st.session_state.sel_district.strip()]

        if matches.empty:
            record = None
            st.warning("Could not match the selected village to soil data — please click Reset and try again.")
        else:
            record = matches.iloc[0]
            location_label = str(record["KGISVill_2"])
            st.success(f"📍 Selected: **{location_label}** ({st.session_state.sel_subdist}, {st.session_state.sel_district})")
    else:
        record = None
        st.info("Click a district boundary, then a sub-district boundary, then a village boundary.")

# ── Mode 3: Lat/Lon — IDW estimate from CSV + show nearest village ────
elif search_mode == "Latitude & Longitude":
    col_a, col_b = st.columns(2)
    input_lat = col_a.number_input("Latitude",  format="%.6f", value=15.0)
    input_lon = col_b.number_input("Longitude", format="%.6f", value=75.0)

    csv_record = idw_estimate(
        csv_tree, csv_df, input_lat, input_lon,
        columns=["SOC", "DEPTH", "TEXTURE", "PH"],
        k=4
    )

    if csv_record is None:
        st.warning("No soil data available near this location (outside coverage area).")
    else:
        _, vill_idx = village_tree.query([[input_lat, input_lon]], k=1)
        nearest_village = village_df.iloc[vill_idx[0]]
        dist_km = haversine_km(input_lat, input_lon,
                               float(nearest_village["latitude"]),
                               float(nearest_village["longitude"]))

        st.markdown("---")
        st.success(
            f"📍 Nearest village: **{nearest_village['KGISVill_2']}** "
            f"({nearest_village['SUB_DIST']}, {nearest_village['DISTRICT']}) "
            f"— {dist_km:.1f} km away"
        )

        record = csv_record
        location_label = f"{input_lat:.4f}, {input_lon:.4f} (near {nearest_village['KGISVill_2']})"

# ── Mode 4: Compare Two Villages — side by side + diff table ───
else:
    st.markdown("### ⚖️ Compare Two Villages")

    def village_picker(label_prefix, key_prefix):
        d = st.selectbox(f"{label_prefix} District", sorted(village_df["DISTRICT"].dropna().astype(str).unique()), key=f"{key_prefix}_d")
        sub = village_df[village_df["DISTRICT"].astype(str) == d]
        s = st.selectbox(f"{label_prefix} Sub-district", sorted(sub["SUB_DIST"].dropna().astype(str).unique()), key=f"{key_prefix}_s")
        vill = sub[sub["SUB_DIST"].astype(str) == s]
        v = st.selectbox(f"{label_prefix} Village", sorted(vill["KGISVill_2"].dropna().astype(str).unique()), key=f"{key_prefix}_v")
        return vill[vill["KGISVill_2"].astype(str) == v].iloc[0]

    col_x, col_y = st.columns(2)
    with col_x:
        rec_a = village_picker("Village A —", "cmp_a")
    with col_y:
        rec_b = village_picker("Village B —", "cmp_b")

    compare_df, score_a, score_b = compare_villages(rec_a, rec_b)
    st.dataframe(compare_df, width='stretch', hide_index=True)

    st.markdown("#### 🗺️ Locations")
    st.map(pd.DataFrame({
        "lat": [rec_a["latitude"], rec_b["latitude"]],
        "lon": [rec_a["longitude"], rec_b["longitude"]],
    }))

    if score_a != score_b:
        winner = rec_a['KGISVill_2'] if score_a > score_b else rec_b['KGISVill_2']
        st.success(f"🏆 **{winner}** has better overall soil fertility.")
    else:
        st.info("Both villages have equal fertility scores.")

    st.stop()

# ══════════════════════════════════════════════════════════════
# Metrics display
# ══════════════════════════════════════════════════════════════
st.markdown("---")

if record is None:
    st.info("Select a valid location to view soil data — or use the chat box below to ask a state/district-wide question like 'highest SOC villages in Belagavi', 'pH between 6 and 7', or 'SOC nearest to 6.2'.")

if search_mode == "Latitude & Longitude" and record is not None:
    st.markdown("#### 📊 IDW-estimated soil data at entered coordinates")
    st.caption("Estimated from the 4 nearest known sample points, weighted by distance.")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Lat / Lon", f"{input_lat:.4f}, {input_lon:.4f}")
    c2.metric("SOC (%)", f"{float(record['SOC']):.2f}")
    c3.metric("Depth (cm)", round(float(record['DEPTH'])))
    c4.metric("Texture",    texture_soil_type(record['TEXTURE']))
    c5.metric("pH",         f"{float(record['PH']):.2f}")

    st.markdown("#### 🏘️ Nearest village soil data (for reference)")
    v1, v2, v3, v4, v5, v6 = st.columns(6)
    v1.metric("Village",    nearest_village["KGISVill_2"])
    v2.metric("District",   nearest_village["DISTRICT"])
    v3.metric("SOC (%)", f"{float(nearest_village['SOC']):.2f}")
    v4.metric("Depth (cm)", int(nearest_village["DEPTH"]))
    v5.metric("Texture",    texture_soil_type(nearest_village['TEXTURE']))
    v6.metric("pH",         f"{float(nearest_village['PH']):.2f}")

    map_df = pd.DataFrame({
        "lat": [input_lat, float(nearest_village["latitude"])],
        "lon": [input_lon, float(nearest_village["longitude"])],
    })
    st.map(map_df)

elif record is not None:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Village",    record["KGISVill_2"])
    c2.metric("District",   record["DISTRICT"])
    c3.metric("SOC (%)", f"{float(record['SOC']):.2f}")
    c4.metric("Depth (cm)", int(record["DEPTH"]))
    c5.metric("Texture",    texture_soil_type(record['TEXTURE']))
    c6.metric("pH",         f"{float(record['PH']):.2f}")
    st.map(pd.DataFrame({"lat": [record["latitude"]], "lon": [record["longitude"]]}))

# ══════════════════════════════════════════════════════════════
# Weather
# ══════════════════════════════════════════════════════════════
if record is not None:
    st.markdown("---")
    st.markdown("#### 🌦️ Weather Forecast")

    if search_mode == "Latitude & Longitude":
        weather_lat, weather_lon = input_lat, input_lon
    else:
        weather_lat, weather_lon = float(record["latitude"]), float(record["longitude"])

    weather_data = fetch_weather(weather_lat, weather_lon)

    if weather_data:
        current = weather_data.get("current", {})
        daily = weather_data.get("daily", {})

        w1, w2, w3 = st.columns(3)
        w1.metric("Temperature", f"{current.get('temperature_2m', 'N/A')} °C")
        w2.metric("Humidity", f"{current.get('relative_humidity_2m', 'N/A')}%")
        w3.metric("Current Rain", f"{current.get('precipitation', 'N/A')} mm")

        precip_sum = daily.get("precipitation_sum", [])
        if precip_sum:
            st.info(f"🌧️ {rainfall_advice(precip_sum)}")

        temp_max = daily.get("temperature_2m_max", [])
        temp_min = daily.get("temperature_2m_min", [])
        dates = daily.get("time", [])
        if dates:
            forecast_df = pd.DataFrame({
                "Date": dates,
                "Max Temp (°C)": temp_max,
                "Min Temp (°C)": temp_min,
                "Rain (mm)": precip_sum,
            })
            st.dataframe(forecast_df, width='stretch', hide_index=True)
    else:
        st.warning("Weather data unavailable right now.")

    # ══════════════════════════════════════════════════════════════
    # PDF export
    # ══════════════════════════════════════════════════════════════
    st.markdown("---")
    if st.button("📄 Export PDF Report", key="btn_export_pdf"):
        pdf_buffer = generate_pdf_report(record, village_name=location_label)
        st.download_button(
            label="⬇️ Download Report",
            data=pdf_buffer,
            file_name=f"soil_report_{location_label.split()[0].replace(',', '')}.pdf",
            mime="application/pdf"
        )

# ══════════════════════════════════════════════════════════════
# Voice input
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("🎤 Voice Input")
audio = mic_recorder(
    start_prompt="🎙️ Start Recording",
    stop_prompt="⏹️ Stop Recording",
    just_once=True,
    use_container_width=True
)

voice_query = None
if audio:
    try:
        st.audio(audio["bytes"])
        with open("voice.wav", "wb") as f:
            f.write(audio["bytes"])
        transcription = groq_client.audio.transcriptions.create(
            file=open("voice.wav", "rb"),
            model="whisper-large-v3"
        )
        voice_query = transcription.text
        try:
            voice_query = GoogleTranslator(source="auto", target="en").translate(voice_query)
        except Exception:
            pass
        st.success(f"You said: {voice_query}")
    except Exception as e:
        st.error(f"Voice error: {e}")

# ══════════════════════════════════════════════════════════════
# Chat
# ══════════════════════════════════════════════════════════════
text_query = st.text_input("💬 Ask about this location's soil, or ask e.g. 'highest SOC villages in Belagavi', 'pH between 6 and 7', or 'SOC nearest to 6.2'")
query = voice_query if voice_query else text_query

if query:
    st.chat_message("user").write(query)

    compare_names = detect_compare_query(query)
    ranking = None if compare_names else detect_ranking_query(query)
    range_q = None if (compare_names or ranking) else detect_range_query(query)
    nearest = None if (compare_names or ranking or range_q) else detect_nearest_query(query)
    already_rendered = False

    if compare_names:
        # "compare X and Y" / "X vs Y" — doesn't need a selected location
        name_a, name_b = compare_names
        rec_a = village_df[village_df["KGISVill_2"] == name_a].iloc[0]
        rec_b = village_df[village_df["KGISVill_2"] == name_b].iloc[0]
        compare_df, score_a, score_b = compare_villages(rec_a, rec_b)

        st.chat_message("assistant").dataframe(compare_df, width='stretch', hide_index=True)
        if score_a != score_b:
            winner = name_a if score_a > score_b else name_b
            answer = f"🏆 {winner} has better overall soil fertility ({max(score_a, score_b)}/4 vs {min(score_a, score_b)}/4)."
        else:
            answer = f"{name_a} and {name_b} have equal fertility scores ({score_a}/4)."
        st.write(answer)
        already_rendered = True

    elif ranking:
        # Dataset-wide ranking query — falls back to current map/dropdown
        # scope when no district/sub-district is named in the query
        param_col, order, district_filter, subdist_filter = ranking
        result_df = rank_villages(param_col, order, district_filter, subdist_filter)
        if result_df.empty:
            answer = "No matching data found for that query."
            st.chat_message("assistant").write(answer)
        else:
            answer = format_ranking_answer(param_col, order, district_filter, subdist_filter, result_df)
            st.chat_message("assistant").write(answer)

            dl1, dl2 = st.columns(2)
            dl1.download_button(
                "⬇️ Download CSV", data=ranking_result_to_csv_bytes(result_df, param_col),
                file_name="soil_ranking.csv", mime="text/csv", key="ranking_csv_dl",
            )
            dl2.download_button(
                "⬇️ Download PDF", data=generate_ranking_pdf(result_df, param_col, order, district_filter, subdist_filter),
                file_name="soil_ranking.pdf", mime="application/pdf", key="ranking_pdf_dl",
            )
        already_rendered = True

    elif range_q:
        # "pH between 6 and 7" / "SOC 0.5 to 0.75 in Mysuru" — falls back to
        # current map/dropdown scope when no place is named in the query
        param_col, lo, hi, district_filter, subdist_filter = range_q
        result_df = rank_villages_range(param_col, lo, hi, district_filter, subdist_filter)
        if result_df.empty:
            answer = "No matching data found for that range."
            st.chat_message("assistant").write(answer)
        else:
            answer = format_range_answer(param_col, lo, hi, district_filter, subdist_filter, result_df)
            st.chat_message("assistant").write(answer)

            dl1, dl2 = st.columns(2)
            dl1.download_button(
                "⬇️ Download CSV", data=range_result_to_csv_bytes(result_df, param_col),
                file_name="soil_range.csv", mime="text/csv", key="range_csv_dl",
            )
            dl2.download_button(
                "⬇️ Download PDF", data=generate_range_pdf(result_df, param_col, lo, hi, district_filter, subdist_filter),
                file_name="soil_range.pdf", mime="application/pdf", key="range_pdf_dl",
            )
        already_rendered = True

    elif nearest:
        # "SOC nearest to 6.2" / "pH close to 7 in Mysuru" — doesn't need a selected location
        param_col, target_value, district_filter = nearest
        result_df = rank_villages_nearest(param_col, target_value, district_filter)
        if result_df.empty:
            answer = "No matching data found for that query."
            st.chat_message("assistant").write(answer)
        else:
            answer = format_nearest_answer(param_col, target_value, district_filter, result_df)
            st.chat_message("assistant").write(answer)

            dl1, dl2 = st.columns(2)
            dl1.download_button(
                "⬇️ Download CSV", data=nearest_result_to_csv_bytes(result_df, param_col),
                file_name="soil_nearest.csv", mime="text/csv", key="nearest_csv_dl",
            )
            dl2.download_button(
                "⬇️ Download PDF", data=generate_nearest_pdf(result_df, param_col, target_value, district_filter),
                file_name="soil_nearest.pdf", mime="application/pdf", key="nearest_pdf_dl",
            )
        already_rendered = True

    elif record is not None:
        answer = keyword_response(query, record, location_name=location_label)
        if answer is None:
            context = f"""Soil expert for Karnataka. Data:
Location: {location_label}
SOC: {record.get('SOC', 'N/A')}%, Depth: {record.get('DEPTH', 'N/A')} cm, Texture: {texture_soil_type(record.get('TEXTURE', 0))}, pH: {record.get('PH', 'N/A')}
Question: {query}. Be concise."""
            try:
                res = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": context}],
                    max_tokens=500
                )
                answer = res.choices[0].message.content
            except Exception as e:
                answer = f"AI unavailable: {e}"
    else:
        answer = "Select a location first, or ask a dataset-wide question like 'areas with high pH in Tumakuru', 'lowest SOC villages', 'pH between 6 and 7', or 'SOC nearest to 6.2'."

    if not already_rendered:
        st.chat_message("assistant").write(answer)
    audio_file = speak_text(str(answer).replace("#", "").replace("*", ""))
    if audio_file:
        st.audio(audio_file)