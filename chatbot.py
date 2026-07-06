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
village_boundaries_gdf = load_village_boundaries()

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
                        border_color="#1a4d1a", color_by_feature=False):
    # Fill in missing/blank names so the tooltip never renders as an empty pill
    for feature in geojson_data.get("features", []):
        props = feature.get("properties", {})
        val = props.get(name_key)
        if val is None or str(val).strip() == "":
            props[name_key] = "Unknown"

    def style_fn(f):
        if color_by_feature:
            c = color_for_name(f["properties"].get(name_key, ""))
            return {"fillColor": c, "color": "#333333", "weight": 1.2, "fillOpacity": 0.45}
        return {"fillColor": fill_color, "color": border_color, "weight": 1.5, "fillOpacity": 0.25}

    folium.GeoJson(
        geojson_data,
        style_function=style_fn,
        highlight_function=lambda f: {"fillColor": "#ffb300", "color": "#e65100", "weight": 2.5, "fillOpacity": 0.55},
        tooltip=folium.GeoJsonTooltip(
            fields=[name_key],
            labels=False,      # show only the value, no field name prefix
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

def reset_map_selection():
    st.session_state.map_level = "district"
    st.session_state.sel_district = None
    st.session_state.sel_subdist = None
    st.session_state.sel_village = None
    st.session_state.zoom_bounds = None

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
        st.rerun()
    if st.session_state.sel_subdist and b3.button(f"◀ {st.session_state.sel_subdist}", key="btn_back_subdist"):
        st.session_state.map_level = "subdistrict"
        st.session_state.sel_subdist = None
        st.session_state.sel_village = None
        st.rerun()

    # ---- LEVEL 1: districts ----
    if st.session_state.map_level == "district":
        district_gdf = get_district_boundaries()
        m = folium.Map(location=[15.3, 75.7], zoom_start=7, tiles="CartoDB positron")
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
                st.rerun()

    # ---- LEVEL 2: sub-districts ----
    elif st.session_state.map_level == "subdistrict":
        taluk_gdf = get_taluk_boundaries(st.session_state.sel_district)
        if taluk_gdf.empty:
            st.error(f"No boundary data found for district '{st.session_state.sel_district}'. Click Reset.")
        else:
            m = folium.Map(tiles="CartoDB positron")
            m.fit_bounds(st.session_state.zoom_bounds)
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
                    st.rerun()

    # ---- LEVEL 3: villages ----
    elif st.session_state.map_level == "village":
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
            m = folium.Map(tiles="CartoDB positron")
            m.fit_bounds(st.session_state.zoom_bounds)
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
    st.dataframe(compare_df, use_container_width=True, hide_index=True)

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
    st.info("Select a valid location to view soil data.")
    st.stop()

if search_mode == "Latitude & Longitude":
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

else:
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
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
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
text_query = st.text_input("💬 Ask about this location's soil")
query = voice_query if voice_query else text_query

if query and record is not None:
    st.chat_message("user").write(query)
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

    st.chat_message("assistant").write(answer)
    audio_file = speak_text(str(answer).replace("#", "").replace("*", ""))
    if audio_file:
        st.audio(audio_file)