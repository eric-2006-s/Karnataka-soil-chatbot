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
from scipy.spatial import cKDTree
from groq import Groq
import edge_tts
from deep_translator import GoogleTranslator
from streamlit_mic_recorder import mic_recorder
import tempfile
import io

groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

BASE = os.path.dirname(os.path.abspath(__file__))
CSV_URL = "https://huggingface.co/datasets/ricu9656/karnataka-soil-data/resolve/main/Export_Output.csv"
CSV_PATH = os.path.join(BASE, "Export_Output.csv")
VILLAGE_GEOJSON_PATH  = os.path.join(BASE, "village_boundaries_simplified_v3.geojson")
DISTRICT_GEOJSON_PATH = os.path.join(BASE, "district_boundaries.geojson")
TALUK_GEOJSON_PATH    = os.path.join(BASE, "taluk_boundaries.geojson")

# ── Data loading ──────────────────────────────────────────────
@st.cache_data
def load_village_data():
    df = pd.read_excel(os.path.join(BASE, "combined_village_data.xlsx"), engine="openpyxl")
    df = df[df['KGISVill_2'].notna() & (df['KGISVill_2'].str.strip() != '')]
    df = df[df['latitude'].notna() & df['longitude'].notna()]
    for col in ["DISTRICT", "SUB_DIST", "KGISVill_2"]:
        df[col] = df[col].astype(str).str.strip()
    for col in ["DISTRICT", "SUB_DIST"]:
        df[col] = df[col].astype("category")
    float_cols = [c for c in df.select_dtypes(include="float64").columns
                  if c not in ("latitude", "longitude")]
    df[float_cols] = df[float_cols].astype("float32")
    return df

def _download_csv_from_hf():
    """Downloads Export_Output.csv from the HuggingFace dataset repo.

    Plain requests.get() on the /resolve/main/... URL gets redirected to a
    signed HuggingFace 'Xet' CDN URL (xet-bridge-us...), and those signed
    redirects intermittently return 403 Forbidden — a known reliability issue
    with HF's newer Xet storage backend, not something specific to this repo
    or this code. huggingface_hub's own downloader handles Xet negotiation,
    retries, and resumable downloads correctly; we also explicitly disable
    Xet (HF_HUB_DISABLE_XET=1), which falls back to the older, more reliable
    plain-HTTP CDN path and is the community-documented fix for this 403.
    """
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        from huggingface_hub import hf_hub_download
        downloaded_path = hf_hub_download(
            repo_id="ricu9656/karnataka-soil-data",
            repo_type="dataset",
            filename="Export_Output.csv",
        )
        import shutil
        shutil.copy(downloaded_path, CSV_PATH)
        return
    except Exception:
        pass  # fall through to the plain-requests fallback below

    # Fallback: plain HTTP with a browser-like User-Agent (some CDN edges
    # reject the default python-requests UA) and a couple of retries, in
    # case huggingface_hub itself isn't available or also fails.
    headers = {"User-Agent": "Mozilla/5.0 (compatible; SoilMitraAI/1.0)"}
    last_err = None
    for attempt in range(3):
        try:
            r = requests.get(CSV_URL, headers=headers, timeout=60)
            r.raise_for_status()
            with open(CSV_PATH, "wb") as f:
                f.write(r.content)
            return
        except Exception as e:
            last_err = e
    raise last_err

@st.cache_data
def load_csv_data():
    if not os.path.exists(CSV_PATH):
        with st.spinner("Downloading soil dataset (first run only)..."):
            _download_csv_from_hf()
    df = pd.read_csv(CSV_PATH)
    df = df.rename(columns={"Depth": "DEPTH", "pH": "PH", "Texture": "TEXTURE", "Longitude": "longitude"})
    df = df[df['latitude'].notna() & df['longitude'].notna()]
    float_cols = [c for c in df.select_dtypes(include="float64").columns
                  if c not in ("latitude", "longitude")]
    df[float_cols] = df[float_cols].astype("float32")
    return df

@st.cache_data
def load_village_boundaries():
    import geopandas as gpd
    gdf = gpd.read_file(VILLAGE_GEOJSON_PATH)
    gdf = gdf[gdf["DISTRICT"].notna()]
    gdf["DISTRICT"]   = gdf["DISTRICT"].astype(str).str.strip()
    gdf["SUB_DIST"]   = gdf["SUB_DIST"].astype(str).str.strip()
    gdf["KGISVill_2"] = gdf["KGISVill_2"].astype(str).str.strip()
    gdf.loc[gdf["KGISVill_2"].isin(["None", "nan", ""]), "KGISVill_2"] = None
    missing_mask = gdf["KGISVill_2"].isna()
    if missing_mask.any():
        vdf = load_village_data()
        filled_names = []
        for idx in gdf.loc[missing_mask].index:
            row = gdf.loc[idx]
            centroid   = row.geometry.centroid
            candidates = vdf[(vdf["DISTRICT"] == row["DISTRICT"]) & (vdf["SUB_DIST"] == row["SUB_DIST"])]
            if candidates.empty:
                filled_names.append(None)
                continue
            dists        = (candidates["latitude"] - centroid.y)**2 + (candidates["longitude"] - centroid.x)**2
            filled_names.append(candidates.loc[dists.idxmin(), "KGISVill_2"])
        gdf.loc[missing_mask, "KGISVill_2"] = filled_names
        gdf["KGISVill_2"] = gdf["KGISVill_2"].fillna("Unknown")
    return gdf

@st.cache_data
def get_district_boundaries():
    import geopandas as gpd
    gdf = gpd.read_file(DISTRICT_GEOJSON_PATH)
    gdf["DISTRICT"] = gdf["DISTRICT"].astype(str).str.strip()
    return gdf

@st.cache_data
def get_taluk_boundaries(district):
    import geopandas as gpd
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

village_df   = load_village_data()
csv_df       = load_csv_data()
village_tree = get_village_tree(village_df)
csv_tree     = get_csv_tree(csv_df)

# ── IDW ───────────────────────────────────────────────────────
def idw_estimate(tree, df, lat, lon, columns, k=4, power=2, max_dist_deg=0.01):
    distances, indices = tree.query([[lat, lon]], k=k)
    distances = distances[0]; indices = indices[0]
    if distances[0] > max_dist_deg: return None
    if distances[0] < 1e-12: return df.iloc[indices[0]][columns].astype(float)
    weights = 1.0 / (distances ** power); weights /= weights.sum()
    return pd.Series({col: float((df.iloc[indices][col].astype(float).values * weights).sum()) for col in columns})

# ── Interpretation ────────────────────────────────────────────
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

TEXTURE_BANDS = [(1.45,"Clay"),(1.48,"Sandy Clay"),(1.49,"Clay Loam"),(1.54,"Loam"),
                 (1.56,"Sandy Clay Loam"),(1.63,"Sandy Loam"),(1.65,"Loamy Sand"),(1.69,"Sand")]
TEXTURE_NOTES = {"Clay":"Heavy, high water retention, poor drainage.","Sandy Clay":"Retains water well, drains slowly.",
                 "Clay Loam":"Balanced retention, moderate drainage.","Loam":"Ideal for most crops.",
                 "Sandy Clay Loam":"Good balance of drainage and retention.","Sandy Loam":"Good drainage, decent retention.",
                 "Loamy Sand":"Fast drainage, low retention.","Sand":"Very fast drainage, low water retention."}
TEXTURE_TYPE_COLORS = {"Clay":"#5b3a29","Sandy Clay":"#8a5a3c","Clay Loam":"#a97450","Loam":"#6b8e23",
                       "Sandy Clay Loam":"#c9a227","Sandy Loam":"#e0b96b","Loamy Sand":"#f0d18c","Sand":"#f5e6b8"}

def texture_soil_type(v):
    v = float(v)
    for upper, label in TEXTURE_BANDS:
        if v <= upper: return label
    return "Sand"

def texture_interp(v):
    t = texture_soil_type(v)
    return f"{t} — {TEXTURE_NOTES.get(t,'')}"

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
    a = math.sin((phi2-phi1)/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(math.radians(lon2-lon1)/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

# ── Weather ───────────────────────────────────────────────────
@st.cache_data(ttl=1800)
def fetch_weather(lat, lon):
    try:
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
               "&current=temperature_2m,relative_humidity_2m,precipitation"
               "&daily=precipitation_sum,temperature_2m_max,temperature_2m_min&timezone=auto&forecast_days=5")
        r = requests.get(url, timeout=10); r.raise_for_status(); return r.json()
    except: return None

def rainfall_advice(p):
    total = sum(p) if p else 0
    if total < 5:   return f"{total:.1f} mm over 5 days — Dry spell. Irrigation likely needed."
    elif total < 25: return f"{total:.1f} mm over 5 days — Moderate rainfall. Monitor soil moisture."
    else:            return f"{total:.1f} mm over 5 days — Heavy rainfall. Watch for waterlogging."

# ── Keyword chat ──────────────────────────────────────────────
def keyword_response(query, record, location_name="Selected location"):
    q = query.lower(); name = location_name
    if any(w in q for w in ["soc","organic carbon","carbon"]):
        return f"**SOC — {name}:** {soc_interp(record['SOC'])}"
    if "depth" in q:
        return f"**Depth — {name}:** {depth_interp(record['DEPTH'])}"
    if "texture" in q or "soil type" in q:
        return f"**Texture — {name}:** {texture_interp(record['TEXTURE'])}"
    if any(w in q for w in ["ph","acidity","acidic","alkaline"]):
        return f"**pH — {name}:** {ph_interp(record['PH'])}"
    if any(w in q for w in ["summary","all","profile","details","overview"]):
        return (f"**Soil Summary — {name}**\n\n"
                f"- SOC: {soc_interp(record['SOC'])}\n- Depth: {depth_interp(record['DEPTH'])}\n"
                f"- Texture: {texture_interp(record['TEXTURE'])}\n- pH: {ph_interp(record['PH'])}")
    if any(w in q for w in ["fertile","fertility","soil quality"]):
        score = fertility_score(record); rating = ["Poor","Low","Moderate","Good","Excellent"][score]
        return f"**Fertility — {name}:** {rating} ({score}/4)"
    return None

# ── Compare ───────────────────────────────────────────────────
def compare_villages(rec_a, rec_b):
    rows = [
        ("SOC (%)",    f"{float(rec_a['SOC']):.2f}",        f"{float(rec_b['SOC']):.2f}",        float(rec_a['SOC'])-float(rec_b['SOC'])),
        ("Depth (cm)", f"{round(float(rec_a['DEPTH']))}",   f"{round(float(rec_b['DEPTH']))}",   float(rec_a['DEPTH'])-float(rec_b['DEPTH'])),
        ("Texture",    texture_soil_type(rec_a['TEXTURE']), texture_soil_type(rec_b['TEXTURE']), None),
        ("pH",         f"{float(rec_a['PH']):.2f}",         f"{float(rec_b['PH']):.2f}",         float(rec_a['PH'])-float(rec_b['PH'])),
    ]
    sa, sb = fertility_score(rec_a), fertility_score(rec_b)
    rating = ["Poor","Low","Moderate","Good","Excellent"]
    rows.append(("Fertility", f"{rating[sa]} ({sa}/4)", f"{rating[sb]} ({sb}/4)", sa-sb))
    df = pd.DataFrame(rows, columns=["Parameter", str(rec_a['KGISVill_2']), str(rec_b['KGISVill_2']), "Diff (A − B)"])
    return df, sa, sb

# ── Ranking / range / nearest helpers ─────────────────────────
PARAM_KEYWORDS = [
    (["organic carbon","soc"],"SOC"),
    (["ph","acidity","acidic","alkaline"],"PH"),
    (["depth"],"DEPTH"),
    (["texture"],"TEXTURE"),
    (["fertility","fertile"],"FERTILITY"),
]
HIGH_WORDS = ["highest","high ","most","top","maximum","max "]
LOW_WORDS  = ["lowest","low ","least","minimum","min "]
PARAM_LABELS = {"SOC":"SOC (%)","PH":"pH","DEPTH":"Depth (cm)","TEXTURE":"Texture (bulk density)","FERTILITY":"Fertility score"}
PARAM_VALID_RANGES = {"SOC":(0.0,20.0),"PH":(2.0,12.0),"DEPTH":(0.0,300.0),"TEXTURE":(1.0,2.2)}

def filter_valid_range(df, param_col):
    if param_col not in PARAM_VALID_RANGES: return df
    lo, hi = PARAM_VALID_RANGES[param_col]
    return df[(df[param_col] >= lo) & (df[param_col] <= hi)]

def _scope_label(d, s):
    if s and d: return f"{s}, {d}"
    elif d: return d
    else: return "Karnataka"

def detect_ranking_query(q):
    ql = q.lower()
    order = "high" if any(w in ql for w in HIGH_WORDS) else ("low" if any(w in ql for w in LOW_WORDS) else None)
    if not order: return None
    param_col = next((col for kws,col in PARAM_KEYWORDS if any(k in ql for k in kws)), None)
    if not param_col: return None
    district_filter = next((d for d in village_df["DISTRICT"].dropna().astype(str).unique() if d.lower() in ql), None)
    sub_scope = village_df[village_df["DISTRICT"].astype(str)==district_filter] if district_filter else village_df
    subdist_filter = next((s for s in sub_scope["SUB_DIST"].dropna().astype(str).unique() if s.lower() in ql), None)
    if not district_filter and not subdist_filter:
        district_filter = st.session_state.get("current_district")
        subdist_filter  = st.session_state.get("current_subdist")
    return param_col, order, district_filter, subdist_filter

def rank_villages(param_col, order="high", district_filter=None, subdist_filter=None, n=10):
    df = village_df.copy()
    if district_filter: df = df[df["DISTRICT"].astype(str)==district_filter]
    if subdist_filter:  df = df[df["SUB_DIST"].astype(str)==subdist_filter]
    if param_col == "FERTILITY":
        for col in ["SOC","DEPTH","TEXTURE","PH"]: df = filter_valid_range(df.dropna(subset=[col]),col)
        if df.empty: return df
        df = df.copy(); df["FERTILITY"] = df.apply(fertility_score, axis=1); sort_col = "FERTILITY"
    else:
        df = filter_valid_range(df[df[param_col].notna()], param_col); sort_col = param_col
    return df.sort_values(sort_col, ascending=(order=="low")).head(n)

def format_ranking_answer(param_col, order, df, district_filter, subdist_filter):
    scope = _scope_label(district_filter, subdist_filter)
    label = PARAM_LABELS.get(param_col, param_col)
    lines = [f"**{'Highest' if order=='high' else 'Lowest'} {label} — {scope}**\n"]
    for _, row in df.iterrows():
        if param_col=="TEXTURE": v = texture_soil_type(row["TEXTURE"])
        elif param_col=="DEPTH": v = f"{round(float(row['DEPTH']))} cm"
        elif param_col=="FERTILITY": v = f"{['Poor','Low','Moderate','Good','Excellent'][int(row['FERTILITY'])]} ({int(row['FERTILITY'])}/4)"
        else: v = f"{float(row[param_col]):.2f}"
        lines.append(f"- {row['KGISVill_2']} ({row['SUB_DIST']}, {row['DISTRICT']}): {v}")
    return "\n".join(lines)

def detect_range_query(q):
    import re
    ql = q.lower()
    if not any(w in ql for w in ["between"," to ","-"," and "]): return None
    param_col = next((col for kws,col in PARAM_KEYWORDS if col!="FERTILITY" and any(k in ql for k in kws)), None)
    if not param_col: return None
    m = re.search(r'(\d+\.?\d*)\s*(?:-|to|and)\s*(\d+\.?\d*)', ql)
    if not m: return None
    lo, hi = min(float(m.group(1)),float(m.group(2))), max(float(m.group(1)),float(m.group(2)))
    if lo==hi: return None
    district_filter = next((d for d in village_df["DISTRICT"].dropna().astype(str).unique() if d.lower() in ql), None)
    sub_scope = village_df[village_df["DISTRICT"].astype(str)==district_filter] if district_filter else village_df
    subdist_filter = next((s for s in sub_scope["SUB_DIST"].dropna().astype(str).unique() if s.lower() in ql), None)
    if not district_filter and not subdist_filter:
        district_filter = st.session_state.get("current_district")
        subdist_filter  = st.session_state.get("current_subdist")
    return param_col, lo, hi, district_filter, subdist_filter

def rank_villages_range(param_col, lo, hi, district_filter=None, subdist_filter=None, n=50):
    df = village_df.copy()
    if district_filter: df = df[df["DISTRICT"].astype(str)==district_filter]
    if subdist_filter:  df = df[df["SUB_DIST"].astype(str)==subdist_filter]
    df = filter_valid_range(df[df[param_col].notna()], param_col)
    if df.empty: return df
    return df[(df[param_col].astype(float)>=lo)&(df[param_col].astype(float)<=hi)].sort_values(param_col).head(n)

def format_range_answer(param_col, lo, hi, df, district_filter, subdist_filter):
    scope = _scope_label(district_filter, subdist_filter)
    label = PARAM_LABELS.get(param_col, param_col)
    lines = [f"**{label} between {lo:g} and {hi:g} — {scope}** ({len(df)} matches)\n"]
    for _, row in df.head(20).iterrows():
        if param_col=="TEXTURE": v = texture_soil_type(row["TEXTURE"])
        elif param_col=="DEPTH": v = f"{round(float(row['DEPTH']))} cm"
        else: v = f"{float(row[param_col]):.2f}"
        lines.append(f"- {row['KGISVill_2']} ({row['SUB_DIST']}, {row['DISTRICT']}): {v}")
    if len(df)>20: lines.append(f"\n_...and {len(df)-20} more. Download CSV/PDF for the full list._")
    return "\n".join(lines)

def detect_nearest_query(q):
    import re
    ql = q.lower()
    if not any(w in ql for w in ["nearest to","closest to","close to","near ","around ","approx"]): return None
    param_col = next((col for kws,col in PARAM_KEYWORDS if col!="FERTILITY" and any(k in ql for k in kws)), None)
    if not param_col: return None
    m = re.search(r'(\d+\.?\d*)', ql)
    if not m: return None
    target = float(m.group(1))
    district_filter = next((d for d in village_df["DISTRICT"].dropna().astype(str).unique() if d.lower() in ql), None)
    return param_col, target, district_filter

def rank_villages_nearest(param_col, target, district_filter=None, n=10):
    df = village_df.copy()
    if district_filter: df = df[df["DISTRICT"].astype(str)==district_filter]
    df = filter_valid_range(df[df[param_col].notna()], param_col)
    if df.empty: return df
    df = df.copy(); df["_diff"] = (df[param_col].astype(float)-target).abs()
    return df.sort_values("_diff").head(n)

def format_nearest_answer(param_col, target, df, district_filter):
    scope = district_filter if district_filter else "Karnataka"
    label = PARAM_LABELS.get(param_col, param_col)
    lines = [f"**{label} closest to {target:g} — {scope}**\n"]
    for _, row in df.iterrows():
        if param_col=="TEXTURE": v = texture_soil_type(row["TEXTURE"])
        elif param_col=="DEPTH": v = f"{round(float(row['DEPTH']))} cm"
        else: v = f"{float(row[param_col]):.2f}"
        lines.append(f"- {row['KGISVill_2']} ({row['SUB_DIST']}, {row['DISTRICT']}): {v} (Δ {row['_diff']:.2f})")
    return "\n".join(lines)

def detect_compare_query(q):
    import re
    ql = q.lower()
    if not any(w in ql for w in ["compare"," vs ","versus"]): return None
    q_clean = re.sub(r'^.*?compare\s+','',ql) if "compare" in ql else ql
    q_clean = re.sub(r'\bsoil\b|\bfertility\b|\bdata\b','',q_clean)
    parts = [p.strip() for p in re.split(r'\s+(?:vs\.?|versus|and)\s+|,',q_clean) if p.strip()]
    if len(parts)<2: return None
    all_villages = village_df["KGISVill_2"].dropna().astype(str).unique()
    matched = []
    for part in parts[:2]:
        hit = next((v for v in all_villages if part==v.lower()),None) or \
              next((v for v in all_villages if part in v.lower() or v.lower() in part),None)
        if hit: matched.append(hit)
    if len(matched)<2 or matched[0]==matched[1]: return None
    return matched[0], matched[1]

# ── PDF helpers ───────────────────────────────────────────────
def _pdf_table_style():
    from reportlab.lib import colors
    from reportlab.platypus import TableStyle
    return TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#2e7d32')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,-1),9),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f7f0')]),
        ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#c8e6c9')),
        ('PADDING',(0,0),(-1,-1),6),
        ('TEXTCOLOR',(0,1),(-1,-1),colors.HexColor('#1a3a1a')),
    ])

def generate_pdf_report(record, village_name=None):
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm, leftMargin=2*cm, rightMargin=2*cm)
    styles = getSampleStyleSheet()
    title_style   = ParagraphStyle('title',   parent=styles['Title'],   fontSize=18, textColor=colors.HexColor('#1a4d1a'), spaceAfter=6)
    heading_style = ParagraphStyle('heading', parent=styles['Heading2'], fontSize=13, textColor=colors.HexColor('#2e7d32'), spaceAfter=4)
    normal_style  = ParagraphStyle('normal',  parent=styles['Normal'],  fontSize=11, textColor=colors.HexColor('#1a3a1a'), spaceAfter=4)
    story = [Paragraph("Karnataka Soil Report", title_style),
             Paragraph(f"Location: {village_name or 'Custom location'}", heading_style), Spacer(1,0.3*cm),
             Paragraph("Soil Parameters", heading_style)]
    soil_data = [["Parameter","Value","Interpretation"],
                 ["SOC (%)",   f"{float(record['SOC']):.2f}",        soc_interp(record['SOC'])],
                 ["Depth (cm)",str(round(float(record['DEPTH']))),   depth_interp(record['DEPTH'])],
                 ["Texture",   texture_soil_type(record['TEXTURE']), texture_interp(record['TEXTURE'])],
                 ["pH",        f"{float(record['PH']):.2f}",         ph_interp(record['PH'])]]
    t = Table(soil_data, colWidths=[4*cm,3*cm,10*cm]); t.setStyle(_pdf_table_style()); story.append(t)
    story.append(Spacer(1,0.5*cm))
    score = fertility_score(record); rating = ["Poor","Low","Moderate","Good","Excellent"][score]
    story += [Paragraph("Overall Fertility", heading_style),
              Paragraph(f"Rating: <b>{rating}</b> ({score}/4 parameters optimal)", normal_style),
              Spacer(1,1*cm),
              Paragraph("Generated by Karnataka Soil Chatbot",
                        ParagraphStyle('footer',parent=styles['Normal'],fontSize=9,textColor=colors.grey))]
    doc.build(story); buffer.seek(0); return buffer

def generate_table_pdf(rows, headers, title_text, subtitle_text):
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm, leftMargin=2*cm, rightMargin=2*cm)
    styles = getSampleStyleSheet()
    title_style   = ParagraphStyle('t', parent=styles['Title'],   fontSize=16, textColor=colors.HexColor('#1a4d1a'), spaceAfter=4)
    heading_style = ParagraphStyle('h', parent=styles['Heading2'], fontSize=12, textColor=colors.HexColor('#2e7d32'), spaceAfter=4)
    col_w = [17*cm/len(headers)]*len(headers)
    t = Table([headers]+rows, colWidths=col_w); t.setStyle(_pdf_table_style())
    doc.build([Paragraph(title_text, title_style), Paragraph(subtitle_text, heading_style), Spacer(1,0.3*cm), t])
    buffer.seek(0); return buffer

def result_to_csv(df, cols):
    return df[cols].to_csv(index=False).encode("utf-8")

# ── TTS ───────────────────────────────────────────────────────
async def _speak_async(text, voice="en-IN-NeerjaNeural"):
    c = edge_tts.Communicate(str(text), voice=voice)
    p = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    await c.save(p); return p

def speak_text(text):
    try: return asyncio.run(_speak_async(text))
    except: return None

# ── Map helpers ───────────────────────────────────────────────
def gdf_to_geojson(gdf):
    return json.loads(gdf.to_json())

def feature_bounds_from_geom(geom):
    minx, miny, maxx, maxy = geom.bounds
    return [[miny, minx], [maxy, maxx]]

DISTRICT_PALETTE = [
    "#e57373","#64b5f6","#81c784","#ffb74d","#ba68c8","#4db6ac","#f06292","#a1887f","#90a4ae","#dce775",
    "#4fc3f7","#ff8a65","#9575cd","#aed581","#7986cb","#fff176","#4dd0e1","#f8bbd0","#c5e1a5","#ffd54f",
    "#ef9a9a","#80cbc4","#ce93d8","#fff59d","#b0bec5","#c5cae9","#ffcc80","#a5d6a7","#f48fb1","#bcaaa4",
]

def color_for_name(name):
    h = int(hashlib.md5(str(name).encode()).hexdigest(), 16)
    return DISTRICT_PALETTE[h % len(DISTRICT_PALETTE)]

def add_boundary_layer(m, geojson_data, name_key, fill_color="#4caf50", border_color="#1a4d1a",
                       color_by_feature=False, value_color_map=None, value_labels=None, default_color="#cccccc",
                       satellite_toggle_layers=None, border_only=False):
    """
    border_only: when True, the layer always renders as outline-only (no fill, no hover-fill),
    independent of which base tile layer is active. This is a user-facing toggle, separate from
    the satellite auto-hide behaviour below — both can be true at once with no conflict, since
    "no fill" is the union of (border_only OR satellite active).

    If satellite_toggle_layers is given (a list of the actual folium.TileLayer objects that are
    'satellite' tiles on this map), the boundary layer is rendered with NO fill and NO
    hover-highlight while one of those tiles is the active base layer, and automatically gets
    its fill + hover-highlight back the moment the user switches to a non-satellite base layer
    via the layer control — live, with no rerun, since it's driven by Leaflet's own
    'baselayerchange' event.

    IMPORTANT FIX (previous version): the injected <script> block referenced `window[mapVarName]`
    / `window[gjVarName]` to look up Folium's generated `map_xxx` / `geo_json_xxx` variables.
    That never worked: Folium wraps its entire generated script for one map in a single anonymous
    function, and declares every `map_xxx`, `tile_layer_xxx`, and `geo_json_xxx` with `var`
    *inside* that function. A `var` inside a function body is scoped to that function — it is
    NEVER attached to `window`, no matter how long you poll for it. So `window[mapVarName]` was
    always `undefined`, the polling loop always timed out, and the satellite no-fill behaviour
    silently never activated (exactly the bug seen in the screenshot: Google Satellite selected,
    boundaries still filled orange).

    THE FIX: our own injected script block is concatenated by Folium into that SAME function
    scope as the map/tile-layer/geojson declarations (it's appended via
    `m.get_root().script.add_child(...)`, which lands inside the same generated <script> tag).
    That means we can reference `map_var` / `gj_var` / the satellite tile-layer variable names
    directly as plain JS identifiers — no `window[...]` indirection required, and no polling
    required either (they're guaranteed to already be declared above us in the same scope by the
    time our IIFE runs). A `try/catch` + short `setTimeout` retry is kept only as a defensive
    fallback in case a future Folium version changes script ordering.

    Assumes the first non-overlay TileLayer added to the map is a satellite tile (so that's the
    state on first load). If satellite_toggle_layers is omitted, the layer just always shows
    filled + highlighted (original behaviour) — what district/sub-district maps (no satellite
    option) want.
    """
    import folium, json as _json

    for feature in geojson_data.get("features",[]):
        props = feature.get("properties",{})
        if not props.get(name_key) or str(props.get(name_key)).strip()=="":
            props[name_key] = "Unknown"
        if value_labels is not None:
            props["_value_label"] = value_labels.get(props[name_key],"No data")
        if value_color_map is not None:
            props["_fillColor"]   = value_color_map.get(props[name_key], default_color)
            props["_borderColor"] = "#333333"
            props["_fillOpacity"] = 0.65
        elif color_by_feature:
            props["_fillColor"]   = color_for_name(props[name_key])
            props["_borderColor"] = "#333333"
            props["_fillOpacity"] = 0.45
        else:
            props["_fillColor"]   = fill_color
            props["_borderColor"] = border_color
            props["_fillOpacity"] = 0.25
        if border_only:
            props["_fillOpacity"] = 0

    def style_fn(f):
        p = f["properties"]
        return {"fillColor": p["_fillColor"], "color": p["_borderColor"], "weight": 1.5, "fillOpacity": p["_fillOpacity"]}

    tooltip_fields = [name_key]+( ["_value_label"] if value_labels is not None else [])
    gj = folium.GeoJson(
        geojson_data, name="🗺️ Boundaries", style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields, labels=(value_labels is not None), sticky=True,
            style="background-color:#fff!important;color:#1a3a1a!important;font-weight:600!important;"
                  "font-size:13px!important;padding:4px 8px!important;border:1px solid #a5d6a7!important;"
                  "border-radius:4px!important;white-space:nowrap!important;",
        ),
    )
    gj.add_to(m)

    # Custom hover + base-layer-aware styling (replaces folium's static highlight_function,
    # which can't react to the user switching tile layers at runtime).
    gj_var  = gj.get_name()
    map_var = m.get_name()
    sat_var_list = [tl.get_name() for tl in (satellite_toggle_layers or [])]
    # Raw JS identifiers (NOT quoted strings) — these are the literal Folium-generated variable
    # names, e.g. tile_layer_a1b2c3. They must be emitted unquoted so the browser resolves them
    # as references to the actual objects already declared earlier in this same <script> scope.
    sat_vars_js = "[" + ",".join(sat_var_list) + "]"
    border_only_js = "true" if border_only else "false"

    script = f"""
    (function() {{
        function tryInit() {{
            try {{
                // map_var / gj_var / satellite tile-layer names are plain identifiers declared
                // earlier by Folium in this SAME enclosing function scope — reference them
                // directly, do not look them up on window.
                var mapObj = {map_var};
                var boundaryLayer = {gj_var};
                var satelliteLayers = {sat_vars_js}.filter(function(l) {{ return typeof l !== 'undefined'; }});
                var forceNoFill = {border_only_js};

                function isSatelliteActive() {{
                    for (var i = 0; i < satelliteLayers.length; i++) {{
                        if (mapObj.hasLayer(satelliteLayers[i])) return true;
                    }}
                    return false;
                }}

                // No fill whenever the user has explicitly asked for borders-only, OR while a
                // satellite base layer is active (the pre-existing auto-hide behaviour). These
                // two triggers are independent and just OR together.
                function isNoFillActive() {{ return forceNoFill || isSatelliteActive(); }}

                function baseStyle(feature) {{
                    var p = feature.properties;
                    var noFill = isNoFillActive();
                    return {{
                        fillColor: p._fillColor,
                        color: p._borderColor,
                        weight: noFill ? 2.5 : 1.5,
                        fillOpacity: noFill ? 0 : p._fillOpacity
                    }};
                }}
                function refreshStyle() {{ boundaryLayer.setStyle(baseStyle); }}

                boundaryLayer.eachLayer(function(layer) {{
                    layer.on('mouseover', function(e) {{
                        if (isNoFillActive()) return;
                        e.target.setStyle({{fillColor:'#ffb300', color:'#e65100', weight:2.5, fillOpacity:0.55}});
                    }});
                    layer.on('mouseout', function(e) {{
                        if (isNoFillActive()) return;
                        e.target.setStyle(baseStyle(e.target.feature));
                    }});
                }});

                if (satelliteLayers.length) {{
                    mapObj.on('baselayerchange', refreshStyle);
                    mapObj.on('layeradd', refreshStyle);
                    mapObj.on('layerremove', refreshStyle);
                    var lastKnown = isSatelliteActive();
                    setInterval(function() {{
                        var nowSat = isSatelliteActive();
                        if (nowSat !== lastKnown) {{ lastKnown = nowSat; refreshStyle(); }}
                    }}, 300);
                }}

                refreshStyle();

                setTimeout(function() {{ mapObj.invalidateSize(); }}, 250);
                setTimeout(function() {{ mapObj.invalidateSize(); }}, 800);
                window.addEventListener('resize', function() {{ mapObj.invalidateSize(); }});

                console.log('[boundary-satellite-toggle] initialized OK');
            }} catch (err) {{
                console.warn('[boundary-satellite-toggle] not ready yet, retrying:', err.message);
                setTimeout(tryInit, 50);
            }}
        }}
        tryInit();
    }})();
    """
    m.get_root().script.add_child(folium.Element(script))

CHOROPLETH_METRIC_OPTIONS = ["Default (by name)","SOC (avg)","pH (avg)","Depth (avg)","Texture (soil type)"]
CHOROPLETH_METRIC_COL_MAP = {"SOC (avg)":"SOC","pH (avg)":"PH","Depth (avg)":"DEPTH","Texture (soil type)":"TEXTURE"}

def compute_choropleth(df_source, group_col, param_col):
    if param_col == "TEXTURE":
        df = filter_valid_range(df_source.dropna(subset=["TEXTURE"]),"TEXTURE")
        if df.empty: return {},{},( None,None)
        df = df.copy(); df["_st"] = df["TEXTURE"].apply(texture_soil_type)
        dominant = df.groupby(group_col)["_st"].agg(lambda s: s.value_counts().idxmax())
        return ({n:TEXTURE_TYPE_COLORS.get(t,"#ccc") for n,t in dominant.items()},
                {n:t for n,t in dominant.items()}, (None,None))
    df = filter_valid_range(df_source.dropna(subset=[param_col]),param_col)
    avg = df.groupby(group_col)[param_col].mean()
    if avg.empty: return {},{},( None,None)
    vmin, vmax = float(avg.min()), float(avg.max())
    import matplotlib as mpl; import matplotlib.colors as mcolors
    norm = mcolors.Normalize(vmin=vmin,vmax=vmax) if vmax>vmin else mcolors.Normalize(vmin=vmin-1,vmax=vmax+1)
    cmap = mpl.colormaps["RdYlGn_r"]
    return ({n:mcolors.rgb2hex(cmap(norm(v))) for n,v in avg.items()},
            {n:f"{v:.2f}" for n,v in avg.items()}, (vmin,vmax))

def render_choropleth_legend(param_col, label, label_map, vmin, vmax, group_word):
    if param_col=="TEXTURE":
        if not label_map: return
        types_present = sorted(set(label_map.values()),
                               key=lambda t: [b[1] for b in TEXTURE_BANDS].index(t) if t in [b[1] for b in TEXTURE_BANDS] else 99)
        swatches = " &nbsp; ".join(
            f'<span style="display:inline-block;width:12px;height:12px;background:{TEXTURE_TYPE_COLORS.get(t,"#ccc")};'
            f'border-radius:2px;margin-right:4px;vertical-align:middle;"></span>{t}' for t in types_present)
        st.markdown(f"🎨 Dominant soil type per {group_word}: {swatches}", unsafe_allow_html=True)
    elif vmin is not None:
        st.caption(f"🎨 Colour scale (green→red): {vmin:.2f} → {vmax:.2f} average {label} per {group_word}")

def init_map_state():
    for k,v in {"map_level":"district","sel_district":None,"sel_subdist":None,
                "sel_village":None,"zoom_bounds":None}.items():
        if k not in st.session_state: st.session_state[k] = v

def set_current_scope(district=None, subdist=None):
    st.session_state["current_district"] = district
    st.session_state["current_subdist"]  = subdist

def reset_map_selection():
    for k,v in {"map_level":"district","sel_district":None,"sel_subdist":None,
                "sel_village":None,"zoom_bounds":None}.items():
        st.session_state[k] = v
    set_current_scope(None,None)

# ══════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════
st.set_page_config(page_title="Karnataka Soil Chatbot", page_icon="🌱", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif!important;color:#1a3a1a!important;}
.stApp{background-color:#f5f7f0!important;}
p,div,span,label,li,td,th,a{color:#1a3a1a!important;}
h1{color:#1a4d1a!important;font-weight:700!important;font-size:1.2rem!important;border-bottom:3px solid #4caf50;padding-bottom:10px;margin-bottom:20px!important;}
h2,h3{color:#1a4d1a!important;font-weight:700!important;}
section[data-testid="stSidebar"]{background-color:#2d5a2d!important;}
section[data-testid="stSidebar"] p,section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] span,section[data-testid="stSidebar"] label{color:#d4edda!important;}
.stSelectbox>div>div{border:1.5px solid #a5d6a7!important;border-radius:8px!important;background-color:#fff!important;}
.stTextInput>div>div>input{border:1.5px solid #a5d6a7!important;border-radius:8px!important;background-color:#fff!important;color:#1a3a1a!important;}
.stButton>button{background-color:#2e7d32!important;color:white!important;border:none!important;border-radius:8px!important;font-weight:600!important;}
.stButton>button:hover{background-color:#1b5e20!important;}
.stButton>button p{color:white!important;}
[data-testid="stChatMessage"]{border-radius:12px!important;padding:12px 16px!important;margin-bottom:8px!important;border:1px solid #c8e6c9!important;background-color:#fff!important;}
.stSuccess{background-color:#e8f5e9!important;border-left:4px solid #4caf50!important;border-radius:6px!important;}
[data-testid="stMetricLabel"]{font-size:11px!important;color:#2e6b2e!important;}
[data-testid="stMetricValue"]{font-size:18px!important;color:#1a3a1a!important;}
/* st_folium known bug: the component's outer iframe sometimes measures itself
   taller than the actual Leaflet map on first render, leaving blank space
   below it. Pin the iframe to the exact height we pass to st_folium(height=520)
   everywhere it's used, instead of letting the component auto-size. */
iframe[title="streamlit_folium.st_folium"] { height: 520px !important; }
</style>""", unsafe_allow_html=True)

st.markdown("<h1>🌱 Karnataka Soil Chatbot</h1>", unsafe_allow_html=True)

search_mode = st.radio("Search by", [
    "District / Sub-district / Village (dropdown)",
    "Map (click to select)",
    "Latitude & Longitude",
    "Compare Two Villages"
])

record = None; nearest_village = None; input_lat = input_lon = None; location_label = "Selected location"

# ── Mode 1: Dropdown ─────────────────────────────────────────
if search_mode == "District / Sub-district / Village (dropdown)":
    selected_district = st.selectbox("District", sorted(village_df["DISTRICT"].dropna().astype(str).unique()))
    sub_df = village_df[village_df["DISTRICT"].astype(str)==selected_district]
    selected_subdist = st.selectbox("Sub-district", sorted(sub_df["SUB_DIST"].dropna().astype(str).unique()))
    vill_df = sub_df[sub_df["SUB_DIST"].astype(str)==selected_subdist]
    selected_village = st.selectbox("Village", sorted(vill_df["KGISVill_2"].dropna().astype(str).unique()))
    record = vill_df[vill_df["KGISVill_2"].astype(str)==selected_village].iloc[0]
    location_label = str(record["KGISVill_2"])
    set_current_scope(selected_district, selected_subdist)

# ── Mode 2: Map drill-down ───────────────────────────────────
elif search_mode == "Map (click to select)":
    import folium
    from streamlit_folium import st_folium
    init_map_state()

    st.markdown("### 🗺️ Click the map: District → Sub-district → Village")
    b1,b2,b3,_ = st.columns([1,1,1,3])
    if b1.button("⟲ Reset", key="btn_reset_map"): reset_map_selection(); st.rerun()
    if st.session_state.sel_district and b2.button(f"◀ {st.session_state.sel_district}", key="btn_back_d"):
        st.session_state.update({"map_level":"district","sel_district":None,"sel_subdist":None,"sel_village":None,"zoom_bounds":None})
        set_current_scope(None,None); st.rerun()
    if st.session_state.sel_subdist and b3.button(f"◀ {st.session_state.sel_subdist}", key="btn_back_s"):
        st.session_state.update({"map_level":"subdistrict","sel_subdist":None,"sel_village":None})
        set_current_scope(st.session_state.sel_district,None); st.rerun()

    # Level 1 — Districts
    if st.session_state.map_level == "district":
        district_gdf = get_district_boundaries()
        cm_col1, cm_col2 = st.columns([3,2])
        color_metric = cm_col1.selectbox("Color districts by", CHOROPLETH_METRIC_OPTIONS, key="d_cm")
        border_only_d = cm_col2.checkbox("🖍️ Borders only (no fill)", key="d_border_only")
        m = folium.Map(location=[15.3,75.7], zoom_start=7, tiles="CartoDB positron")
        if color_metric in CHOROPLETH_METRIC_COL_MAP:
            pc = CHOROPLETH_METRIC_COL_MAP[color_metric]
            cm_map, lm, (vmin,vmax) = compute_choropleth(village_df,"DISTRICT",pc)
            add_boundary_layer(m, gdf_to_geojson(district_gdf), "DISTRICT", value_color_map=cm_map, value_labels=lm,
                               border_only=border_only_d)
            render_choropleth_legend(pc, color_metric.split()[0], lm, vmin, vmax, "district")
        else:
            add_boundary_layer(m, gdf_to_geojson(district_gdf), "DISTRICT", color_by_feature=True,
                               border_only=border_only_d)
        map_data = st_folium(m, height=520, width=None, key="district_map")
        clicked = map_data.get("last_active_drawing")
        if clicked:
            name = clicked["properties"]["DISTRICT"].strip()
            if name != st.session_state.sel_district:
                row = district_gdf[district_gdf["DISTRICT"]==name]
                if not row.empty:
                    st.session_state.update({"sel_district":name,"sel_subdist":None,"sel_village":None,
                                             "map_level":"subdistrict","zoom_bounds":feature_bounds_from_geom(row.iloc[0].geometry)})
                    set_current_scope(name,None); st.rerun()

    # Level 2 — Sub-districts
    elif st.session_state.map_level == "subdistrict":
        taluk_gdf = get_taluk_boundaries(st.session_state.sel_district)
        if taluk_gdf.empty:
            st.error(f"No boundary data for '{st.session_state.sel_district}'. Click Reset.")
        else:
            cm_col1, cm_col2 = st.columns([3,2])
            color_metric = cm_col1.selectbox("Color sub-districts by", CHOROPLETH_METRIC_OPTIONS, key="s_cm")
            border_only_s = cm_col2.checkbox("🖍️ Borders only (no fill)", key="s_border_only")
            m = folium.Map(tiles="CartoDB positron"); m.fit_bounds(st.session_state.zoom_bounds)
            dist_villages = village_df[village_df["DISTRICT"].astype(str)==st.session_state.sel_district]
            if color_metric in CHOROPLETH_METRIC_COL_MAP:
                pc = CHOROPLETH_METRIC_COL_MAP[color_metric]
                cm_map, lm, (vmin,vmax) = compute_choropleth(dist_villages,"SUB_DIST",pc)
                add_boundary_layer(m, gdf_to_geojson(taluk_gdf), "SUB_DIST", value_color_map=cm_map, value_labels=lm,
                                   border_only=border_only_s)
                render_choropleth_legend(pc, color_metric.split()[0], lm, vmin, vmax, "sub-district")
            else:
                add_boundary_layer(m, gdf_to_geojson(taluk_gdf), "SUB_DIST", fill_color="#4caf50", border_color="#1a4d1a",
                                   border_only=border_only_s)
            map_data = st_folium(m, height=520, width=None, key="subdist_map")
            clicked = map_data.get("last_active_drawing")
            if clicked:
                name = clicked["properties"]["SUB_DIST"].strip()
                if name != st.session_state.sel_subdist:
                    row = taluk_gdf[taluk_gdf["SUB_DIST"]==name]
                    if not row.empty:
                        st.session_state.update({"sel_subdist":name,"sel_village":None,"map_level":"village",
                                                 "zoom_bounds":feature_bounds_from_geom(row.iloc[0].geometry)})
                        set_current_scope(st.session_state.sel_district,name); st.rerun()

    # Level 3 — Villages (with satellite imagery)
    elif st.session_state.map_level == "village":
        village_boundaries_gdf = load_village_boundaries()
        vill_gdf = village_boundaries_gdf[
            (village_boundaries_gdf["DISTRICT"]==st.session_state.sel_district) &
            (village_boundaries_gdf["SUB_DIST"]==st.session_state.sel_subdist)
        ]
        if vill_gdf.empty:
            st.error(f"No village boundaries for '{st.session_state.sel_subdist}'. Click Reset.")
        else:
            cm_col1, cm_col2 = st.columns([3,2])
            color_metric = cm_col1.selectbox("Color villages by", CHOROPLETH_METRIC_OPTIONS, key="v_cm")
            border_only_v = cm_col2.checkbox("🖍️ Borders only (no fill)", key="v_border_only")

            # ── Satellite base map ──────────────────────────
            m = folium.Map(tiles=None)
            m.fit_bounds(st.session_state.zoom_bounds)

            satellite_tile = folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri World Imagery", name="🛰️ Satellite", overlay=False, control=True, show=True,
            ).add_to(m)
            google_tile = folium.TileLayer(
                tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
                attr="Google", name="🌍 Google Satellite", overlay=False, control=True, show=False,
            ).add_to(m)
            folium.TileLayer(
                tiles="CartoDB positron", name="🗺️ Street map", overlay=False, control=True, show=False,
            ).add_to(m)

            subdist_villages = village_df[
                (village_df["DISTRICT"].astype(str)==st.session_state.sel_district) &
                (village_df["SUB_DIST"].astype(str)==st.session_state.sel_subdist)
            ]
            if color_metric in CHOROPLETH_METRIC_COL_MAP:
                pc = CHOROPLETH_METRIC_COL_MAP[color_metric]
                cm_map, lm, (vmin,vmax) = compute_choropleth(subdist_villages,"KGISVill_2",pc)
                add_boundary_layer(m, gdf_to_geojson(vill_gdf), "KGISVill_2",
                                   value_color_map=cm_map, value_labels=lm,
                                   satellite_toggle_layers=[satellite_tile, google_tile],
                                   border_only=border_only_v)
                render_choropleth_legend(pc, color_metric.split()[0], lm, vmin, vmax, "village")
            else:
                add_boundary_layer(m, gdf_to_geojson(vill_gdf), "KGISVill_2",
                                   fill_color="#ff9800", border_color="#e65100",
                                   satellite_toggle_layers=[satellite_tile, google_tile],
                                   border_only=border_only_v)

            folium.LayerControl(position="topright", collapsed=False).add_to(m)

            map_data = st_folium(m, height=520, width=None, key="village_map")
            clicked = map_data.get("last_active_drawing")
            if clicked:
                name = clicked["properties"]["KGISVill_2"].strip()
                if name and name != st.session_state.sel_village:
                    st.session_state.sel_village = name; st.rerun()

    if st.session_state.sel_village:
        matches = village_df[
            (village_df["KGISVill_2"].astype(str).str.strip()==st.session_state.sel_village.strip()) &
            (village_df["SUB_DIST"].astype(str).str.strip()==(st.session_state.sel_subdist or "").strip()) &
            (village_df["DISTRICT"].astype(str).str.strip()==(st.session_state.sel_district or "").strip())
        ]
        if matches.empty:
            record = None; st.warning("Could not match village to soil data. Click Reset.")
        else:
            record = matches.iloc[0]; location_label = str(record["KGISVill_2"])
            st.success(f"📍 Selected: **{location_label}** ({st.session_state.sel_subdist}, {st.session_state.sel_district})")
    else:
        record = None
        st.info("Click a district, then a sub-district, then a village.")

# ── Mode 3: Lat/Lon ──────────────────────────────────────────
elif search_mode == "Latitude & Longitude":
    col_a,col_b = st.columns(2)
    input_lat = col_a.number_input("Latitude",  format="%.6f", value=15.0)
    input_lon = col_b.number_input("Longitude", format="%.6f", value=75.0)
    csv_record = idw_estimate(csv_tree, csv_df, input_lat, input_lon, columns=["SOC","DEPTH","TEXTURE","PH"], k=4)
    if csv_record is None:
        st.warning("No soil data available near this location (outside coverage area).")
    else:
        _, vill_idx = village_tree.query([[input_lat, input_lon]], k=1)
        nearest_village = village_df.iloc[vill_idx[0]]
        dist_km = haversine_km(input_lat, input_lon, float(nearest_village["latitude"]), float(nearest_village["longitude"]))
        st.markdown("---")
        st.success(f"📍 Nearest village: **{nearest_village['KGISVill_2']}** ({nearest_village['SUB_DIST']}, {nearest_village['DISTRICT']}) — {dist_km:.1f} km away")
        record = csv_record
        location_label = f"{input_lat:.4f}, {input_lon:.4f} (near {nearest_village['KGISVill_2']})"

# ── Mode 4: Compare ──────────────────────────────────────────
else:
    st.markdown("### ⚖️ Compare Two Villages")
    def village_picker(prefix, key):
        d = st.selectbox(f"{prefix} District", sorted(village_df["DISTRICT"].dropna().astype(str).unique()), key=f"{key}_d")
        sub = village_df[village_df["DISTRICT"].astype(str)==d]
        s = st.selectbox(f"{prefix} Sub-district", sorted(sub["SUB_DIST"].dropna().astype(str).unique()), key=f"{key}_s")
        vill = sub[sub["SUB_DIST"].astype(str)==s]
        v = st.selectbox(f"{prefix} Village", sorted(vill["KGISVill_2"].dropna().astype(str).unique()), key=f"{key}_v")
        return vill[vill["KGISVill_2"].astype(str)==v].iloc[0]
    col_x,col_y = st.columns(2)
    with col_x: rec_a = village_picker("Village A —", "cmp_a")
    with col_y: rec_b = village_picker("Village B —", "cmp_b")
    compare_df, score_a, score_b = compare_villages(rec_a, rec_b)
    st.dataframe(compare_df, hide_index=True)
    st.markdown("#### 🗺️ Locations")
    st.map(pd.DataFrame({"lat":[rec_a["latitude"],rec_b["latitude"]],"lon":[rec_a["longitude"],rec_b["longitude"]]}))
    if score_a != score_b:
        winner = rec_a['KGISVill_2'] if score_a>score_b else rec_b['KGISVill_2']
        st.success(f"🏆 **{winner}** has better overall soil fertility.")
    else:
        st.info("Both villages have equal fertility scores.")
    st.stop()

# ══════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════
st.markdown("---")
if record is None:
    st.info("Select a valid location, or ask a dataset-wide question in the chat below.")

if search_mode == "Latitude & Longitude" and record is not None:
    st.markdown("#### 📊 IDW-estimated soil data at entered coordinates")
    st.caption("Estimated from the 4 nearest known sample points, weighted by distance.")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Lat / Lon", f"{input_lat:.4f}, {input_lon:.4f}")
    c2.metric("SOC (%)",   f"{float(record['SOC']):.2f}")
    c3.metric("Depth (cm)",round(float(record['DEPTH'])))
    c4.metric("Texture",   texture_soil_type(record['TEXTURE']))
    c5.metric("pH",        f"{float(record['PH']):.2f}")
    st.markdown("#### 🏘️ Nearest village soil data (for reference)")
    v1,v2,v3,v4,v5,v6 = st.columns(6)
    v1.metric("Village",    nearest_village["KGISVill_2"])
    v2.metric("District",   nearest_village["DISTRICT"])
    v3.metric("SOC (%)",    f"{float(nearest_village['SOC']):.2f}")
    v4.metric("Depth (cm)", int(nearest_village["DEPTH"]))
    v5.metric("Texture",    texture_soil_type(nearest_village['TEXTURE']))
    v6.metric("pH",         f"{float(nearest_village['PH']):.2f}")
    st.map(pd.DataFrame({"lat":[input_lat,float(nearest_village["latitude"])],"lon":[input_lon,float(nearest_village["longitude"])]}))
elif record is not None:
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Village",    record["KGISVill_2"])
    c2.metric("District",   record["DISTRICT"])
    c3.metric("SOC (%)",    f"{float(record['SOC']):.2f}")
    c4.metric("Depth (cm)", int(record["DEPTH"]))
    c5.metric("Texture",    texture_soil_type(record['TEXTURE']))
    c6.metric("pH",         f"{float(record['PH']):.2f}")
    st.map(pd.DataFrame({"lat":[record["latitude"]],"lon":[record["longitude"]]}))

# ══════════════════════════════════════════════════════════════
# Weather
# ══════════════════════════════════════════════════════════════
if record is not None:
    st.markdown("---")
    st.markdown("#### 🌦️ Weather Forecast")
    weather_lat = input_lat if search_mode=="Latitude & Longitude" else float(record["latitude"])
    weather_lon = input_lon if search_mode=="Latitude & Longitude" else float(record["longitude"])
    wd = fetch_weather(weather_lat, weather_lon)
    if wd:
        cur = wd.get("current",{}); daily = wd.get("daily",{})
        w1,w2,w3 = st.columns(3)
        w1.metric("Temperature",  f"{cur.get('temperature_2m','N/A')} °C")
        w2.metric("Humidity",     f"{cur.get('relative_humidity_2m','N/A')}%")
        w3.metric("Current Rain", f"{cur.get('precipitation','N/A')} mm")
        precip_sum = daily.get("precipitation_sum",[])
        if precip_sum: st.info(f"🌧️ {rainfall_advice(precip_sum)}")
        dates = daily.get("time",[])
        if dates:
            st.dataframe(pd.DataFrame({"Date":dates,"Max Temp (°C)":daily.get("temperature_2m_max",[]),
                                       "Min Temp (°C)":daily.get("temperature_2m_min",[]),"Rain (mm)":precip_sum}),
                         hide_index=True)
    else:
        st.warning("Weather data unavailable right now.")

    st.markdown("---")
    if st.button("📄 Export PDF Report", key="btn_pdf"):
        pdf = generate_pdf_report(record, village_name=location_label)
        st.download_button("⬇️ Download Report", data=pdf,
                           file_name=f"soil_report_{location_label.split()[0].replace(',','')}.pdf",
                           mime="application/pdf")

# ══════════════════════════════════════════════════════════════
# Voice
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("🎤 Voice Input")
audio = mic_recorder(start_prompt="🎙️ Start Recording", stop_prompt="⏹️ Stop Recording",
                     just_once=True, use_container_width=True)
voice_query = None
if audio:
    try:
        st.audio(audio["bytes"])
        with open("voice.wav","wb") as f: f.write(audio["bytes"])
        transcription = groq_client.audio.transcriptions.create(file=open("voice.wav","rb"), model="whisper-large-v3")
        voice_query = transcription.text
        try: voice_query = GoogleTranslator(source="auto",target="en").translate(voice_query)
        except: pass
        st.success(f"You said: {voice_query}")
    except Exception as e:
        st.error(f"Voice error: {e}")

# ══════════════════════════════════════════════════════════════
# Chat
# ══════════════════════════════════════════════════════════════
text_query = st.text_input("💬 Ask about soil — or try 'highest SOC in Belagavi', 'pH between 6 and 7', 'SOC nearest to 6.2', 'compare X and Y'")
query = voice_query if voice_query else text_query

if query:
    st.chat_message("user").write(query)
    compare_names = detect_compare_query(query)
    ranking  = None if compare_names else detect_ranking_query(query)
    range_q  = None if (compare_names or ranking) else detect_range_query(query)
    nearest  = None if (compare_names or ranking or range_q) else detect_nearest_query(query)
    already_rendered = False

    if compare_names:
        name_a, name_b = compare_names
        rec_a = village_df[village_df["KGISVill_2"]==name_a].iloc[0]
        rec_b = village_df[village_df["KGISVill_2"]==name_b].iloc[0]
        cdf, sa, sb = compare_villages(rec_a, rec_b)
        with st.chat_message("assistant"):
            st.dataframe(cdf, hide_index=True)
            winner = name_a if sa>sb else (name_b if sb>sa else None)
            answer = (f"🏆 {winner} has better fertility ({max(sa,sb)}/4 vs {min(sa,sb)}/4)." if winner
                      else f"{name_a} and {name_b} have equal fertility ({sa}/4).")
            st.write(answer)
        already_rendered = True

    elif ranking:
        param_col, order, df_filter, sf = ranking
        result_df = rank_villages(param_col, order, df_filter, sf)
        if result_df.empty:
            answer = "No matching data found."
            st.chat_message("assistant").write(answer)
        else:
            answer = format_ranking_answer(param_col, order, result_df, df_filter, sf)
            st.chat_message("assistant").write(answer)
            cols_out = ["KGISVill_2","SUB_DIST","DISTRICT",param_col if param_col!="FERTILITY" else "FERTILITY"]
            if "FERTILITY" in result_df.columns or param_col!="FERTILITY":
                dl1,dl2 = st.columns(2)
                dl1.download_button("⬇️ CSV", data=result_to_csv(result_df,[c for c in cols_out if c in result_df.columns]),
                                    file_name="ranking.csv", mime="text/csv", key="r_csv")
                rows_out = [[r["KGISVill_2"],r["SUB_DIST"],r["DISTRICT"],
                             f"{float(r[param_col if param_col!='FERTILITY' else 'FERTILITY']):.2f}" if param_col not in ("TEXTURE","FERTILITY")
                             else (texture_soil_type(r["TEXTURE"]) if param_col=="TEXTURE"
                                   else ["Poor","Low","Moderate","Good","Excellent"][int(r["FERTILITY"])])]
                            for _,r in result_df.iterrows()]
                dl2.download_button("⬇️ PDF",
                                    data=generate_table_pdf(rows_out,["Village","Sub-district","District",PARAM_LABELS.get(param_col,param_col)],
                                                            "Karnataka Soil — Ranking",
                                                            f"{'Highest' if order=='high' else 'Lowest'} {PARAM_LABELS.get(param_col,param_col)} — {_scope_label(df_filter,sf)}"),
                                    file_name="ranking.pdf", mime="application/pdf", key="r_pdf")
        already_rendered = True

    elif range_q:
        param_col, lo, hi, df_filter, sf = range_q
        result_df = rank_villages_range(param_col, lo, hi, df_filter, sf)
        if result_df.empty:
            answer = "No matching data found for that range."
            st.chat_message("assistant").write(answer)
        else:
            answer = format_range_answer(param_col, lo, hi, result_df, df_filter, sf)
            st.chat_message("assistant").write(answer)
            dl1,dl2 = st.columns(2)
            dl1.download_button("⬇️ CSV", data=result_to_csv(result_df,["KGISVill_2","SUB_DIST","DISTRICT",param_col]),
                                file_name="range.csv", mime="text/csv", key="rng_csv")
            rows_out = [[r["KGISVill_2"],r["SUB_DIST"],r["DISTRICT"],
                         texture_soil_type(r["TEXTURE"]) if param_col=="TEXTURE" else f"{float(r[param_col]):.2f}"]
                        for _,r in result_df.iterrows()]
            dl2.download_button("⬇️ PDF",
                                data=generate_table_pdf(rows_out,["Village","Sub-district","District",PARAM_LABELS.get(param_col,param_col)],
                                                        "Karnataka Soil — Range Query",
                                                        f"{PARAM_LABELS.get(param_col,param_col)} between {lo:g} and {hi:g} — {_scope_label(df_filter,sf)}"),
                                file_name="range.pdf", mime="application/pdf", key="rng_pdf")
        already_rendered = True

    elif nearest:
        param_col, target, df_filter = nearest
        result_df = rank_villages_nearest(param_col, target, df_filter)
        if result_df.empty:
            answer = "No matching data found."
            st.chat_message("assistant").write(answer)
        else:
            answer = format_nearest_answer(param_col, target, result_df, df_filter)
            st.chat_message("assistant").write(answer)
            dl1,dl2 = st.columns(2)
            dl1.download_button("⬇️ CSV", data=result_to_csv(result_df,["KGISVill_2","SUB_DIST","DISTRICT",param_col,"_diff"]),
                                file_name="nearest.csv", mime="text/csv", key="n_csv")
            rows_out = [[r["KGISVill_2"],r["SUB_DIST"],r["DISTRICT"],
                         texture_soil_type(r["TEXTURE"]) if param_col=="TEXTURE" else f"{float(r[param_col]):.2f}",
                         f"{r['_diff']:.2f}"]
                        for _,r in result_df.iterrows()]
            dl2.download_button("⬇️ PDF",
                                data=generate_table_pdf(rows_out,["Village","Sub-district","District",PARAM_LABELS.get(param_col,param_col),"Δ from target"],
                                                        "Karnataka Soil — Nearest Value",
                                                        f"{PARAM_LABELS.get(param_col,param_col)} closest to {target:g} — {df_filter or 'Karnataka'}"),
                                file_name="nearest.pdf", mime="application/pdf", key="n_pdf")
        already_rendered = True

    elif record is not None:
        answer = keyword_response(query, record, location_name=location_label)
        if answer is None:
            context = (f"Soil expert for Karnataka.\nLocation: {location_label}\n"
                       f"SOC: {record.get('SOC','N/A')}%, Depth: {record.get('DEPTH','N/A')} cm, "
                       f"Texture: {texture_soil_type(record.get('TEXTURE',0))}, pH: {record.get('PH','N/A')}\n"
                       f"Question: {query}. Be concise.")
            try:
                res = groq_client.chat.completions.create(model="llama-3.1-8b-instant",
                      messages=[{"role":"user","content":context}], max_tokens=500)
                answer = res.choices[0].message.content
            except Exception as e:
                answer = f"AI unavailable: {e}"
    else:
        answer = "Select a location first, or ask a dataset-wide question like 'highest SOC in Belagavi', 'pH between 6 and 7', or 'SOC nearest to 6.2'."

    if not already_rendered:
        st.chat_message("assistant").write(answer)
    audio_file = speak_text(str(answer).replace("#","").replace("*",""))
    if audio_file: st.audio(audio_file)
