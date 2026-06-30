import asyncio
import sys
if sys.platform == "win32" and sys.version_info >= (3, 14):
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

import os
import requests
import streamlit as st
import pandas as pd
from scipy.spatial import cKDTree
from groq import Groq
from gtts import gTTS
from deep_translator import GoogleTranslator
from streamlit_mic_recorder import mic_recorder
import tempfile
import io
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm

# ── API key (from Streamlit secrets, never hardcoded) ──────────
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ── Paths ────────────────────────────────────────────────────
# Relative path: put combined_village_data.xlsx in the same repo folder as this script
BASE = os.path.dirname(os.path.abspath(__file__))

# Replace this with your actual Hugging Face dataset URL (use /resolve/main/, not /blob/main/)
CSV_URL = "https://huggingface.co/datasets/ricu9656/karnataka-soil-data/resolve/main/Export_Output.csv"
CSV_PATH = os.path.join(BASE, "Export_Output.csv")

# ── Data loading ──────────────────────────────────────────────
@st.cache_data
def load_village_data():
    df = pd.read_excel(os.path.join(BASE, "combined_village_data.xlsx"), engine="openpyxl")
    df = df[df['KGISVill_2'].notna() & (df['KGISVill_2'].str.strip() != '')]
    return df

@st.cache_data
def load_csv_data():
    # Download once, cache locally on the server's disk
    if not os.path.exists(CSV_PATH):
        with st.spinner("Downloading soil dataset (first run only)..."):
            r = requests.get(CSV_URL)
            r.raise_for_status()
            with open(CSV_PATH, "wb") as f:
                f.write(r.content)

    df = pd.read_csv(CSV_PATH)
    # Normalize CSV headers to match Excel column names
    df = df.rename(columns={
        "Depth":     "DEPTH",
        "pH":        "PH",
        "Texture":   "TEXTURE",
        "Longitude": "longitude",  # capital L -> lowercase
    })
    df = df[df['latitude'].notna() & df['longitude'].notna()]
    return df

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

# ── Interpretation helpers ─────────────────────────────────────
def soc_interp(v):
    v = float(v)
    if v < 1:   return f"{v:.2f} g/kg — Low. Needs organic amendments."
    elif v < 2: return f"{v:.2f} g/kg — Medium. Moderately fertile."
    else:        return f"{v:.2f} g/kg — High. Good fertility."

def depth_interp(v):
    v = int(v)
    if v < 25:   return f"{v} cm — Very shallow. Limited crop options."
    elif v < 50: return f"{v} cm — Shallow. Short-rooted crops only."
    elif v < 75: return f"{v} cm — Moderate. Suitable for most crops."
    else:         return f"{v} cm — Deep. Suitable for all crops."

def texture_interp(v):
    v = float(v)
    if v < 1.2:   return f"{v:.2f} — Fine/clay-like. Good water retention."
    elif v < 1.6: return f"{v:.2f} — Loam-like. Ideal for most crops."
    else:          return f"{v:.2f} — Coarse/sandy. Good drainage, low retention."

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
        if float(record['SOC']) >= 2: score += 1
    except: pass
    try:
        if int(record['DEPTH']) >= 75: score += 1
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

# ── Keyword response ───────────────────────────────────────────
def keyword_response(query, record):
    q = query.lower()
    name = record.get('KGISVill_2', 'Selected location')

    if any(w in q for w in ["soc", "organic carbon", "carbon"]):
        return f"**SOC — {name}:** {soc_interp(record['SOC'])}"
    if "depth" in q:
        return f"**Depth — {name}:** {depth_interp(record['DEPTH'])}"
    if "texture" in q:
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
    label = village_name or str(record.get('KGISVill_2', 'Custom location'))
    story.append(Paragraph(f"Location: {label}", heading_style))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Soil Parameters", heading_style))
    soil_data = [
        ["Parameter", "Value", "Interpretation"],
        ["SOC (g/kg)", f"{float(record['SOC']):.2f}", soc_interp(record['SOC'])],
        ["Depth (cm)", str(int(record['DEPTH'])), depth_interp(record['DEPTH'])],
        ["Texture",    f"{float(record['TEXTURE']):.2f}", texture_interp(record['TEXTURE'])],
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

def speak_text(text):
    try:
        tts = gTTS(text=str(text))
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            return fp.name
    except Exception:
        return None

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

search_mode = st.radio("Search by", ["District / Sub-district / Village", "Latitude & Longitude"])

record        = None
csv_record    = None
nearest_village = None
input_lat = input_lon = None

# ── Mode 1: Village picker ─────────────────────────────────────
if search_mode == "District / Sub-district / Village":
    selected_district = st.selectbox("District", sorted(village_df["DISTRICT"].dropna().astype(str).unique()))
    sub_df = village_df[village_df["DISTRICT"].astype(str) == selected_district]
    selected_subdist = st.selectbox("Sub-district", sorted(sub_df["SUB_DIST"].dropna().astype(str).unique()))
    vill_df = sub_df[sub_df["SUB_DIST"].astype(str) == selected_subdist]
    selected_village = st.selectbox("Village", sorted(vill_df["KGISVill_2"].dropna().astype(str).unique()))
    record = vill_df[vill_df["KGISVill_2"].astype(str) == selected_village].iloc[0]

# ── Mode 2: Lat/Lon — pull from CSV + show nearest village ────
else:
    col_a, col_b = st.columns(2)
    input_lat = col_a.number_input("Latitude",  format="%.6f", value=15.0)
    input_lon = col_b.number_input("Longitude", format="%.6f", value=75.0)

    # Nearest point in CSV (actual soil data at that coordinate)
    _, csv_idx = csv_tree.query([[input_lat, input_lon]], k=1)
    csv_record = csv_df.iloc[csv_idx[0]]

    # Nearest named village from village dataset
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

    # Use CSV record as the primary data source
    record = csv_record

# ══════════════════════════════════════════════════════════════
# Metrics display
# ══════════════════════════════════════════════════════════════
st.markdown("---")

if search_mode == "Latitude & Longitude" and csv_record is not None:
    st.markdown("#### 📊 Soil data at entered coordinates (from CSV)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Lat / Lon", f"{input_lat:.4f}, {input_lon:.4f}")
    c2.metric("SOC (g/kg)", f"{float(csv_record['SOC']):.2f}" if 'SOC' in csv_record else "N/A")
    c3.metric("Depth (cm)", int(csv_record['DEPTH']) if 'DEPTH' in csv_record else "N/A")
    c4.metric("Texture",    f"{float(csv_record['TEXTURE']):.2f}" if 'TEXTURE' in csv_record else "N/A")
    c5.metric("pH",         f"{float(csv_record['PH']):.2f}" if 'PH' in csv_record else "N/A")

    st.markdown("#### 🏘️ Nearest village soil data")
    v1, v2, v3, v4, v5, v6 = st.columns(6)
    v1.metric("Village",    nearest_village["KGISVill_2"])
    v2.metric("District",   nearest_village["DISTRICT"])
    v3.metric("SOC (g/kg)", f"{float(nearest_village['SOC']):.2f}")
    v4.metric("Depth (cm)", int(nearest_village["DEPTH"]))
    v5.metric("Texture",    f"{float(nearest_village['TEXTURE']):.2f}")
    v6.metric("pH",         f"{float(nearest_village['PH']):.2f}")

    # Map showing both points
    map_df = pd.DataFrame({
        "lat": [input_lat, float(nearest_village["latitude"])],
        "lon": [input_lon, float(nearest_village["longitude"])],
    })
    st.map(map_df)

else:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Village",    record["KGISVill_2"])
    c2.metric("District",   record["DISTRICT"])
    c3.metric("SOC (g/kg)", f"{float(record['SOC']):.2f}")
    c4.metric("Depth (cm)", int(record["DEPTH"]))
    c5.metric("Texture",    f"{float(record['TEXTURE']):.2f}")
    c6.metric("pH",         f"{float(record['PH']):.2f}")
    st.map(pd.DataFrame({"lat": [record["latitude"]], "lon": [record["longitude"]]}))

# ══════════════════════════════════════════════════════════════
# PDF export
# ══════════════════════════════════════════════════════════════
st.markdown("---")
if st.button("📄 Export PDF Report"):
    village_label = (
        f"{nearest_village['KGISVill_2']} (nearest to {input_lat:.4f}, {input_lon:.4f})"
        if search_mode == "Latitude & Longitude"
        else str(record["KGISVill_2"])
    )
    pdf_buffer = generate_pdf_report(record, village_name=village_label)
    st.download_button(
        label="⬇️ Download Report",
        data=pdf_buffer,
        file_name=f"soil_report_{village_label.split()[0]}.pdf",
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
    answer = keyword_response(query, record)

    if answer is None:
        village_name = (
            nearest_village["KGISVill_2"]
            if search_mode == "Latitude & Longitude"
            else record.get("KGISVill_2", "selected location")
        )
        context = f"""Soil expert for Karnataka. Data:
Location: {village_name}
SOC: {record.get('SOC', 'N/A')} g/kg, Depth: {record.get('DEPTH', 'N/A')} cm, Texture: {record.get('TEXTURE', 'N/A')}, pH: {record.get('PH', 'N/A')}
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