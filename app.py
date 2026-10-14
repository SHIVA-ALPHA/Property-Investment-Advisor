#=============================================================== Modules Loading ===============================================================
import html
import io
import os
import random
import re
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
)

#=============================================================== Page Configuration ============================================================

st.set_page_config(
    page_title="AI Property Investment Advisor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_MODEL = "gemini-3.5-flash-lite"   # change in the sidebar if your key uses another model

#=============================================================== Custom CSS (blueprint theme) ===================================================
# Palette: ink-blue background with a faint drafting grid, brass accent, teal = good, coral = risk.

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=DM+Sans:wght@400;500;600&display=swap');

:root{
    --ink:#0b1626;
    --panel:#12233a;
    --panel-2:#183050;
    --line:rgba(160,190,230,.16);
    --text:#e9eef6;
    --muted:#9db0c8;
    --brass:#e0b04f;
    --teal:#3cc2b0;
    --amber:#f2b441;
    --coral:#ee6a5b;
}

/* ---- App background: ink blue with a drafting grid ---- */
.stApp{
    background-color:var(--ink);
    background-image:
        linear-gradient(var(--line) 1px, transparent 1px),
        linear-gradient(90deg, var(--line) 1px, transparent 1px);
    background-size:44px 44px;
    background-attachment:fixed;
    font-family:'DM Sans','Segoe UI',system-ui,sans-serif;
    color:var(--text);
}
[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:1.6rem; max-width:1280px;}

/* ---- Sidebar ---- */
[data-testid="stSidebar"]{
    background:#0a1220;
    border-right:1px solid var(--line);
}

/* ---- Hero ---- */
.hero{
    border:1px solid var(--line);
    border-left:6px solid var(--brass);
    background:linear-gradient(135deg, rgba(24,48,80,.92), rgba(11,22,38,.92));
    padding:30px 36px 26px 36px;
    border-radius:6px;
    margin-bottom:18px;
}
.hero h1{
    font-family:'Fraunces',Georgia,serif;
    font-weight:700;
    font-size:2.7rem;
    line-height:1.12;
    margin:0 0 10px 0;
    color:var(--text);
    letter-spacing:-.5px;
}
.hero p{margin:0; color:var(--muted); font-size:1.05rem; max-width:62ch; line-height:1.55;}
.hero .tags{margin-top:16px; display:flex; gap:8px; flex-wrap:wrap;}
.hero .tag{
    border:1px solid var(--line); border-radius:999px;
    padding:3px 12px; font-size:.82rem; color:var(--muted);
    background:rgba(255,255,255,.03);
}

/* ---- Section headings ---- */
h2, h3, h4{font-family:'Fraunces',Georgia,serif !important; color:var(--text) !important; letter-spacing:-.2px;}

/* ---- Bordered containers (cards) ---- */
[data-testid="stVerticalBlockBorderWrapper"]{
    background:rgba(18,35,58,.78);
    border-color:var(--line) !important;
    border-radius:8px !important;
}

/* ---- Metrics ---- */
[data-testid="stMetric"]{
    background:rgba(18,35,58,.85);
    border:1px solid var(--line);
    border-top:3px solid var(--brass);
    padding:12px 16px;
    border-radius:6px;
}
[data-testid="stMetricLabel"] p{color:var(--muted) !important; font-size:.85rem;}
[data-testid="stMetricValue"]{font-family:'Fraunces',Georgia,serif; font-size:1.55rem;}

/* ---- Tabs ---- */
button[data-baseweb="tab"]{font-weight:600; font-size:1rem; padding:10px 18px;}
button[data-baseweb="tab"][aria-selected="true"]{color:var(--brass) !important;}
div[data-baseweb="tab-highlight"]{background-color:var(--brass) !important;}

/* ---- Buttons ---- */
.stButton > button, .stDownloadButton > button{
    border-radius:6px; font-weight:600; border:1px solid var(--line);
}
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"]{
    background:linear-gradient(180deg,#e9bd5f,#c9972f) !important;
    color:#1a1204 !important;
    border:none !important;
    font-size:1.05rem;
    padding:.6rem 1rem;
}
.stButton > button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover{
    filter:brightness(1.07);
}

/* ---- Outlook legend / badges ---- */
.badge{display:inline-block; padding:2px 10px; border-radius:999px; font-size:.8rem; font-weight:600; margin-right:6px;}
.b-high{background:rgba(60,194,176,.16); color:#6fe0cf; border:1px solid rgba(60,194,176,.45);}
.b-mid {background:rgba(242,180,65,.14); color:#f5c866; border:1px solid rgba(242,180,65,.45);}
.b-low {background:rgba(238,106,91,.14); color:#f58b7e; border:1px solid rgba(238,106,91,.45);}

/* ---- Verdict banner ---- */
.verdict{
    border-radius:6px; padding:14px 18px; font-weight:600; font-size:1.05rem;
    border:1px solid var(--line); margin-top:4px;
}
.v-good{background:rgba(60,194,176,.12); border-left:5px solid var(--teal);}
.v-ok  {background:rgba(242,180,65,.12); border-left:5px solid var(--amber);}
.v-bad {background:rgba(238,106,91,.12); border-left:5px solid var(--coral);}

/* ---- Pick-of-two-cities line ---- */
.pick{
    border:1px dashed rgba(224,176,79,.55); border-radius:6px;
    padding:12px 16px; background:rgba(224,176,79,.07); margin-top:6px;
}

.small-note{color:var(--muted); font-size:.85rem;}
footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

#=============================================================== Helper utilities ==============================================================

def _st_version():
    nums = re.findall(r"\d+", st.__version__)[:2]
    return tuple(int(n) for n in nums)


NEW_STRETCH = _st_version() >= (1, 50)   # newer Streamlit uses width="stretch" instead of use_container_width


def stretch(fn, *args, **kwargs):
    """Make a Streamlit widget fill its container on both old and new Streamlit versions."""
    if NEW_STRETCH:
        return fn(*args, width="stretch", **kwargs)
    return fn(*args, use_container_width=True, **kwargs)


def inr(amount):
    """Format a rupee amount the Indian way (lakh / crore)."""
    amount = float(amount)
    if amount >= 1e7:
        return f"₹{amount / 1e7:.2f} Cr"
    if amount >= 1e5:
        return f"₹{amount / 1e5:.2f} L"
    return f"₹{amount:,.0f}"


def get_secret_key():
    """Look for an API key in Streamlit secrets or the environment so it need not be typed each time."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", "")


def style_fig(fig, height=340):
    """Apply the dark blueprint look to a Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, Segoe UI, sans-serif", color="#e9eef6"),
        margin=dict(l=20, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(gridcolor="rgba(160,190,230,.12)")
    fig.update_yaxes(gridcolor="rgba(160,190,230,.12)")
    return fig

#=============================================================== City & Area Data ==============================================================
# NOTE: Appreciation % and rental yield % below are indicative, illustrative
# sample estimates used to demonstrate the scoring/reporting logic for this
# project. They are NOT live market data and should be replaced with a real
# data source (e.g. a property-listing API) for production use.

PREMIUM_CITIES = [
    "Delhi", "Mumbai", "Bangalore", "Hyderabad",
    "Pune", "Chennai", "Gurugram", "Noida"
]

CITIES = [
    "Delhi", "Mumbai", "Bangalore", "Hyderabad",
    "Pune", "Chennai", "Ahmedabad", "Kolkata",
    "Jaipur", "Lucknow", "Noida", "Gurugram",
    "Indore", "Chandigarh", "Kochi", "Surat"
]

# city -> list of (area, appreciation_pct, rental_yield_pct)
AREA_DATA = {
    "Delhi": [
        ("Dwarka", 9.5, 2.8), ("Rohini", 8.0, 2.6),
        ("Lajpat Nagar", 6.5, 3.0), ("Najafgarh", 4.0, 2.0),
    ],
    "Mumbai": [
        ("Andheri West", 10.5, 3.2), ("Powai", 11.0, 3.5),
        ("Thane", 9.0, 3.0), ("Virar", 5.0, 2.2),
    ],
    "Bangalore": [
        ("Whitefield", 13.0, 3.8), ("Electronic City", 12.0, 3.6),
        ("Sarjapur Road", 11.5, 3.4), ("Yelahanka", 7.5, 2.8),
    ],
    "Hyderabad": [
        ("Gachibowli", 14.0, 4.0), ("Kondapur", 12.5, 3.7),
        ("HITEC City", 13.5, 3.9), ("Uppal", 6.0, 2.5),
    ],
    "Pune": [
        ("Hinjewadi", 12.0, 3.6), ("Baner", 11.0, 3.3),
        ("Wakad", 10.0, 3.1), ("Hadapsar", 7.0, 2.7),
    ],
    "Chennai": [
        ("OMR", 10.5, 3.4), ("Velachery", 9.0, 3.0),
        ("Porur", 8.0, 2.8), ("Ambattur", 5.5, 2.3),
    ],
    "Ahmedabad": [
        ("SG Highway", 8.5, 3.0), ("Bopal", 7.5, 2.8),
        ("Satellite", 7.0, 2.6), ("Naroda", 4.5, 2.0),
    ],
    "Kolkata": [
        ("Salt Lake", 6.5, 2.7), ("New Town", 8.0, 2.9),
        ("Rajarhat", 7.0, 2.6), ("Howrah", 3.5, 1.8),
    ],
    "Jaipur": [
        ("Vaishali Nagar", 6.0, 2.5), ("Mansarovar", 5.5, 2.4),
        ("Malviya Nagar", 5.0, 2.3), ("Sanganer", 3.0, 1.6),
    ],
    "Lucknow": [
        ("Gomti Nagar", 6.5, 2.6), ("Hazratganj", 5.0, 2.2),
        ("Indira Nagar", 4.5, 2.1), ("Alambagh", 3.0, 1.7),
    ],
    "Noida": [
        ("Sector 150", 11.0, 3.3), ("Sector 62", 9.5, 3.0),
        ("Greater Noida West", 8.0, 2.7), ("Sector 63", 7.0, 2.5),
    ],
    "Gurugram": [
        ("Sector 65 (Golf Course Ext.)", 13.5, 3.7), ("DLF Cyber City area", 12.0, 3.5),
        ("Sohna Road", 9.0, 2.9), ("Dwarka Expressway", 10.5, 3.1),
    ],
    "Indore": [
        ("Vijay Nagar", 6.0, 2.5), ("Rau", 4.5, 2.1),
        ("Bhawarkuan", 4.0, 2.0), ("Palasia", 5.5, 2.3),
    ],
    "Chandigarh": [
        ("Sector 20", 6.5, 2.6), ("Zirakpur", 7.5, 2.8),
        ("Mohali Phase 7", 8.0, 2.9), ("Panchkula", 5.5, 2.3),
    ],
    "Kochi": [
        ("Kakkanad", 7.5, 2.9), ("Edappally", 6.5, 2.6),
        ("Vyttila", 5.5, 2.4), ("Aluva", 4.5, 2.1),
    ],
    "Surat": [
        ("Vesu", 7.0, 2.8), ("Adajan", 6.0, 2.5),
        ("Piplod", 5.5, 2.4), ("Katargam", 3.5, 1.9),
    ],
}

OUTLOOK_COLORS = {
    "High Profit Potential": "#3cc2b0",
    "Moderate / Stable": "#f2b441",
    "Loss Risk / Weak Growth": "#ee6a5b",
}


def classify_area(appreciation_pct, rental_yield_pct):
    """Deterministic profit/loss classification from appreciation + yield."""
    combined = (appreciation_pct * 1.5) + (rental_yield_pct * 2)
    if combined >= 24:
        return "High Profit Potential", "#b6fcb6"
    elif combined >= 16:
        return "Moderate / Stable", "#ffe699"
    else:
        return "Loss Risk / Weak Growth", "#ffcccc"


def build_area_report(city):
    rows = []
    for area, appr, yld in AREA_DATA.get(city, []):
        label, color = classify_area(appr, yld)
        rows.append({
            "Area": area,
            "Appreciation (%/yr)": appr,
            "Rental Yield (%)": yld,
            "Outlook": label,
            "_color": color
        })
    return pd.DataFrame(rows)


def city_strength(df):
    """Weighted average outlook value for a city (same formula as the area classification)."""
    return df["Appreciation (%/yr)"].mean() * 1.5 + df["Rental Yield (%)"].mean() * 2


def calculate_score(budget, city):
    score = 50

    if city in PREMIUM_CITIES:
        score += 25
    else:
        score += 10
    if budget >= 10000000:
        score += 25
    elif budget >= 5000000:
        score += 15
    else:
        score += 5
    return min(score, 100)


def extract_text(response):
    """
    Safely pull plain text out of a LangChain / Gemini response, regardless of
    whether .content comes back as a plain string or a list of content blocks.
    """
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                text_val = block.get("text")
                if text_val:
                    parts.append(text_val)
            elif isinstance(block, str):
                parts.append(block)
        if parts:
            return "\n".join(parts)

    return str(content) if content else "Sorry, I couldn't generate a response. Please try again."

#=============================================================== PDF report ====================================================================

def _rl_clean(text):
    """Make LLM markdown safe for ReportLab: escape, convert bold/italic, drop unsupported glyphs."""
    text = text.replace("₹", "Rs. ")
    text = text.encode("cp1252", "ignore").decode("cp1252")   # Helvetica can't draw other symbols
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\*\w])\*(?!\s)(.+?)(?<!\s)\*(?![\*\w])", r"<i>\1</i>", text)
    return text


def _df_to_table(df, col_widths=None):
    """Convert the area report DataFrame (with _color column) into a styled ReportLab table."""
    colours = df["_color"].tolist()
    plain = df.drop(columns=["_color"])
    data = [list(plain.columns)] + plain.values.tolist()
    table = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E88E5")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    outlook_col = list(plain.columns).index("Outlook")
    for i, c in enumerate(colours, start=1):
        style.append(("BACKGROUND", (outlook_col, i), (outlook_col, i), colors.HexColor(c)))
    table.setStyle(TableStyle(style))
    return table


def build_pdf_report(ctx, recommendation_text):
    """Build the full investment report as PDF bytes, using the saved analysis context."""
    city, city2, score = ctx["city"], ctx["city2"], ctx["score"]
    area_df, area_df2 = build_area_report(city), build_area_report(city2)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], textColor=colors.HexColor("#1E88E5"), fontSize=22
    )
    heading_style = ParagraphStyle(
        "ReportHeading", parent=styles["Heading2"], textColor=colors.HexColor("#1E88E5"),
        spaceBefore=14, spaceAfter=6
    )
    body_style = ParagraphStyle("ReportBody", parent=styles["Normal"], fontSize=10, leading=15)
    bullet_style = ParagraphStyle("ReportBullet", parent=body_style, leftIndent=14, bulletIndent=4)

    story = [
        Paragraph("AI Property Investment Advisor", title_style),
        Paragraph("Investment Report", styles["Heading3"]),
        Paragraph(datetime.now().strftime("Generated on %d %b %Y, %I:%M %p"), body_style),
        Spacer(1, 12),
        Paragraph("Your Inputs", heading_style),
    ]

    input_rows = [
        ["Budget", f"Rs. {ctx['budget']:,}"],
        ["Preferred City", city],
        ["Comparison City", city2],
        ["Property Type", ctx["property_type"]],
        ["Investment Goal", ctx["goal"]],
        ["Investment Score", f"{score}/100"],
    ]
    input_table = Table(input_rows, colWidths=[5 * cm, 9 * cm])
    input_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef4fc")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(input_table)

    for name, df in ((city, area_df), (city2, area_df2)):
        story.append(Paragraph(f"Area-Wise Outlook - {name}", heading_style))
        if not df.empty:
            story.append(_df_to_table(df, col_widths=[4.5 * cm, 3.5 * cm, 3.5 * cm, 4 * cm]))
        else:
            story.append(Paragraph("No area-level data available.", body_style))

    story.append(PageBreak())
    story.append(Paragraph("AI Recommendation", heading_style))
    for line in recommendation_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            story.append(Paragraph(f"<b>{_rl_clean(line.lstrip('#').strip())}</b>", body_style))
        elif line.startswith(("* ", "- ")):
            story.append(Paragraph(_rl_clean(line[2:]), bullet_style, bulletText="\u2022"))
        else:
            story.append(Paragraph(_rl_clean(line), body_style))
        story.append(Spacer(1, 4))

    story.append(Spacer(1, 20))
    note_style = ParagraphStyle("Note", parent=styles["Normal"], fontSize=8, textColor=colors.grey)
    story.append(Paragraph(
        "Note: Appreciation and rental yield figures are indicative, illustrative estimates "
        "for demonstration purposes, not live market data. The recommendation is general "
        "guidance, not professional financial or legal advice.", note_style))
    story.append(Paragraph("Developed using Streamlit, LangChain and Google Gemini.", note_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

#=============================================================== Prompt templates ==============================================================

prompt = PromptTemplate.from_template("""
You are an experienced Property Investment Advisor.
A user wants to invest in real estate in India.
Details= Budget : Rs. {budget} , Preferred City : {city} , Comparison City : {city2}
      Property Type : {property_type} ,
      Investment Goal : {goal} ,
      Investment Score : {score}/100

Indicative area data (sample estimates, not live market data):
{area_summary}

Based on these details provide, using short markdown headings:
1. Explain the investment score.
2. Advantages of investing.
3. Possible risks.
4. Expected return.
5. Final recommendation.
Keep the answer professional and within 250 words.
""")

CHATBOT_SYSTEM_PROMPT = PromptTemplate.from_template("""
You are a helpful, professional Real Estate & Property Investment Assistant embedded
inside an "AI Property Investment Advisor" app.

Your main focus is: real estate, property investment, budgeting for property purchase,
cities/areas for investment, rental yield, appreciation, home loans, property types, and
the user's current analysis context below. Answer these fully and helpfully.

If the user asks something outside this focus, still try to give a genuinely useful,
honest answer using your general knowledge. If you are unsure or don't know the answer,
simply say so plainly (e.g. "I'm not sure about that") instead of refusing to engage.
Do not lecture the user about staying on topic - just answer or say you don't know.

Current analysis context (may be empty if no analysis has been run yet):
{context}

Conversation so far:
{history}

User's new question: {question}

Respond concisely and professionally in a few sentences.
""")


@st.cache_resource(show_spinner=False)
def get_llm(api_key, model, temperature):
    """One cached Gemini client per (key, model, temperature) so reruns stay fast."""
    return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=temperature)


def area_summary_text(city):
    df = build_area_report(city)
    if df.empty:
        return f"{city}: no data."
    lines = [f"- {r['Area']}: {r['Appreciation (%/yr)']}%/yr growth, {r['Rental Yield (%)']}% yield ({r['Outlook']})"
             for _, r in df.iterrows()]
    return f"{city}:\n" + "\n".join(lines)

#=============================================================== Session state =================================================================

def pick_other_city():
    """Random city different from the currently selected preferred city."""
    current = st.session_state.get("city_sel", CITIES[0])
    return random.choice([c for c in CITIES if c != current])


def shuffle_city():
    st.session_state["auto_city2"] = pick_other_city()


def queue_question(text):
    st.session_state["queued_q"] = text


def reset_all():
    for key in ("analysis", "chat_history", "queued_q"):
        st.session_state.pop(key, None)


st.session_state.setdefault("chat_history", [])   # list of (role, text)

#=============================================================== Sidebar =======================================================================

with st.sidebar:
    st.markdown("### Gemini access")
    typed_key = st.text_input("Gemini API key", type="password", placeholder="Paste your key here")
    api_key = typed_key or get_secret_key()
    model_name = st.text_input("Model", value=DEFAULT_MODEL,
                               help="Change this if your key does not have access to the default model.")

    if api_key:
        st.success("API key ready")
    else:
        st.warning("Add your API key to enable the AI recommendation and chat.")

    st.divider()
    st.markdown("### How it works")
    st.markdown(
        "1. Enter your budget, city, property type, and goal.\n"
        "2. See the area-wise outlook for two cities instantly.\n"
        "3. Press **Analyze investment** for your score and the AI recommendation.\n"
        "4. Download the PDF report or ask the assistant a follow-up."
    )
    st.divider()
    st.button("Start over", on_click=reset_all, help="Clears the analysis and the chat.")
    st.caption("Area figures are sample estimates for demonstration, not live market data.")

#=============================================================== Hero ==========================================================================

st.markdown("""
<div class="hero">
    <h1>Compare two cities.<br>Find the area worth your money.</h1>
    <p>Enter your budget and goal. The advisor ranks areas by growth and rental yield,
    scores your investment out of 100, and explains the result in plain language.</p>
    <div class="tags">
        <span class="tag">16 Indian cities</span>
        <span class="tag">64 areas</span>
        <span class="tag">LangChain + Gemini</span>
        <span class="tag">PDF report</span>
    </div>
</div>
""", unsafe_allow_html=True)

#=============================================================== Inputs ========================================================================

with st.container(border=True):
    st.markdown("#### Your investment details")
    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
    with c1:
        budget = st.number_input("Budget (₹)", min_value=100000, value=5000000, step=100000)
        st.caption(f"That is **{inr(budget)}**")
    with c2:
        city = st.selectbox("Preferred city", CITIES, key="city_sel")
    with c3:
        property_type = st.selectbox("Property type", ["Apartment", "Villa", "Commercial", "Plot"])
    with c4:
        goal = st.selectbox("Investment goal", ["Rental Income", "Long Term Growth", "Resale", "Self Use"])

    st.markdown("#### Compare against a second city")
    k1, k2, k3 = st.columns([1.2, 1.6, 1])
    other_cities = [c for c in CITIES if c != city]
    with k1:
        auto_pick = st.checkbox("Pick the second city for me", value=True)
    # keep the auto-picked city stable between reruns; re-roll only if it clashes with the preferred city
    if st.session_state.get("auto_city2") not in other_cities:
        st.session_state["auto_city2"] = pick_other_city()
    with k2:
        if auto_pick:
            city2 = st.session_state["auto_city2"]
            st.info(f"Comparing with **{city2}**")
        else:
            city2 = st.selectbox("Second city", other_cities)
    with k3:
        if auto_pick:
            st.button("Shuffle city", on_click=shuffle_city, help="Pick a different comparison city")

    analyze = stretch(st.button, "Analyze investment", type="primary")

#=============================================================== Tabs ==========================================================================

tab_area, tab_score, tab_chat = st.tabs(["Area outlook", "Score and recommendation", "Assistant"])

# ---------------------------------------------------------------- Tab 1: Area outlook ----------------------------------------------------------

with tab_area:
    st.subheader(f"{city} vs {city2}")
    st.markdown(
        '<span class="badge b-high">High Profit Potential</span>'
        '<span class="badge b-mid">Moderate / Stable</span>'
        '<span class="badge b-low">Loss Risk / Weak Growth</span>'
        '<span class="small-note">Indicative estimates, not live market data.</span>',
        unsafe_allow_html=True,
    )

    area_df = build_area_report(city)
    area_df2 = build_area_report(city2)

    def render_city_report(target_city, df, container):
        with container:
            with st.container(border=True):
                st.markdown(f"**{target_city}**")
                if df.empty:
                    st.warning("No area-level data available for this city yet.")
                    return
                stretch(
                    st.dataframe,
                    df.drop(columns=["_color"]),
                    hide_index=True,
                    column_config={
                        "Appreciation (%/yr)": st.column_config.ProgressColumn(
                            "Appreciation (%/yr)", min_value=0, max_value=15, format="%.1f"),
                        "Rental Yield (%)": st.column_config.ProgressColumn(
                            "Rental Yield (%)", min_value=0, max_value=5, format="%.1f"),
                    },
                )
                fig_bar = px.bar(
                    df, x="Area", y="Appreciation (%/yr)", color="Outlook",
                    color_discrete_map=OUTLOOK_COLORS,
                    title=f"{target_city}: area-wise appreciation",
                )
                fig_bar.update_layout(showlegend=False)
                stretch(st.plotly_chart, style_fig(fig_bar, 320))

    left, right = st.columns(2)
    render_city_report(city, area_df, left)
    render_city_report(city2, area_df2, right)

    if not area_df.empty and not area_df2.empty:
        st.markdown("#### City vs city")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"{city}: avg growth", f"{area_df['Appreciation (%/yr)'].mean():.1f}%/yr")
        m2.metric(f"{city}: avg yield", f"{area_df['Rental Yield (%)'].mean():.1f}%")
        m3.metric(f"{city2}: avg growth", f"{area_df2['Appreciation (%/yr)'].mean():.1f}%/yr")
        m4.metric(f"{city2}: avg yield", f"{area_df2['Rental Yield (%)'].mean():.1f}%")

        both = pd.concat([area_df.assign(City=city), area_df2.assign(City=city2)])
        ch1, ch2 = st.columns(2)
        with ch1:
            cmp_fig = go.Figure(data=[
                go.Bar(name="Avg appreciation (%/yr)", x=[city, city2],
                       y=[area_df["Appreciation (%/yr)"].mean(), area_df2["Appreciation (%/yr)"].mean()],
                       marker_color="#e0b04f"),
                go.Bar(name="Avg rental yield (%)", x=[city, city2],
                       y=[area_df["Rental Yield (%)"].mean(), area_df2["Rental Yield (%)"].mean()],
                       marker_color="#3cc2b0"),
            ])
            cmp_fig.update_layout(barmode="group", title="Average outlook")
            stretch(st.plotly_chart, style_fig(cmp_fig, 360))
        with ch2:
            scatter = px.scatter(
                both, x="Rental Yield (%)", y="Appreciation (%/yr)", color="Outlook", symbol="City",
                text="Area", color_discrete_map=OUTLOOK_COLORS, title="Every area: yield vs growth",
            )
            scatter.update_traces(marker=dict(size=13), textposition="top center", textfont=dict(size=10))
            scatter.update_layout(showlegend=False)
            stretch(st.plotly_chart, style_fig(scatter, 360))

        leader = city if city_strength(area_df) >= city_strength(area_df2) else city2
        st.markdown(
            f'<div class="pick">Based on average area appreciation and rental yield, '
            f'<b>{leader}</b> currently looks like the stronger pick of the two.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("No area-level data available for one or both cities yet.")

# ---------------------------------------------------------------- Run the analysis -------------------------------------------------------------

if analyze:
    if not api_key:
        st.toast("Add your Gemini API key in the sidebar first.", icon="🔑")
        st.session_state["need_key"] = True
    else:
        st.session_state["need_key"] = False
        score = calculate_score(budget, city)
        ctx_new = {
            "budget": budget, "city": city, "city2": city2,
            "property_type": property_type, "goal": goal, "score": score,
        }
        try:
            with st.status("Analyzing your investment...", expanded=True) as status:
                st.write("Preparing your investment profile")
                st.write("Asking Gemini to review your budget, cities, and goal")
                chain = prompt | get_llm(api_key, model_name, 0.3)
                response = chain.invoke({
                    "budget": f"{budget:,}", "city": city, "city2": city2,
                    "property_type": property_type, "goal": goal, "score": score,
                    "area_summary": area_summary_text(city) + "\n" + area_summary_text(city2),
                })
                st.write("Formatting your recommendation")
                status.update(label="Analysis complete", state="complete", expanded=False)
            st.session_state["analysis"] = {"ctx": ctx_new, "text": extract_text(response)}
            st.toast("Analysis ready. Open the Score and recommendation tab.", icon="✅")
        except Exception as e:
            st.session_state.pop("analysis", None)
            st.error(f"The AI request failed: {e}. Check your API key and model name in the sidebar.")

# ---------------------------------------------------------------- Tab 2: Score and recommendation ----------------------------------------------

with tab_score:
    result = st.session_state.get("analysis")
    if st.session_state.get("need_key") and not result:
        st.warning("Add your Gemini API key in the sidebar, then press Analyze investment.")
    elif not result:
        st.info("Fill in your details above and press **Analyze investment** to see your score and recommendation.")
    else:
        ctx = result["ctx"]
        recommendation_text = result["text"]
        score = ctx["score"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Budget", inr(ctx["budget"]))
        m2.metric("City", ctx["city"], help=f"Compared with {ctx['city2']}")
        m3.metric("Property", ctx["property_type"])
        m4.metric("Goal", ctx["goal"])

        g1, g2 = st.columns([1, 1.15])
        with g1:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                number={"font": {"size": 54, "family": "Fraunces, Georgia, serif"}},
                title={"text": "Investment score"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#e0b04f", "thickness": 0.28},
                    "steps": [
                        {"range": [0, 40], "color": "rgba(238,106,91,.35)"},
                        {"range": [40, 70], "color": "rgba(242,180,65,.35)"},
                        {"range": [70, 100], "color": "rgba(60,194,176,.35)"},
                    ],
                },
            ))
            stretch(st.plotly_chart, style_fig(gauge, 330))
            if score >= 80:
                st.markdown('<div class="verdict v-good">Excellent investment opportunity</div>', unsafe_allow_html=True)
            elif score >= 60:
                st.markdown('<div class="verdict v-ok">Good investment opportunity</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="verdict v-bad">High risk investment</div>', unsafe_allow_html=True)
        with g2:
            with st.container(border=True):
                st.markdown("#### AI recommendation")
                st.markdown(recommendation_text)

        try:
            pdf_bytes = build_pdf_report(ctx, recommendation_text)
            stretch(
                st.download_button,
                "Download report as PDF",
                data=pdf_bytes,
                file_name=f"Property_Investment_Report_{ctx['city']}_vs_{ctx['city2']}.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.error(f"Could not build the PDF report: {e}")

# ---------------------------------------------------------------- Tab 3: Assistant -------------------------------------------------------------

with tab_chat:
    st.subheader("Ask the property assistant")
    st.caption("Ask about cities, budgets, rental yield, loans, or your analysis.")

    if not api_key:
        st.info("Add your Gemini API key in the sidebar to chat with the assistant.")
    else:
        chat_llm = get_llm(api_key, model_name, 0.4)
        chatbot_chain = CHATBOT_SYSTEM_PROMPT | chat_llm

        if not st.session_state.chat_history:
            st.markdown("**Try one of these**")
            s1, s2, s3 = st.columns(3)
            s1.button("Is rental yield or growth better for me?", on_click=queue_question,
                      args=("Should I focus on rental yield or price growth for my goal?",))
            s2.button("What hidden costs should I plan for?", on_click=queue_question,
                      args=("What hidden costs should I budget for when buying property in India?",))
            s3.button("How much home loan can I afford?", on_click=queue_question,
                      args=("How do I work out a comfortable home loan size for my budget?",))

        for role, text in st.session_state.chat_history:
            with st.chat_message(role):
                st.write(text)

        user_question = st.chat_input("Ask about property investment, cities, budgets, risks...")
        user_question = user_question or st.session_state.pop("queued_q", None)

        if user_question:
            st.session_state.chat_history.append(("user", user_question))
            with st.chat_message("user"):
                st.write(user_question)

            saved = st.session_state.get("analysis")
            context = saved["ctx"] if saved else {}
            context_str = (
                f"Budget: Rs. {context.get('budget', 'N/A')}, City: {context.get('city', 'N/A')}, "
                f"Comparison City: {context.get('city2', 'N/A')}, "
                f"Property Type: {context.get('property_type', 'N/A')}, Goal: {context.get('goal', 'N/A')}, "
                f"Investment Score: {context.get('score', 'N/A')}"
                if context else "No analysis run yet."
            )
            history_str = "\n".join(
                f"{role.capitalize()}: {text}" for role, text in st.session_state.chat_history[-6:]
            )

            with st.chat_message("assistant"):
                with st.spinner("Your assistant is thinking..."):
                    try:
                        bot_response = chatbot_chain.invoke({
                            "context": context_str,
                            "history": history_str,
                            "question": user_question,
                        })
                        answer_text = extract_text(bot_response)
                        st.write(answer_text)
                        st.session_state.chat_history.append(("assistant", answer_text))
                    except Exception as e:
                        st.error(f"The assistant could not answer: {e}")

        if st.session_state.chat_history:
            st.button("Clear chat", on_click=lambda: st.session_state.update(chat_history=[]))

#=============================================================== Footer ========================================================================
st.divider()
st.caption("Developed using Streamlit • LangChain • Google Gemini. Suggestions are general guidance, not professional financial advice.")
