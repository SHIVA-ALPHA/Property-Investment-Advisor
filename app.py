#================================================================Modules Loading======================================================
import streamlit as st
import random
import io
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)

# =======================================================================Page Configuration===================================================

st.set_page_config(
    page_title="AI Property Investment Advisor",
    page_icon="🏡",
    layout="wide"
)

#================================================= Custom CSS=========================================

st.markdown("""
<style>

/* -------- Property-themed background (shown on every load/deploy) -------- */
.stApp{
    background:
        linear-gradient(rgba(255,255,255,0.88), rgba(255,255,255,0.90)),
        url("https://images.unsplash.com/photo-1560518883-ce09059eeffa?auto=format&fit=crop&w=1950&q=80");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Give the main content cards a soft translucent panel so text stays readable
   over the background image */
section.main > div.block-container{
    background: rgba(255,255,255,0.55);
    border-radius: 16px;
    padding: 1.5rem 2rem;
}

.main-title{
    text-align:center;
    color:#1E88E5;
    font-size:56px;
    font-weight:800;
    letter-spacing:1px;
    text-shadow: 1px 2px 6px rgba(0,0,0,0.15);
    margin-bottom:0;
}

.sub-title{
    text-align:center;
    color:#444;
    font-size:19px;
    margin-top:4px;
}

</style>
""", unsafe_allow_html=True)

#================================================== Heading================================================

st.markdown(
    "<p class='main-title'>🏡 AI Property Investment Advisor</p>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='sub-title'>Budget & City Based Investment Recommendation using LangChain + Gemini</p>",
    unsafe_allow_html=True
)

st.markdown("""
> **ℹ️ What you can tell the advisor:** your **budget**, a **primary city**, a **second
> city to compare it against** (pick it yourself or let the app auto-select one that
> refreshes every run), the **property type** you're eyeing, and your **investment goal**.
> An area-wise outlook for both cities appears immediately below — no need to click
> anything. Click **🚀 Analyze Investment** further down only when you want the full
> AI-generated recommendation and investment score.
""")

st.divider()

#================================================ Sidebar========================================================

with st.sidebar:
    st.header("🔑 Gemini API")
    api_key = st.text_input(
        "Enter Gemini API Key",
        type="password"
    )
    st.divider()
    st.markdown("### About")
    st.info("""
          This AI tool recommends
        property investments based on
            • Budget
            • Preferred City
            • Property Type
            • Investment Goal
        using Google Gemini.
    """)

#================================================ City & Area Data ========================================================
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


def extract_text(response):
    """
    Safely pull plain text out of a LangChain / Gemini response, regardless of
    whether .content comes back as a plain string or a list of content blocks.
    Never raises — falls back to a stringified response so the UI always shows
    *something* instead of silently failing.
    """
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                # common shapes: {"type": "text", "text": "..."} or {"text": "..."}
                text_val = block.get("text")
                if text_val:
                    parts.append(text_val)
            elif isinstance(block, str):
                parts.append(block)
        if parts:
            return "\n".join(parts)

    # Last-resort fallback so the user always sees a response
    return str(content) if content else "Sorry, I couldn't generate a response. Please try again."


def _df_to_table(df, col_widths=None):
    """Convert a pandas DataFrame (area report) into a styled reportlab Table."""
    data = [list(df.columns)] + df.values.tolist()
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E88E5")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7fa")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def build_pdf_report(ctx, score, recommendation_text, city, city2, area_df, area_df2):
    """Build the full investment report as a PDF and return it as bytes,
    ready for st.download_button."""
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

    story = []

    story.append(Paragraph("🏡 AI Property Investment Advisor", title_style))
    story.append(Paragraph("Investment Report", styles["Heading3"]))
    story.append(Paragraph(datetime.now().strftime("Generated on %d %b %Y, %I:%M %p"), body_style))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Your Inputs", heading_style))
    input_rows = [
        ["Budget", f"Rs. {ctx['budget']:,}"],
        ["Preferred City", ctx["city"]],
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

    story.append(Paragraph(f"Area-Wise Outlook — {city}", heading_style))
    if not area_df.empty:
        story.append(_df_to_table(area_df.drop(columns=["_color"]),
                                   col_widths=[4.5 * cm, 3.5 * cm, 3.5 * cm, 4 * cm]))
    else:
        story.append(Paragraph("No area-level data available.", body_style))

    story.append(Paragraph(f"Area-Wise Outlook — {city2}", heading_style))
    if not area_df2.empty:
        story.append(_df_to_table(area_df2.drop(columns=["_color"]),
                                   col_widths=[4.5 * cm, 3.5 * cm, 3.5 * cm, 4 * cm]))
    else:
        story.append(Paragraph("No area-level data available.", body_style))

    story.append(PageBreak())
    story.append(Paragraph("AI Recommendation", heading_style))
    for para in recommendation_text.split("\n"):
        if para.strip():
            story.append(Paragraph(para.strip(), body_style))
            story.append(Spacer(1, 6))

    story.append(Spacer(1, 20))
    story.append(Paragraph(
        "Note: Appreciation and rental yield figures are indicative, illustrative estimates "
        "for demonstration purposes, not live market data.",
        ParagraphStyle("Disclaimer", parent=styles["Normal"], fontSize=8, textColor=colors.grey)
    ))
    story.append(Paragraph(
        "Developed using Streamlit, LangChain and Google Gemini.",
        ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8, textColor=colors.grey)
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


#======================================================== User Inputs====================================================

col1, col2 = st.columns(2)
with col1:
    budget = st.number_input(
                            "💰 Budget (₹)",
                                        min_value=100000,
                                            value=5000000,
                                                step=100000
    )

    city = st.selectbox(
        "🏙 Preferred City",
        CITIES
    )
with col2:
    property_type = st.selectbox(
        "🏢 Property Type",
        [
            "Apartment",
            "Villa",
            "Commercial",
            "Plot"
        ])

    goal = st.selectbox(
        "🎯 Investment Goal",
        [
            "Rental Income",
            "Long Term Growth",
            "Resale",
            "Self Use"
        ])

#================================================ Comparison City Selection ========================================================

st.markdown("#### 🔀 Compare Against a Second City")
cmp_col1, cmp_col2 = st.columns([1, 2])

with cmp_col1:
    auto_pick = st.checkbox("🎲 Auto-select comparison city", value=True,
                             help="Picks a random other city; changes every time the app runs.")

other_cities = [c for c in CITIES if c != city]

with cmp_col2:
    if auto_pick:
        # Re-rolled on every script run (i.e. every time the app is opened/rerun),
        # not cached in session_state, so it changes each run as requested.
        city2 = random.choice(other_cities)
        st.info(f"Auto-selected comparison city: **{city2}**")
    else:
        city2 = st.selectbox("🏙 Comparison City", other_cities)

st.divider()
analyze = st.button(
    "🚀 Analyze Investment",
    use_container_width=True)

#======================================================== Area-Wise Report (always visible, two-city comparison) ==============================

st.subheader(f"📍 Area-Wise Outlook — {city} vs {city2}")
st.caption("Indicative estimates for demonstration purposes, not live market data. "
           "Shown automatically as soon as you pick your cities — no need to click Analyze.")


def render_city_report(target_city, container):
    """Render the area table + bar chart for one city inside the given column."""
    df = build_area_report(target_city)
    with container:
        st.markdown(f"**{target_city}**")
        if df.empty:
            st.warning("No area-level data available for this city yet.")
            return df
        st.dataframe(
            df.drop(columns=["_color"]),
            use_container_width=True,
            hide_index=True
        )
        bar_fig = px.bar(
            df,
            x="Area",
            y="Appreciation (%/yr)",
            color="Outlook",
            color_discrete_map={
                "High Profit Potential": "#2e7d32",
                "Moderate / Stable": "#f9a825",
                "Loss Risk / Weak Growth": "#c62828"
            },
            title=f"{target_city}: Area-Wise Appreciation"
        )
        bar_fig.update_layout(height=320, showlegend=False)
        st.plotly_chart(bar_fig, use_container_width=True)
    return df

rep_col1, rep_col2 = st.columns(2)
area_df = render_city_report(city, rep_col1)
area_df2 = render_city_report(city2, rep_col2)

if not area_df.empty and not area_df2.empty:
    avg_appr_1 = area_df["Appreciation (%/yr)"].mean()
    avg_yield_1 = area_df["Rental Yield (%)"].mean()
    avg_appr_2 = area_df2["Appreciation (%/yr)"].mean()
    avg_yield_2 = area_df2["Rental Yield (%)"].mean()

    st.markdown("#### ⚖️ City vs City — Average Outlook")
    cmp_fig = go.Figure(data=[
        go.Bar(name="Avg Appreciation (%/yr)", x=[city, city2], y=[avg_appr_1, avg_appr_2],
               marker_color="#1E88E5"),
        go.Bar(name="Avg Rental Yield (%)", x=[city, city2], y=[avg_yield_1, avg_yield_2],
               marker_color="#43A047"),
    ])
    cmp_fig.update_layout(barmode="group", height=350)
    st.plotly_chart(cmp_fig, use_container_width=True)

    leader = city if (avg_appr_1 * 1.5 + avg_yield_1 * 2) >= (avg_appr_2 * 1.5 + avg_yield_2 * 2) else city2
    st.success(f"🌟 Based on average area appreciation & rental yield, **{leader}** currently looks like the stronger pick between the two.")
else:
    st.warning("No area-level data available for one or both cities yet.")

st.divider()

# -------------------------------------------------
# Investment Score Logic
# -------------------------------------------------

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

#================================================== Prompt Template===================================

prompt = PromptTemplate.from_template("""
You are an experienced Property Investment Advisor.
A user wants to invest in real estate.
Details= Budget : ₹{budget} , Preferred City : {city}
      Property Type : {property_type} ,
      Investment Goal : {goal} ,
      Investment Score : {score}/100

Based on these details provide:
1. Explain the investment score.
2. Advantages of investing.
3. Possible risks.
4. Expected return.
5. Final recommendation.
Keep the answer professional and within 250 words.
""")

# ============================================Stop if never analyzed yet=========================================================

if analyze:
    if not api_key:
        st.warning("Please enter your Gemini API Key.")
        st.stop()

    #===================================== Gemini Model===========================================================

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=api_key,
        temperature=0.3
    )

    # =================================================LangChain Runnable Chain=================================
    chain = prompt | llm

    score = calculate_score(budget, city)

    with st.status("🤖 AI Agent is running...", expanded=True) as agent_status:
        st.write("📋 Preparing your investment profile...")
        st.write("🧠 Calling Gemini AI model — analyzing budget, city & goal...")
        response = chain.invoke({"budget": budget, "city": city,
                                        "property_type": property_type, "goal": goal, "score": score
        })
        st.write("📝 Formatting your recommendation...")
        agent_status.update(label="✅ Analysis complete", state="complete", expanded=False)

    # Persist results across reruns (e.g. when the chatbot triggers a rerun)
    st.session_state["analyzed"] = True
    st.session_state["last_score"] = score
    st.session_state["last_recommendation"] = extract_text(response)
    st.session_state["last_context"] = {
        "budget": budget,
        "city": city,
        "property_type": property_type,
        "goal": goal,
        "score": score
    }

if not st.session_state.get("analyzed", False):
    st.info("👆 Fill in your details above and click **Analyze Investment** to generate your report.")
    st.stop()

score = st.session_state["last_score"]
recommendation_text = st.session_state["last_recommendation"]

# ========================Display Investment Score==============================================

st.divider()
st.subheader("📊 Investment Score")
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=score,
    title={"text": "Investment Score"},
    gauge={"axis": {"range": [0, 100]},
        "bar": {"color": "green"},
        "steps": [
            {"range": [0, 40], "color": "#ffcccc"},
            {"range": [40, 70], "color": "#ffe699"},
            {"range": [70, 100], "color": "#b6fcb6"}
        ]}
))
fig.update_layout(height=350)
st.plotly_chart(fig, use_container_width=True)

# =================================================Score Message========================================================

if score >= 80:
    st.success("Excellent Investment Opportunity ⭐")
elif score >= 60:
    st.warning("Good Investment Opportunity")
else:
    st.error("High Risk Investment")

#========================================================User Summary=====================================================

ctx = st.session_state["last_context"]

col1, col2 = st.columns(2)
with col1:
  st.info(f"💰 Budget : ₹{ctx['budget']:,}")
  st.info(f"🏙 City : {ctx['city']}")
with col2:
  st.info(f"🏢 Property : {ctx['property_type']}")
  st.info(f"🎯 Goal : {ctx['goal']}")
  
#=========================================================== AI Recommendation=========================================================

st.divider()
st.subheader("🤖 AI Recommendation")
st.write(recommendation_text)

#=========================================================== PDF Report Download=========================================================

pdf_bytes = build_pdf_report(
    ctx=ctx,
    score=score,
    recommendation_text=recommendation_text,
    city=city,
    city2=city2,
    area_df=area_df,
    area_df2=area_df2,
)

st.download_button(
    label="📄 Download Report as PDF",
    data=pdf_bytes,
    file_name=f"Property_Investment_Report_{ctx['city']}_vs_{city2}.pdf",
    mime="application/pdf",
    use_container_width=True,
)

# Save context so the chatbot below can reference the latest analysis
st.session_state["last_context"] = {
    "budget": budget,
    "city": city,
    "property_type": property_type,
    "goal": goal,
    "score": score
}

#=================================================================== Chatbot ============================================================

st.divider()
st.subheader("💬 Ask the Property Investment Assistant")
st.caption("Ask follow-up questions about property investment, budgeting, cities, or this analysis. "
           "Off-topic questions will be politely declined.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, text)

CHATBOT_SYSTEM_PROMPT = PromptTemplate.from_template("""
You are a helpful, professional Real Estate & Property Investment Assistant embedded
inside an "AI Property Investment Advisor" app.

Your main focus is: real estate, property investment, budgeting for property purchase,
cities/areas for investment, rental yield, appreciation, home loans, property types, and
the user's current analysis context below. Answer these fully and helpfully.

If the user asks something outside this focus, still try to give a genuinely useful,
honest answer using your general knowledge. If you are unsure or don't know the answer,
simply say so plainly (e.g. "I'm not sure about that") instead of refusing to engage.
Do not lecture the user about staying on topic — just answer or say you don't know.

Current analysis context (may be empty if no analysis has been run yet):
{context}

Conversation so far:
{history}

User's new question: {question}

Respond concisely and professionally in a few sentences.
""")

if not api_key:
    st.info("Enter your Gemini API Key in the sidebar to chat with the assistant.")
else:
    chat_llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=api_key,
        temperature=0.4
    )
    chatbot_chain = CHATBOT_SYSTEM_PROMPT | chat_llm

    # Render existing chat history
    for role, text in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(text)

    user_question = st.chat_input("Ask about property investment, cities, budgets, risks...")

    if user_question:
        st.session_state.chat_history.append(("user", user_question))
        with st.chat_message("user"):
            st.write(user_question)

        context = st.session_state.get("last_context", {})
        context_str = (
            f"Budget: ₹{context.get('budget', 'N/A')}, City: {context.get('city', 'N/A')}, "
            f"Property Type: {context.get('property_type', 'N/A')}, Goal: {context.get('goal', 'N/A')}, "
            f"Investment Score: {context.get('score', 'N/A')}"
            if context else "No analysis run yet."
        )

        history_str = "\n".join(
            f"{role.capitalize()}: {text}" for role, text in st.session_state.chat_history[-6:]
        )

        with st.chat_message("assistant"):
            with st.spinner("🤖 Your AI Assistant is thinking..."):
                try:
                    bot_response = chatbot_chain.invoke({
                        "context": context_str,
                        "history": history_str,
                        "question": user_question
                    })
                    answer_text = extract_text(bot_response)
                    st.write(answer_text)
                    st.session_state.chat_history.append(("assistant", answer_text))
                except Exception as e:
                    st.error(f"Error: {e}")

#=================================================================== Footer============================================================
st.divider()
st.caption("Developed using Streamlit • LangChain • Google Gemini")
