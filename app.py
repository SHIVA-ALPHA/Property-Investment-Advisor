#================================================================Modules Loading======================================================
import streamlit as st

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# =======================================================================Page Configuration===================================================

st.set_page_config(
    page_title="AI Property Investment Advisor",
    page_icon="🏡",
    layout="wide"
)

#================================================= Custom CSS=========================================

st.markdown("""
<style>
.main-title{
    text-align:center;
    color:#1E88E5;
    font-size:42px;
    font-weight:bold;
}

.sub-title{
    text-align:center;
    color:gray;
    font-size:18px;
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

st.divider()
analyze = st.button(
    "🚀 Analyze Investment",
    use_container_width=True)

#======================================================== Area-Wise Report (always visible) ==============================

st.subheader(f"📍 Area-Wise Outlook — {city}")
st.caption("Indicative estimates for demonstration purposes, not live market data.")

area_df = build_area_report(city)

if not area_df.empty:
    rep_col1, rep_col2 = st.columns([1, 1])

    with rep_col1:
        st.dataframe(
            area_df.drop(columns=["_color"]),
            use_container_width=True,
            hide_index=True
        )

    with rep_col2:
        bar_fig = px.bar(
            area_df,
            x="Area",
            y="Appreciation (%/yr)",
            color="Outlook",
            color_discrete_map={
                "High Profit Potential": "#2e7d32",
                "Moderate / Stable": "#f9a825",
                "Loss Risk / Weak Growth": "#c62828"
            },
            title=f"{city}: Area-Wise Appreciation Outlook"
        )
        bar_fig.update_layout(height=350)
        st.plotly_chart(bar_fig, use_container_width=True)

    best_area = area_df.sort_values("Appreciation (%/yr)", ascending=False).iloc[0]
    worst_area = area_df.sort_values("Appreciation (%/yr)", ascending=True).iloc[0]
    m1, m2 = st.columns(2)
    m1.success(f"🌟 Most Profitable Area: **{best_area['Area']}** ({best_area['Appreciation (%/yr)']}%/yr appreciation)")
    m2.error(f"⚠️ Weakest Area: **{worst_area['Area']}** ({worst_area['Appreciation (%/yr)']}%/yr appreciation)")
else:
    st.warning("No area-level data available for this city yet.")

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

    with st.spinner("Analyzing Property Investment..."):
        response = chain.invoke({"budget": budget, "city": city,
                                        "property_type": property_type, "goal": goal, "score": score
        })

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
            with st.spinner("Thinking..."):
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
