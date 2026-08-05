
import streamlit as st
import plotly.graph_objects as go

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------

st.set_page_config(
    page_title="AI Property Investment Advisor",
    page_icon="🏡",
    layout="wide"
)

# -------------------------------------------------
# Custom CSS
# -------------------------------------------------

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

# -------------------------------------------------
# Heading
# -------------------------------------------------

st.markdown(
    "<p class='main-title'>🏡 AI Property Investment Advisor</p>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='sub-title'>Budget & City Based Investment Recommendation using LangChain + Gemini</p>",
    unsafe_allow_html=True
)

st.divider()

# -------------------------------------------------
# Sidebar
# -------------------------------------------------

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
        ["Delhi", "Mumbai",
            "Bangalore", "Hyderabad",
             "Pune", "Chennai",
              "Ahmedabad", "Kolkata"
        ])
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

# ============================================Stop if button not clicked=========================================================

if not analyze:
    st.stop()

if not api_key:
    st.warning("Please enter your Gemini API Key.")
    st.stop()

#===================================== Gemini Model===========================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=api_key,
    temperature=0.3
)

# -------------------------------------------------
# Investment Score Logic
# -------------------------------------------------

def calculate_score(budget, city):
    score = 50
    premium_cities = [
        "Delhi",
        "Mumbai",
        "Bangalore",
        "Hyderabad"]
    
    if city in premium_cities:
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

# ================================Calculate Score=====================================

score = calculate_score(budget, city)

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



# =================================================LangChain Runnable Chain=================================

chain = prompt | llm

#============================================= Generate AI Response=========================================

with st.spinner("Analyzing Property Investment..."):
  response = chain.invoke({"budget": budget, "city": city,
                                    "property_type": property_type, "goal": goal, "score": score
    })

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

col1, col2 = st.columns(2)
with col1:
  st.info(f"💰 Budget : ₹{budget:,}")
  st.info(f"🏙 City : {city}")
with col2:
  st.info(f"🏢 Property : {property_type}")
  st.info(f"🎯 Goal : {goal}")
  
#=========================================================== AI Recommendation=========================================================

st.divider()
st.subheader("🤖 AI Recommendation")
st.write(response.content)

#=================================================================== Footer============================================================
st.divider()
st.caption("Developed using Streamlit • LangChain • Google Gemini")
