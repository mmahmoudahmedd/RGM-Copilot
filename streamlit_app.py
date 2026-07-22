import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="RGM Copilot", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------------
# Data loading (cached so it only runs once per session)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/rgm_copilot_dataset.csv", parse_dates=["date"])
    forecast_lookup = pd.read_csv("models/forecast_lookup.csv", parse_dates=["date"])
    risk_lookup = pd.read_csv("models/stockout_risk_lookup.csv", parse_dates=["date"])
    return df, forecast_lookup, risk_lookup

df, forecast_lookup, risk_lookup = load_data()

# ---------------------------------------------------------------------------
# RAG knowledge base (same grounded notes as notebook 4)
# ---------------------------------------------------------------------------
@st.cache_resource
def build_rag():
    notes = []

    stockout_by_channel = df.groupby("channel")["stockout_flag"].mean().sort_values(ascending=False)
    worst_channel = stockout_by_channel.index[0]
    notes.append(
        f"{worst_channel} has the highest stockout rate of any channel, at {stockout_by_channel.iloc[0]:.1%}. "
        f"This channel tends to have spikier, less predictable demand and thinner inventory buffers compared to "
        f"{stockout_by_channel.index[-1]}, which has the lowest stockout rate at {stockout_by_channel.iloc[-1]:.1%}."
    )

    sku_info = df[["sku", "lead_time_days"]].drop_duplicates().sort_values("lead_time_days", ascending=False)
    longest_lead_sku = sku_info.iloc[0]
    notes.append(
        f"{longest_lead_sku['sku']} has the longest replenishment lead time in the portfolio at "
        f"{longest_lead_sku['lead_time_days']:.0f} days, making it more vulnerable to stockouts when demand spikes "
        f"unexpectedly, since there is less time to react with a reorder."
    )

    promo_effect = df.groupby("on_promo")["units_sold"].mean()
    uplift = promo_effect[True] / promo_effect[False] - 1
    notes.append(
        f"Promotional activity drives an average {uplift:.0%} uplift in units sold across the portfolio. "
        f"This uplift needs to be factored into demand forecasts and inventory planning ahead of promo periods, "
        f"otherwise stockout risk increases sharply during promotions."
    )

    top_risk = (risk_lookup.groupby(["region", "channel", "sku"])["stockout_risk_score"]
                .mean().sort_values(ascending=False).head(5))
    for (region, channel, sku), score in top_risk.items():
        notes.append(
            f"{sku} in {region} through the {channel} channel is among the highest stockout-risk combinations "
            f"in the model's predictions, with an average predicted risk of {score:.1%}. "
            f"This combination should be prioritized for inventory review."
        )

    extreme = risk_lookup[risk_lookup["stockout_risk_score"] > 0.9]
    extreme_counts = extreme.groupby(["region", "channel", "sku"]).size().sort_values(ascending=False)
    if len(extreme_counts) > 0:
        (top_region, top_channel, top_sku), top_count = extreme_counts.index[0], extreme_counts.iloc[0]
        notes.append(
            f"{top_sku} in {top_region} ({top_channel}) recorded {int(top_count)} days with extreme stockout risk "
            f"(predicted risk above 90%) during the test period — more than any other combination. "
            f"This pattern repeats across multiple {top_channel} locations for {top_sku}, suggesting a "
            f"structural issue (likely its longer lead time combined with high seasonal demand) rather than "
            f"a one-off event."
        )

    region_rev = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
    notes.append(
        f"{region_rev.index[0]} is the highest-revenue region in the portfolio, generating "
        f"{region_rev.iloc[0]:,.0f} in total revenue over the dataset period, "
        f"reflecting its larger market size relative to other regions."
    )

    notes.append(
        "Demand across the portfolio follows a clear summer seasonality pattern, with cold beverages "
        "(colas, sodas, sparkling water, iced tea) seeing the strongest uplift during peak summer months, "
        "increasing both sales volume and stockout risk if inventory isn't scaled up ahead of the season."
    )

    vectorizer = TfidfVectorizer(stop_words="english")
    note_vectors = vectorizer.fit_transform(notes)
    return notes, vectorizer, note_vectors

notes, vectorizer, note_vectors = build_rag()

def retrieve_context(query: str, k: int = 2) -> list:
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, note_vectors)[0]
    top_idx = sims.argsort()[::-1][:k]
    return [notes[i] for i in top_idx if sims[i] > 0]

# ---------------------------------------------------------------------------
# Function-calling tools
# ---------------------------------------------------------------------------
def get_demand_forecast(region: str, channel: str, sku: str) -> str:
    """Get the demand forecast for a specific region, channel, and SKU combination."""
    sub = forecast_lookup[
        (forecast_lookup["region"].str.lower() == region.lower()) &
        (forecast_lookup["channel"].str.lower() == channel.lower()) &
        (forecast_lookup["sku"].str.lower() == sku.lower())
    ]
    if sub.empty:
        return f"No forecast data found for {sku} in {region}, {channel}."
    latest = sub.sort_values("date").iloc[-1]
    avg_recent = sub.sort_values("date").tail(14)["forecast"].mean()
    return (
        f"Latest forecast for {sku} in {region} ({channel}) on {latest['date'].date()}: "
        f"{latest['forecast']:.0f} units (actual was {latest['actual']:.0f}). "
        f"14-day average forecasted demand: {avg_recent:.0f} units/day."
    )

def get_stockout_risk(region: str, channel: str, sku: str) -> str:
    """Get the stockout risk assessment for a specific region, channel, and SKU combination."""
    sub = risk_lookup[
        (risk_lookup["region"].str.lower() == region.lower()) &
        (risk_lookup["channel"].str.lower() == channel.lower()) &
        (risk_lookup["sku"].str.lower() == sku.lower())
    ]
    if sub.empty:
        return f"No stockout risk data found for {sku} in {region}, {channel}."
    avg_risk = sub["stockout_risk_score"].mean()
    max_risk = sub["stockout_risk_score"].max()
    actual_stockouts = sub["stockout_actual"].sum()
    return (
        f"Stockout risk for {sku} in {region} ({channel}): "
        f"average risk score {avg_risk:.1%}, peak risk {max_risk:.1%}. "
        f"{int(actual_stockouts)} actual stockout days observed in the test period."
    )

def get_top_risk_items(n: int = 5) -> str:
    """Get the top N region/channel/SKU combinations with the highest average stockout risk."""
    top = (risk_lookup.groupby(["region", "channel", "sku"])["stockout_risk_score"]
           .mean().sort_values(ascending=False).head(n))
    lines = [f"{i+1}. {sku} - {region} - {channel}: {score:.1%} avg risk"
             for i, ((region, channel, sku), score) in enumerate(top.items())]
    return "Top stockout-risk items:\n" + "\n".join(lines)

SYSTEM_INSTRUCTION = """You are RGM Copilot, an analytics assistant for an FMCG Revenue Growth
Management team. You help stakeholders understand demand forecasts and stockout risk.

Rules:
- For questions asking for specific numbers (forecasts, risk scores, top-risk items), use the
  available functions to get real data. Never make up numbers.
- For "why" questions, use the provided context notes, which are grounded in real analysis.
- Be concise and business-focused. Avoid jargon where possible.
- If you don't have enough information to answer confidently, say so rather than guessing."""

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("📊 RGM Copilot")
st.caption(
    "An FMCG Revenue Growth Management analytics assistant — demand forecasting, "
    "stockout-risk classification, and a GenAI copilot grounded in real model output."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Regions", df["region"].nunique())
c2.metric("Channels", df["channel"].nunique())
c3.metric("SKUs", df["sku"].nunique())
c4.metric("Days of data", df["date"].nunique())

tab1, tab2 = st.tabs(["📈 Dashboard", "💬 Chat"])

# --- Dashboard tab ---
with tab1:
    st.subheader("Explore forecast & stockout risk")

    col1, col2, col3 = st.columns(3)
    region = col1.selectbox("Region", sorted(df["region"].unique()))
    channel = col2.selectbox("Channel", sorted(df["channel"].unique()))
    sku = col3.selectbox("SKU", sorted(df["sku"].unique()))

    sub = forecast_lookup[
        (forecast_lookup["region"] == region) &
        (forecast_lookup["channel"] == channel) &
        (forecast_lookup["sku"] == sku)
    ].sort_values("date")

    if not sub.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sub["date"], y=sub["actual"], name="Actual", line=dict(width=2)))
        fig.add_trace(go.Scatter(x=sub["date"], y=sub["forecast"], name="Forecast", line=dict(width=2, dash="dot")))
        fig.update_layout(
            title=f"Actual vs Forecast — {sku} in {region} ({channel})",
            xaxis_title="Date", yaxis_title="Units sold",
            height=400, margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    risk_sub = risk_lookup[
        (risk_lookup["region"] == region) &
        (risk_lookup["channel"] == channel) &
        (risk_lookup["sku"] == sku)
    ]
    if not risk_sub.empty:
        avg_risk = risk_sub["stockout_risk_score"].mean()
        risk_label = "🟢 Low" if avg_risk < 0.1 else ("🟡 Medium" if avg_risk < 0.3 else "🔴 High")
        st.metric("Average stockout risk for this combination", f"{avg_risk:.1%}", risk_label)

    st.subheader("🔥 Top 5 highest stockout-risk items")
    top5 = (risk_lookup.groupby(["region", "channel", "sku"])["stockout_risk_score"]
            .mean().sort_values(ascending=False).head(5).reset_index())
    top5.columns = ["Region", "Channel", "SKU", "Avg. Risk"]
    top5["Avg. Risk"] = top5["Avg. Risk"].apply(lambda x: f"{x:.1%}")
    st.dataframe(top5, hide_index=True, use_container_width=True)

# --- Chat tab ---
with tab2:
    st.subheader("Ask RGM Copilot")

    api_key = st.secrets.get("GEMINI_API_KEY", None)
    if not api_key:
        api_key = st.text_input("Enter your Gemini API key to use the chat:", type="password")

    if api_key:
        from google import genai
        from google.genai import types

        if "gemini_client" not in st.session_state:
            st.session_state.gemini_client = genai.Client(api_key=api_key)
        if "chat_session" not in st.session_state:
            st.session_state.chat_session = st.session_state.gemini_client.chats.create(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    tools=[get_demand_forecast, get_stockout_risk, get_top_risk_items],
                ),
            )
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for role, text in st.session_state.chat_history:
            with st.chat_message(role):
                st.write(text)

        question = st.chat_input("Ask about demand forecasts or stockout risk...")
        if question:
            st.session_state.chat_history.append(("user", question))
            with st.chat_message("user"):
                st.write(question)

            context_notes = retrieve_context(question, k=2)
            context_block = "\n".join(f"- {n}" for n in context_notes) if context_notes else "(no relevant context found)"
            augmented_input = f"""Relevant business context (from prior analysis):
{context_block}

Question: {question}"""

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.chat_session.send_message(augmented_input)
                    st.write(response.text)
            st.session_state.chat_history.append(("assistant", response.text))
    else:
        st.info("Enter a Gemini API key above to start chatting. Get a free one at [aistudio.google.com](https://aistudio.google.com).")
