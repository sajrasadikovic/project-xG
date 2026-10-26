import pandas as pd
import streamlit as st

from src.plots import shot_map

st.set_page_config(page_title="Lautaro xG explorer", layout="wide")
st.title("Lautaro Martinez in Serie A - shot explorer")

@st.cache_data
def load_data():
    df=pd.read_csv("app_data/lautaro_shots.csv")
    shots=df[df["situation"]!="Penalty"]
    penalties=df[df["situation"]=="Penalty"]
    return shots, penalties

shots, penalties = load_data()

seasons = ["All seasons"]+ sorted(shots["season_label"].unique())
choice = st.sidebar.selectbox("Season", seasons)
df= shots if choice == "All seasons" else shots [shots["season_label"]==choice]

goals = int (df["is_goal"].sum())
my_xg=df["my_xg"].sum()
us_xg=df["xG"].sum()

c1, c2, c3, c4, c5=st.columns(5)
c1.metric("Shots", len(df))
c2.metric("Goals", goals)
c3.metric("My xG", f"{my_xg:.1f}")
c4.metric("Understat xG", f"{us_xg:.1f}")
c5.metric("Goals minus my xG", f"{goals-my_xg:+.1f}")

left, right = st.columns(2)

with left:
    st.subheader("Shot map")
    st.pyplot(shot_map(df, f"Lautaro Martinez - {choice}"))

with right:
    st.subheader("Goals vs xG by season")
    trend =shots.groupby("season_label").agg(
        goals=("is_goal", "sum"),
        my_xg=("my_xg", "sum"),
        understat_xg=("xG", "sum"),
    )
    st.bar_chart(trend)
    st.dataframe(trend.round(1))

st.subheader("Penalties (not included above)")
st.write(f"{len(penalties)} penalties, {int(penalties['is_goal'].sum())} scored.")

with st.expander("How to read this"):
    st.write(
        "xG is the probability that a shot becomes a goal. My model uses distance, "
        "angle, header and open play, and was trained on other Serie A players, "
        "so it represents an average finisher. 'Goals minus xG' above zero means "
        "Lautaro scored more than expected."
    )

st.caption("Data: Understand. Serie A 2018/19 - 2025/26. Penalties exluded from xG.")