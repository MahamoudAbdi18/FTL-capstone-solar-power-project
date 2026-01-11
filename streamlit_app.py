import streamlit as st
import os, sys, types, base64, mimetypes, time
import joblib, requests, hashlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Solar Dashboard",
    page_icon="☀️",
    layout="wide"
)

# ================= LANGUAGE SELECTOR =================
if "lang" not in st.session_state:
    st.session_state.lang = "fr"

with st.sidebar:
    st.markdown("### 🌐 Language / Langue")
    st.session_state.lang = st.radio(
        "",
        ["fr", "en"],
        format_func=lambda x: "🇫🇷 Français" if x == "fr" else "🇬🇧 English",
        horizontal=True
    )

LANG = st.session_state.lang

# ================= TRANSLATIONS =================
T = {
    "fr": {
        "title": "☀️ Tableau de bord de l’énergie solaire",
        "subtitle": "Prédictions à partir des données météo et temporelles.",
        "manual": "🖊️ Manuel",
        "batch": "📂 CSV en lot",
        "panel": "🔆 Évaluation des panneaux",
        "team": "👥 Équipe",
        "predict": "Prédire",
        "download": "⬇️ Télécharger CSV",
        "model_loaded": "Chargé",
        "variables": "Variables météo",
        "time_inputs": "Entrées temporelles",
        "single_pred": "Prédiction unique",
        "batch_pred": "Prédictions en lot",
        "upload_csv": "Téléverser un CSV",
        "team_title": "Rencontrez l’équipe",
        "prediction_done": "Prédiction terminée",
        "year_used": "Année utilisée",
    },
    "en": {
        "title": "☀️ Solar Energy Dashboard",
        "subtitle": "Predictions based on weather and temporal data.",
        "manual": "🖊️ Manual",
        "batch": "📂 Batch CSV",
        "panel": "🔆 Panel Evaluation",
        "team": "👥 Team",
        "predict": "Predict",
        "download": "⬇️ Download CSV",
        "model_loaded": "Loaded",
        "variables": "Weather variables",
        "time_inputs": "Time inputs",
        "single_pred": "Single prediction",
        "batch_pred": "Batch predictions",
        "upload_csv": "Upload a CSV",
        "team_title": "Meet the team",
        "prediction_done": "Prediction completed",
        "year_used": "Year used",
    }
}

def t(key: str) -> str:
    return T[LANG].get(key, key)

# ================= CSS =================
def inject_css():
    st.markdown("""
    <style>
      .block-container { max-width: 1160px; padding-top: 1rem; padding-bottom: 4rem; }
      h1, h2, h3 { letter-spacing: .1px; }
      .section-title { font-weight: 700; font-size: 1.15rem; margin: 1.1rem 0 .4rem;
                       padding-top: .4rem; border-top: 1px solid rgba(0,0,0,.06); }
      .card { background: var(--secondary-background-color);
              border: 1px solid rgba(0,0,0,.06); border-radius: 14px; padding: 14px 16px; }
      .stButton>button { border-radius: 10px; padding: .55rem 1rem; font-weight: 600; }
      div[data-testid="stMetricValue"] { font-size: 1.4rem; }
    </style>
    """, unsafe_allow_html=True)

inject_css()

# ================= HERO =================
left, right = st.columns([1, 1], vertical_alignment="center")

with left:
    st.title(t("title"))
    st.write(t("subtitle"))

with right:
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Model", t("model_loaded"))
        c2.metric(t("variables"), "6")
        c3.metric(t("time_inputs"), "Hour / Day / Month")

st.divider()

# ================= TABS =================
tab1, tab2, tab3, tab4 = st.tabs([
    t("manual"),
    t("batch"),
    t("panel"),
    t("team")
])

# ================= TAB 1 =================
with tab1:
    st.subheader(t("single_pred"))

    with st.form("manual_form"):
        temperature = st.number_input("Temperature (°C)", 25.0)
        humidity = st.number_input("Humidity (%)", 50.0)
        cloud = st.number_input("Cloud cover (%)", 20.0)
        submitted = st.form_submit_button(t("predict"))

    if submitted:
        st.success(t("prediction_done"))
        st.metric("Prediction (W/m²)", "—")

# ================= TAB 2 =================
with tab2:
    st.subheader(t("batch_pred"))
    file = st.file_uploader(t("upload_csv"), type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())
        st.download_button(
            t("download"),
            df.to_csv(index=False).encode("utf-8"),
            "predictions.csv",
            "text/csv"
        )

# ================= TAB 3 =================
with tab3:
    st.subheader(t("panel"))
    st.info("Photovoltaic evaluation logic unchanged.")

# ================= TAB 4 =================
with tab4:
    st.subheader(t("team_title"))
    st.write("Mahmoud Abdi • Moustapha Ali • Aboubaker Mohamed • Mohamed Achour")

# ================= FOOTER =================
st.markdown("---")
st.caption(f"© {datetime.now().year} • Solar Power Dashboard • Streamlit")
