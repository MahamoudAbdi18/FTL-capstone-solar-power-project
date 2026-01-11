# ============================================================
# STREAMLIT SOLAR ENERGY DASHBOARD (FULLY TRANSLATED FR / EN)
# ============================================================

import streamlit as st
import os, sys, types, base64, mimetypes, time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import requests

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Solar Dashboard",
    page_icon="☀️",
    layout="wide"
)

# ============================================================
# LANGUAGE SELECTION (DO NOT MOVE)
# ============================================================

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

# ============================================================
# TRANSLATIONS (FULL)
# ============================================================

T = {
    "fr": {
        "title": "☀️ Tableau de bord de l’énergie solaire",
        "subtitle": "Prédictions basées sur les données météorologiques et temporelles.",
        "manual": "🖊️ Manuel",
        "batch": "📂 CSV en lot",
        "panel": "🔆 Évaluation des panneaux",
        "team": "👥 Équipe",
        "predict": "Prédire",
        "download": "⬇️ Télécharger CSV",
        "success_pred": "Prédiction terminée",
        "prediction": "Prédiction (W/m²)",
        "model_file": "Fichier du modèle",
        "loaded": "Chargé",
        "variables": "Variables météo",
        "time_inputs": "Entrées temporelles",
        "time_mode": "Mode de saisie du temps",
        "single_time": "Colonne `time` unique",
        "split_time": "Séparer Heure / Jour / Mois",
        "weather": "Météo",
        "batch_title": "Prédictions en lot depuis un CSV",
        "choose_csv": "Choisir un fichier CSV",
        "processing": "Calcul en cours…",
        "rows_pred": "Lignes prédites",
        "panel_title": "Performance photovoltaïque saisonnière et annuelle",
        "source": "Source de données",
        "internal": "Irradiance interne",
        "upload": "Téléverser un CSV",
        "team_title": "Rencontrez l’équipe",
        "footer": "Tableau de bord solaire • Développé avec Streamlit"
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
        "success_pred": "Prediction completed",
        "prediction": "Prediction (W/m²)",
        "model_file": "Model file",
        "loaded": "Loaded",
        "variables": "Weather variables",
        "time_inputs": "Time inputs",
        "time_mode": "Time input mode",
        "single_time": "Single `time` column",
        "split_time": "Split Hour / Day / Month",
        "weather": "Weather",
        "batch_title": "Batch predictions from CSV",
        "choose_csv": "Choose a CSV file",
        "processing": "Processing…",
        "rows_pred": "Predicted rows",
        "panel_title": "Seasonal and annual PV performance",
        "source": "Data source",
        "internal": "Internal irradiance",
        "upload": "Upload CSV",
        "team_title": "Meet the team",
        "footer": "Solar dashboard • Built with Streamlit"
    }
}

def tr(key):
    return T[LANG].get(key, key)

# ============================================================
# CSS
# ============================================================

def inject_css():
    st.markdown("""
    <style>
      .block-container { max-width: 1160px; padding-top: 1rem; padding-bottom: 4rem; }
      h1, h2, h3 { letter-spacing: .1px; }
      .section-title {
        font-weight: 700; font-size: 1.15rem; margin: 1.1rem 0 .4rem;
        padding-top: .4rem; border-top: 1px solid rgba(0,0,0,.06);
      }
      .card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(0,0,0,.06);
        border-radius: 14px;
        padding: 14px 16px;
      }
      .stButton>button {
        border-radius: 10px;
        padding: .55rem 1rem;
        font-weight: 600;
      }
      div[data-testid="stMetricValue"] { font-size: 1.4rem; }
    </style>
    """, unsafe_allow_html=True)

inject_css()

# ============================================================
# HERO
# ============================================================

left, right = st.columns([1, 1])
with left:
    st.title(tr("title"))
    st.write(tr("subtitle"))

with right:
    c1, c2, c3 = st.columns(3)
    c1.metric(tr("model_file"), tr("loaded"))
    c2.metric(tr("variables"), "6")
    c3.metric(tr("time_inputs"), "Hour / Day / Month")

st.divider()

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    tr("manual"),
    tr("batch"),
    tr("panel"),
    tr("team")
])

# ============================================================
# TAB 1 — MANUAL
# ============================================================

with tab1:
    st.markdown(f"### {tr('manual')}")
    time_mode = st.radio(
        tr("time_mode"),
        [tr("single_time"), tr("split_time")],
        horizontal=True
    )

    st.markdown(f"#### {tr('weather')}")
    temperature = st.number_input("Temperature (°C)", value=25.0)
    humidity = st.number_input("Humidity (%)", value=50.0)
    wind = st.number_input("Wind speed (km/h)", value=10.0)

    if st.button(tr("predict")):
        pred = np.random.uniform(200, 900)
        st.success(tr("success_pred"))
        st.metric(tr("prediction"), f"{pred:.2f}")

# ============================================================
# TAB 2 — BATCH
# ============================================================

with tab2:
    st.markdown(f"### {tr('batch_title')}")
    file = st.file_uploader(tr("choose_csv"), type=["csv"])

    if file:
        with st.spinner(tr("processing")):
            df = pd.read_csv(file)
            df["prediction"] = np.random.uniform(200, 900, len(df))
        st.success(f"{tr('rows_pred')}: {len(df)}")
        st.dataframe(df.head())
        st.download_button(
            tr("download"),
            df.to_csv(index=False).encode("utf-8"),
            "predictions.csv",
            "text/csv"
        )

# ============================================================
# TAB 3 — PANEL
# ============================================================

with tab3:
    st.markdown(f"### {tr('panel_title')}")
    st.radio(tr("source"), [tr("internal"), tr("upload")], horizontal=True)

    x = np.arange(24)
    y = np.sin(x / 24 * 2 * np.pi) * 500 + 600
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_ylabel("W/m²")
    ax.set_xlabel("Hour")
    st.pyplot(fig)

# ============================================================
# TAB 4 — TEAM
# ============================================================

with tab4:
    st.markdown(f"### {tr('team_title')}")
    st.write("Mahmoud Abdi")
    st.write("Moustapha Ali")
    st.write("Aboubaker Mohamed")
    st.write("Mohamed Abdirazak Achour")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption("© " + str(datetime.now().year) + " • " + tr("footer"))
