# import streamlit as st
# import os, sys, types
# import joblib
# import pandas as pd
# import numpy as np
# from datetime import datetime

# # --- Show runtime versions to debug env mismatches ---
# try:
#     import sklearn, numpy, pandas, joblib as _joblib
#     with st.sidebar:
#         st.caption("Runtime versions")
#         st.code(
#             f"python: 3.11 expected\n"
#             f"sklearn={sklearn.__version__}\n"
#             f"numpy={numpy.__version__}\n"
#             f"pandas={pandas.__version__}\n"
#             f"joblib={_joblib.__version__}",
#             language="text"
#         )
# except Exception:
#     pass

# # ---------------- TimeFeatures transformer ----------------
# RAW_TIME = 'time'
# TIME_FEATURES = ['Hour', 'Day', 'Month']
# BASE_COLS = [
#     'temperature_2m (°C)',
#     'relative_humidity_2m (%)',
#     'dew_point_2m (°C)',
#     'wind_speed_10m (km/h)',
#     'wind_direction_10m (°)',
#     'cloud_cover (%)'
# ]
# ALL_FEATURES_WITH_TIME = [RAW_TIME] + BASE_COLS + TIME_FEATURES

# class TimeFeatures:
#     """Transformer that extracts Hour/Day/Month from a 'time' column."""
#     def fit(self, X, y=None): return self
#     def transform(self, X):
#         X = X.copy()
#         if RAW_TIME in X.columns:
#             X[RAW_TIME] = pd.to_datetime(X[RAW_TIME])
#             X['Hour']  = X[RAW_TIME].dt.hour
#             X['Day']   = X[RAW_TIME].dt.day
#             X['Month'] = X[RAW_TIME].dt.month
#             X = X.drop(columns=[RAW_TIME])
#         for c in TIME_FEATURES:
#             if c not in X.columns:
#                 X[c] = 0
#         return X
#     def get_params(self, deep=True): return {}
#     def set_params(self, **params): return self

# # Make TimeFeatures available under __main__ for joblib unpickling
# if "__main__" not in sys.modules:
#     sys.modules["__main__"] = types.ModuleType("__main__")
# setattr(sys.modules["__main__"], "TimeFeatures", TimeFeatures)
# # ------------------------------------------------------------

# MODEL_PATH = "model_stacking_pipeline.pkl"


# def _maybe_download_model():
#     """Optional fallback: download model if MODEL_URL env var is set."""
#     url = os.environ.get("MODEL_URL", "").strip()
#     if url and not os.path.exists(MODEL_PATH):
#         import urllib.request
#         st.info("Downloading model…")
#         urllib.request.urlretrieve(url, MODEL_PATH)
#         st.success("Model downloaded.")

# @st.cache_resource
# def load_model():
#     # Try local file first
#     if not os.path.exists(MODEL_PATH):
#         _maybe_download_model()
#     if not os.path.exists(MODEL_PATH):
#         st.error("Model file not found. "
#                  "Ensure `model_stacking_pipeline.pkl` is in the repo (Git LFS), "
#                  "or set an env var MODEL_URL to a direct download link.")
#         st.stop()
#     try:
#         return joblib.load(MODEL_PATH)
#     except Exception as e:
#         st.error(f"Failed to load model: {e}")
#         st.stop()

# model = load_model()




# st.title("☀️ Solar Power Prediction — Saisie manuelle")

# st.write("Renseignez soit un **horaire** (champ unique), soit **Heure/Jour/Mois**, "
#          "puis les variables météo suivantes :")
# st.code(", ".join(BASE_COLS), language="text")

# # --- Choix de la méthode de temps ---
# time_mode = st.radio(
#     "Comment fournir le temps ?",
#     options=["Colonne 'time' (unique)", "Heure / Jour / Mois"],
#     horizontal=True,
# )

# with st.form("manual_input_form"):
#     st.subheader("Temps")
#     if time_mode == "Colonne 'time' (unique)":
#         date_val = st.date_input("Date")
#         time_val = st.time_input("Heure")
#         dt = pd.to_datetime(f"{date_val} {time_val}")
#         hour = day = month = None
#     else:
#         col_h, col_d, col_m = st.columns(3)
#         with col_h:
#             hour = st.number_input("Hour", min_value=0, max_value=23, value=12, step=1)
#         with col_d:
#             day = st.number_input("Day", min_value=1, max_value=31, value=15, step=1)
#         with col_m:
#             month = st.number_input("Month", min_value=1, max_value=12, value=6, step=1)
#         dt = None

#     st.subheader("Variables météo")
#     c1, c2 = st.columns(2)
#     with c1:
#         temperature_2m = st.number_input("temperature_2m (°C)", value=25.0)
#         relative_humidity_2m = st.number_input("relative_humidity_2m (%)", min_value=0.0, max_value=100.0, value=50.0)
#         dew_point_2m = st.number_input("dew_point_2m (°C)", value=15.0)
#     with c2:
#         wind_speed_10m = st.number_input("wind_speed_10m (km/h)", min_value=0.0, value=10.0)
#         wind_direction_10m = st.number_input("wind_direction_10m (°)", min_value=0.0, max_value=360.0, value=180.0)
#         cloud_cover = st.number_input("cloud_cover (%)", min_value=0.0, max_value=100.0, value=20.0)

#     submitted = st.form_submit_button("Prédire")

# if submitted:
#     row = {}
#     if time_mode == "Colonne 'time' (unique)":
#         row[RAW_TIME] = dt
#         row["Hour"] = 0
#         row["Day"] = 0
#         row["Month"] = 0
#     else:
#         row["Hour"] = int(hour)
#         row["Day"] = int(day)
#         row["Month"] = int(month)
#         row[RAW_TIME] = pd.NaT

#     row['temperature_2m (°C)'] = float(temperature_2m)
#     row['relative_humidity_2m (%)'] = float(relative_humidity_2m)
#     row['dew_point_2m (°C)'] = float(dew_point_2m)
#     row['wind_speed_10m (km/h)'] = float(wind_speed_10m)
#     row['wind_direction_10m (°)'] = float(wind_direction_10m)
#     row['cloud_cover (%)'] = float(cloud_cover)

#     for col in ALL_FEATURES_WITH_TIME:
#         row.setdefault(col, 0 if col in TIME_FEATURES else 0.0)

#     X = pd.DataFrame([row], columns=ALL_FEATURES_WITH_TIME)

#     try:
#         y = model.predict(X)
#         pred = float(y[0])
#         st.success("✅ Prédiction effectuée")
#         st.metric("Prediction (W/m²)", f"{pred:,.2f}")
#         out = X.copy()
#         out["prediction_W_m2"] = pred
#         st.download_button(
#             "⬇️ Télécharger la prédiction (CSV)",
#             data=out.to_csv(index=False).encode("utf-8"),
#             file_name="prediction_unique.csv",
#             mime="text/csv",
#         )
#     except Exception as e:
#         st.error(f"Erreur de prédiction : {e}")

# #*****************************

# import matplotlib.pyplot as plt

# st.markdown("---")
# st.header("📈 Évaluation de performance des panneaux (à partir d’un CSV)")

# eval_file = st.file_uploader("Importer un CSV (avec une colonne d’irradiance inclinée)", type=["csv"], key="eval_csv")

# with st.expander("Paramètres des panneaux (modifiable)"):
#     # Dimensions en mètres (L x l) et Pmax en W
#     default_panels = {
#         "TRINA": {"Pmax": 620.0, "dims": (2.382, 1.134)},
#         "LONGi": {"Pmax": 550.0, "dims": (2.278, 1.134)},
#         "JINKO": {"Pmax": 460.0, "dims": (2.182, 1.029)},
#     }
#     cols = st.columns(3)
#     edited = {}
#     for i, name in enumerate(default_panels.keys()):
#         with cols[i]:
#             st.markdown(f"**{name}**")
#             pmax = st.number_input(f"Pmax {name} (W)", value=float(default_panels[name]["Pmax"]), step=10.0, key=f"pmax_{name}")
#             L = st.number_input(f"Longueur {name} (m)", value=float(default_panels[name]["dims"][0]), step=0.001, format="%.3f", key=f"L_{name}")
#             W = st.number_input(f"Largeur {name} (m)", value=float(default_panels[name]["dims"][1]), step=0.001, format="%.3f", key=f"W_{name}")
#             edited[name] = {"Pmax": pmax, "dims": (L, W)}

# st.caption("Astuce : l’irradiance inclinée est souvent nommée comme `global_tilted_irradiance`.")

# if eval_file is not None:
#     try:
#         df = pd.read_csv(eval_file)
#         # Nettoyage index
#         if "Unnamed: 0" in df.columns:
#             df = df.drop(columns="Unnamed: 0")
#         if "time" not in df.columns:
#             st.error("Colonne `time` manquante dans le CSV.")
#             st.stop()
#         df["time"] = pd.to_datetime(df["time"], errors="coerce")
#         df = df.dropna(subset=["time"]).set_index("time").sort_index()

#         # Détection colonnes
#         tilt_candidates = [c for c in df.columns if "global_tilted_irradiance" in c.lower()]
#         if not tilt_candidates:
#             st.error("Aucune colonne contenant 'global_tilted_irradiance' trouvée.")
#             st.stop()
#         tilt_col = tilt_candidates[0]

#         horiz_cols = [c for c in df.columns if ("horizontal" in c.lower() and "irradiance" in c.lower())]
#         horiz_col = horiz_cols[0] if horiz_cols else None

#         # Horaire → moyenne horaire
#         df_hourly = df.resample("H").mean(numeric_only=True)
#         df_hourly["date"] = df_hourly.index.normalize()
#         df_hourly["year"] = df_hourly.index.year

#         # Choix de l’année avec le plus de jours couverts
#         days_per_year = df_hourly.groupby("year")["date"].nunique().sort_values(ascending=False)
#         if days_per_year.empty:
#             st.error("Aucune donnée horaire après agrégation.")
#             st.stop()
#         best_year = int(days_per_year.index[0])
#         year_df = df_hourly[df_hourly.index.year == best_year].copy()

#         # Énergie journalière proxy (Wh/m² ≈ somme des W/m² horaires)
#         daily_energy = year_df[tilt_col].resample("D").sum(min_count=1).dropna()
#         daily_energy.index = daily_energy.index.normalize()

#         def in_range(start_mmdd, end_mmdd):
#             return (daily_energy.index >= pd.Timestamp(f"{best_year}-{start_mmdd}")) & \
#                    (daily_energy.index <= pd.Timestamp(f"{best_year}-{end_mmdd}"))

#         windows = {
#             "Spring (≈Mar 20)": in_range("03-10", "03-30"),
#             "Summer (≈Jun 21)": in_range("06-11", "07-01"),
#             "Autumn (≈Sep 22)": in_range("09-12", "10-02"),
#             "Winter (≈Dec 21)": in_range("12-11", "12-31"),
#         }
#         qmask = {
#             "Spring (≈Mar 20)": ((daily_energy.index >= f"{best_year}-01-01") & (daily_energy.index <= f"{best_year}-03-31")),
#             "Summer (≈Jun 21)": ((daily_energy.index >= f"{best_year}-04-01") & (daily_energy.index <= f"{best_year}-06-30")),
#             "Autumn (≈Sep 22)": ((daily_energy.index >= f"{best_year}-07-01") & (daily_energy.index <= f"{best_year}-09-30")),
#             "Winter (≈Dec 21)": ((daily_energy.index >= f"{best_year}-10-01") & (daily_energy.index <= f"{best_year}-12-31")),
#         }

#         # Sélection des 4 jours
#         selected = {}
#         for label, wmask in windows.items():
#             cand = daily_energy[wmask]
#             if not cand.empty:
#                 selected[label] = cand.idxmax().date()
#             else:
#                 qcand = daily_energy[qmask[label]]
#                 if not qcand.empty:
#                     selected[label] = qcand.idxmax().date()

#         # Unicité
#         used = set()
#         for label in list(selected.keys()):
#             d = selected[label]
#             if d in used:
#                 wmask = windows[label]
#                 fallback = daily_energy[wmask].drop(pd.Timestamp(d), errors="ignore")
#                 if fallback.empty:
#                     fallback = daily_energy[qmask[label]].drop(pd.Timestamp(d), errors="ignore")
#                 if not fallback.empty:
#                     selected[label] = fallback.idxmax().date()
#             used.add(selected[label])

#         ordered_labels = ["Spring (≈Mar 20)", "Summer (≈Jun 21)", "Autumn (≈Sep 22)", "Winter (≈Dec 21)"]
#         selected_dates = [pd.Timestamp(selected[l]) for l in ordered_labels if l in selected]
#         if len(selected_dates) < 4:
#             top = daily_energy.sort_values(ascending=False)
#             uniq = []
#             for d in top.index:
#                 if d.date() not in uniq:
#                     uniq.append(d.date())
#                 if len(uniq) == 4:
#                     break
#             selected_dates = [pd.Timestamp(d) for d in sorted(uniq)]

#         season_fr = {
#             "Spring (≈Mar 20)":  "Équinoxe de printemps",
#             "Summer (≈Jun 21)":  "Solstice d’été",
#             "Autumn (≈Sep 22)":  "Équinoxe d’automne",
#             "Winter (≈Dec 21)":  "Solstice d’hiver",
#         }

#         # Construire les blocs journaliers
#         blocks, labels_for_blocks = [], []
#         for lab, d in zip(ordered_labels, selected_dates):
#             day_block = year_df[year_df.index.normalize() == d]
#             if not day_block.empty:
#                 blocks.append(day_block)
#                 labels_for_blocks.append(season_fr[lab])

#         if len(blocks) < 2:
#             st.error("Jours distincts insuffisants pour tracer la figure saisonnière.")
#             st.stop()

#         df_sel = pd.concat(blocks, axis=0)
#         df_sel["time_series_h"] = np.arange(1, len(df_sel) + 1)

#         # Séparateurs verticaux
#         boundary_idx = []
#         cum = 0
#         for blk in blocks[:-1]:
#             cum += len(blk)
#             boundary_idx.append(cum)

#         # Puissance surfacique instantanée (W/m²) pour chaque panneau
#         panels = edited  # depuis l’expander
#         for name, p in panels.items():
#             area = p["dims"][0] * p["dims"][1]          # m²
#             eta  = p["Pmax"] / (1000.0 * area)          # rendement STC
#             df_sel[name] = eta * df_sel[tilt_col]       # W/m²

#         # ===== Figure saisonnière (deux sous-graphiques) =====
#         fig = plt.figure(figsize=(11, 8))

#         ax1 = fig.add_subplot(2, 1, 1)
#         if horiz_col is not None:
#             ax1.plot(df_sel["time_series_h"], df_sel[horiz_col], linewidth=2, label="Irradiance horizontale globale")
#         ax1.plot(df_sel["time_series_h"], df_sel[tilt_col], linewidth=2, label="Irradiance sur panneaux inclinés")
#         for x in boundary_idx:
#             ax1.axvline(x)
#         # Étiquettes centrées
#         start = 0
#         for i, blk in enumerate(blocks):
#             end = start + len(blk)
#             center = (start + end) / 2
#             ax1.text(center, ax1.get_ylim()[1] * 0.95, labels_for_blocks[i], ha="center", va="top", fontsize=10, fontweight="bold")
#             start = end
#         ncols = 2 if horiz_col is not None else 1
#         ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.18), ncol=ncols, frameon=True)
#         ax1.set_ylabel("Rayonnement solaire (W/m²)")
#         ax1.set_xticks([])

#         ax2 = fig.add_subplot(2, 1, 2)
#         for name in panels.keys():
#             style = "--" if name.upper() == "JINKO" else "-"
#             ax2.plot(df_sel["time_series_h"], df_sel[name], linewidth=2, linestyle=style, label=name)
#         for x in boundary_idx:
#             ax2.axvline(x)
#         ax2.set_ylabel("Puissance de sortie (W/m²)")
#         ax2.set_xlabel("Série temporelle (h)")
#         ax2.legend(loc="upper left", frameon=True)

#         plt.tight_layout()
#         st.pyplot(fig, clear_figure=True)

#         st.info(f"Année utilisée : **{best_year}** · Jours sélectionnés : " +
#                 ", ".join([d.strftime('%Y-%m-%d') for d in selected_dates]))

#         # ===== Énergie annuelle =====
#         annual_density_kWh_m2 = {}
#         annual_module_kWh = {}
#         for name, p in panels.items():
#             area = p["dims"][0] * p["dims"][1]
#             eta  = p["Pmax"] / (1000.0 * area)
#             power_density = eta * year_df[tilt_col]           # W/m² (horaire)
#             energy_density_kWh_m2 = power_density.sum() / 1000.0  # Wh/m² -> kWh/m²
#             annual_density_kWh_m2[name] = energy_density_kWh_m2
#             annual_module_kWh[name] = energy_density_kWh_m2 * area

#         labels = list(panels.keys())
#         vals_module = [annual_module_kWh[k] for k in labels]
#         vals_density = [annual_density_kWh_m2[k] for k in labels]

#         fig4 = plt.figure(figsize=(10, 8))
#         ax4a = fig4.add_subplot(2, 1, 1)
#         bars1 = ax4a.bar(labels, vals_module)
#         ax4a.set_ylabel("Énergie annuelle (kWh/an)")
#         ax4a.set_title("(a) Module unique")
#         for b, v in zip(bars1, vals_module):
#             ax4a.text(b.get_x() + b.get_width()/2, b.get_height()*1.01, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

#         ax4b = fig4.add_subplot(2, 1, 2)
#         bars2 = ax4b.bar(labels, vals_density)
#         ax4b.set_ylabel("Énergie annuelle (kWh/m²·an)")
#         ax4b.set_title("(b) Par unité de surface")
#         ax4b.set_xlabel("Panneau")
#         for b, v in zip(bars2, vals_density):
#             ax4b.text(b.get_x() + b.get_width()/2, b.get_height()*1.01, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

#         plt.tight_layout()
#         st.pyplot(fig4, clear_figure=True)

#         # Résumé numérique
#         st.subheader("Résumé annuel estimé")
#         st.dataframe(
#             pd.DataFrame({
#                 "Panneau": labels,
#                 "Énergie module (kWh/an)": [round(x, 1) for x in vals_module],
#                 "Énergie surfacique (kWh/m²·an)": [round(x, 1) for x in vals_density],
#             })
#         )

#     except Exception as e:
#         st.error(f"Erreur durant l’évaluation : {e}")
# else:
#     st.info("Importez un CSV pour lancer l’évaluation.")


# #**************
# st.title("☀️ Solar Power Prediction ")

# st.write("Upload a CSV that contains either a `time` column **or** the three columns "
#          "`Hour`, `Day`, `Month`, plus these weather columns:")
# st.code(", ".join(BASE_COLS), language="text")

# file = st.file_uploader("Choose CSV", type=["csv"])
# if file is not None:
#     try:
#         df = pd.read_csv(file)

#         # Ensure time features exist (TimeFeatures in the pipeline will also help)
#         for c in TIME_FEATURES:
#             if c not in df.columns:
#                 df[c] = 0

#         # Keep/align columns; fill any missing weather cols with 0 (StandardScaler can't handle NaN)
#         for col in ALL_FEATURES_WITH_TIME:
#             if col not in df.columns:
#                 df[col] = 0.0

#         X = df[ALL_FEATURES_WITH_TIME]
#         preds = model.predict(X)
#         out = df.copy()
#         out["prediction_W_m2"] = preds

#         st.success(f"Predicted {len(out)} rows.")
#         st.dataframe(out.head(100), use_container_width=True)
#         st.download_button("⬇️ Download predictions.csv",
#                            out.to_csv(index=False).encode("utf-8"),
#                           "predictions.csv",
#                           "text/csv")
#     except Exception as e:
#         st.error(f"Batch prediction error: {e}")


import streamlit as st
import os, sys, types
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ========= Où se trouve TON fichier d’irradiance interne ? =========
# Mets ici le chemin vers ton CSV dans le repo (par ex. data/Energy_solar.csv)
IRR_PATH = "Energy_solar.csv"


# --- Show runtime versions to debug env mismatches ---
try:
    import sklearn, numpy, pandas, joblib as _joblib
    with st.sidebar:
        st.caption("Runtime versions")
        st.code(
            f"python: 3.11 expected\n"
            f"sklearn={sklearn.__version__}\n"
            f"numpy={numpy.__version__}\n"
            f"pandas={pandas.__version__}\n"
            f"joblib={_joblib.__version__}",
            language="text"
        )
except Exception:
    pass

# ---------------- TimeFeatures transformer ----------------
RAW_TIME = 'time'
TIME_FEATURES = ['Hour', 'Day', 'Month']
BASE_COLS = [
    'temperature_2m (°C)',
    'relative_humidity_2m (%)',
    'dew_point_2m (°C)',
    'wind_speed_10m (km/h)',
    'wind_direction_10m (°)',
    'cloud_cover (%)'
]
ALL_FEATURES_WITH_TIME = [RAW_TIME] + BASE_COLS + TIME_FEATURES

class TimeFeatures:
    """Transformer that extracts Hour/Day/Month from a 'time' column."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        if RAW_TIME in X.columns:
            X[RAW_TIME] = pd.to_datetime(X[RAW_TIME])
            X['Hour']  = X[RAW_TIME].dt.hour
            X['Day']   = X[RAW_TIME].dt.day
            X['Month'] = X[RAW_TIME].dt.month
            X = X.drop(columns=[RAW_TIME])
        for c in TIME_FEATURES:
            if c not in X.columns:
                X[c] = 0
        return X
    def get_params(self, deep=True): return {}
    def set_params(self, **params): return self

# Rendre TimeFeatures accessible pour joblib
if "__main__" not in sys.modules:
    sys.modules["__main__"] = types.ModuleType("__main__")
setattr(sys.modules["__main__"], "TimeFeatures", TimeFeatures)

# ---------------- Chargement du modèle ----------------
MODEL_PATH = "model_stacking_pipeline.pkl"

def _maybe_download_model():
    url = os.environ.get("MODEL_URL", "").strip()
    if url and not os.path.exists(MODEL_PATH):
        import urllib.request
        st.info("Downloading model…")
        urllib.request.urlretrieve(url, MODEL_PATH)
        st.success("Model downloaded.")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        _maybe_download_model()
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. "
                 "Assure-toi que `model_stacking_pipeline.pkl` est dans le repo (Git LFS), "
                 "ou définis la variable d’env. MODEL_URL vers un lien direct.")
        st.stop()
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

model = load_model()

# ======================= Onglets =======================
st.title("☀️ Solar Power Dashboard ")
tab1, tab2, tab3, tab4 = st.tabs(["🖊️ Saisie manuelle", "📂 Batch CSV", "🔆 Évaluation panneaux", "👥 Équipe"])


# ---------------- TAB 1 : Saisie manuelle ----------------
with tab1:
    st.subheader("Prédiction unitaire — saisie manuelle")
    st.write("Renseignez soit un **horaire** unique, soit **Heure/Jour/Mois**, puis les variables météo :")
    st.code(", ".join(BASE_COLS), language="text")

    time_mode = st.radio(
        "Comment fournir le temps ?",
        options=["Colonne 'time' (unique)", "Heure / Jour / Mois"],
        horizontal=True,
        key="time_mode_manual"
    )

    with st.form("manual_input_form"):
        st.markdown("**Temps**")
        if time_mode == "Colonne 'time' (unique)":
            date_val = st.date_input("Date")
            time_val = st.time_input("Heure")
            dt = pd.to_datetime(f"{date_val} {time_val}")
            hour = day = month = None
        else:
            col_h, col_d, col_m = st.columns(3)
            with col_h:
                hour = st.number_input("Hour", min_value=0, max_value=23, value=12, step=1)
            with col_d:
                day = st.number_input("Day", min_value=1, max_value=31, value=15, step=1)
            with col_m:
                month = st.number_input("Month", min_value=1, max_value=12, value=6, step=1)
            dt = None

        st.markdown("**Variables météo**")
        c1, c2 = st.columns(2)
        with c1:
            temperature_2m = st.number_input("temperature_2m (°C)", value=25.0)
            relative_humidity_2m = st.number_input("relative_humidity_2m (%)", min_value=0.0, max_value=100.0, value=50.0)
            dew_point_2m = st.number_input("dew_point_2m (°C)", value=15.0)
        with c2:
            wind_speed_10m = st.number_input("wind_speed_10m (km/h)", min_value=0.0, value=10.0)
            wind_direction_10m = st.number_input("wind_direction_10m (°)", min_value=0.0, max_value=360.0, value=180.0)
            cloud_cover = st.number_input("cloud_cover (%)", min_value=0.0, max_value=100.0, value=20.0)

        submitted = st.form_submit_button("Prédire")

    if submitted:
        row = {}
        if time_mode == "Colonne 'time' (unique)":
            row[RAW_TIME] = dt
            row["Hour"] = 0
            row["Day"] = 0
            row["Month"] = 0
        else:
            row["Hour"] = int(hour)
            row["Day"] = int(day)
            row["Month"] = int(month)
            row[RAW_TIME] = pd.NaT

        row['temperature_2m (°C)'] = float(temperature_2m)
        row['relative_humidity_2m (%)'] = float(relative_humidity_2m)
        row['dew_point_2m (°C)'] = float(dew_point_2m)
        row['wind_speed_10m (km/h)'] = float(wind_speed_10m)
        row['wind_direction_10m (°)'] = float(wind_direction_10m)
        row['cloud_cover (%)'] = float(cloud_cover)

        for col in ALL_FEATURES_WITH_TIME:
            row.setdefault(col, 0 if col in TIME_FEATURES else 0.0)

        X = pd.DataFrame([row], columns=ALL_FEATURES_WITH_TIME)

        try:
            y = model.predict(X)
            pred = float(y[0])
            st.success("✅ Prédiction effectuée")
            st.metric("Prediction (W/m²)", f"{pred:,.2f}")
            out = X.copy()
            out["prediction_W_m2"] = pred
            st.download_button(
                "⬇️ Télécharger la prédiction (CSV)",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="prediction_unique.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Erreur de prédiction : {e}")

# ---------------- TAB 2 : Batch CSV ----------------
with tab2:
    st.subheader("Prédictions en lot — CSV")
    st.write("Le CSV doit contenir une colonne `time` **ou** les colonnes `Hour`, `Day`, `Month`, plus :")
    st.code(", ".join(BASE_COLS), language="text")

    file = st.file_uploader("Choose CSV", type=["csv"], key="batch_csv")
    if file is not None:
        try:
            df = pd.read_csv(file)

            # Garantir les colonnes temporelles
            for c in TIME_FEATURES:
                if c not in df.columns:
                    df[c] = 0

            # Aligner / compléter les colonnes attendues
            for col in ALL_FEATURES_WITH_TIME:
                if col not in df.columns:
                    df[col] = 0.0

            X = df[ALL_FEATURES_WITH_TIME]
            preds = model.predict(X)
            out = df.copy()
            out["prediction_W_m2"] = preds

            st.success(f"Predicted {len(out)} rows.")
            st.dataframe(out.head(100), use_container_width=True)
            st.download_button("⬇️ Download predictions.csv",
                               out.to_csv(index=False).encode("utf-8"),
                               "predictions.csv",
                               "text/csv")
        except Exception as e:
            st.error(f"Batch prediction error: {e}")

# ---------------- TAB 3 : Évaluation panneaux (irradiance interne) ----------------
with tab3:
    st.subheader("Évaluation de performance des panneaux (irradiance interne)")

    def load_internal_irradiance() -> pd.DataFrame:
        # Priorité : DataFrame déjà chargé ailleurs
        if "irr_df" in st.session_state and isinstance(st.session_state["irr_df"], pd.DataFrame):
            df = st.session_state["irr_df"].copy()
            src = "session_state"
        else:
            if not os.path.exists(IRR_PATH):
                st.error(
                    "Irradiance introuvable. "
                    "Soit place ton DataFrame dans st.session_state['irr_df'], "
                    f"soit ajoute un fichier au chemin interne : `{IRR_PATH}`."
                )
                st.stop()
            df = pd.read_csv(IRR_PATH)
            src = IRR_PATH

        if "Unnamed: 0" in df.columns:
            df = df.drop(columns="Unnamed: 0")
        if "time" not in df.columns:
            st.error(f"La colonne `time` est manquante dans la source irradiance ({src}).")
            st.stop()
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).set_index("time").sort_index()

        return df

    irr = load_internal_irradiance()

    # Détection colonnes
    tilt_candidates = [c for c in irr.columns if "global_tilted_irradiance" in c.lower()]
    if not tilt_candidates:
        st.error("Aucune colonne contenant 'global_tilted_irradiance' trouvée.")
        st.stop()
    tilt_col = tilt_candidates[0]

    horiz_cols = [c for c in irr.columns if ("horizontal" in c.lower() and "irradiance" in c.lower())]
    horiz_col = horiz_cols[0] if horiz_cols else None

    # Agrégation horaire + sélection meilleure année couverte
    df_hourly = irr.resample("H").mean(numeric_only=True)
    df_hourly["date"] = df_hourly.index.normalize()
    df_hourly["year"] = df_hourly.index.year
    days_per_year = df_hourly.groupby("year")["date"].nunique().sort_values(ascending=False)
    if days_per_year.empty:
        st.error("Aucune donnée après agrégation horaire.")
        st.stop()
    best_year = int(days_per_year.index[0])
    year_df = df_hourly[df_hourly.index.year == best_year].copy()

    # Énergie journalière proxy
    daily_energy = year_df[tilt_col].resample("D").sum(min_count=1).dropna()
    daily_energy.index = daily_energy.index.normalize()

    def in_range(start_mmdd, end_mmdd):
        return (daily_energy.index >= pd.Timestamp(f"{best_year}-{start_mmdd}")) & \
               (daily_energy.index <= pd.Timestamp(f"{best_year}-{end_mmdd}"))

    windows = {
        "Spring (≈Mar 20)": in_range("03-10", "03-30"),
        "Summer (≈Jun 21)": in_range("06-11", "07-01"),
        "Autumn (≈Sep 22)": in_range("09-12", "10-02"),
        "Winter (≈Dec 21)": in_range("12-11", "12-31"),
    }
    qmask = {
        "Spring (≈Mar 20)": ((daily_energy.index >= f"{best_year}-01-01") & (daily_energy.index <= f"{best_year}-03-31")),
        "Summer (≈Jun 21)": ((daily_energy.index >= f"{best_year}-04-01") & (daily_energy.index <= f"{best_year}-06-30")),
        "Autumn (≈Sep 22)": ((daily_energy.index >= f"{best_year}-07-01") & (daily_energy.index <= f"{best_year}-09-30")),
        "Winter (≈Dec 21)": ((daily_energy.index >= f"{best_year}-10-01") & (daily_energy.index <= f"{best_year}-12-31")),
    }

    # Sélection des 4 jours
    selected = {}
    for label, wmask in windows.items():
        cand = daily_energy[wmask]
        if not cand.empty:
            selected[label] = cand.idxmax().date()
        else:
            qcand = daily_energy[qmask[label]]
            if not qcand.empty:
                selected[label] = qcand.idxmax().date()

    # Unicité
    used = set()
    for label in list(selected.keys()):
        d = selected[label]
        if d in used:
            wmask = windows[label]
            fallback = daily_energy[wmask].drop(pd.Timestamp(d), errors="ignore")
            if fallback.empty:
                fallback = daily_energy[qmask[label]].drop(pd.Timestamp(d), errors="ignore")
            if not fallback.empty:
                selected[label] = fallback.idxmax().date()
        if label in selected:
            used.add(selected[label])

    ordered_labels = ["Spring (≈Mar 20)", "Summer (≈Jun 21)", "Autumn (≈Sep 22)", "Winter (≈Dec 21)"]
    selected_dates = [pd.Timestamp(selected[l]) for l in ordered_labels if l in selected]
    if len(selected_dates) < 4:
        top = daily_energy.sort_values(ascending=False)
        uniq = []
        for d in top.index:
            if d.date() not in uniq:
                uniq.append(d.date())
            if len(uniq) == 4:
                break
        selected_dates = [pd.Timestamp(d) for d in sorted(uniq)]

    season_fr = {
        "Spring (≈Mar 20)":  "Équinoxe de printemps",
        "Summer (≈Jun 21)":  "Solstice d’été",
        "Autumn (≈Sep 22)":  "Équinoxe d’automne",
        "Winter (≈Dec 21)":  "Solstice d’hiver",
    }

    # >>>>>>>>>>>>>>> ICI : EXPANDER *DANS* tab3 (avec clés uniques) <<<<<<<<<<<<<<<
    with st.expander("⚙️ Panneaux à comparer (nom, Pmax, dimensions)"):
        nb_panels = st.number_input(
            "Nombre de panneaux à comparer",
            min_value=1, max_value=8, value=3, step=1, key="nb_panels_tab3"
        )

        defaults = [
            ("TRINA", 620.0, 2.382, 1.134),
            ("LONGi", 550.0, 2.278, 1.134),
            ("JINKO", 460.0, 2.182, 1.029),
        ]

        panels = {}
        for i in range(int(nb_panels)):
            name_d, pmax_d, L_d, W_d = defaults[i] if i < len(defaults) else (f"PANEL{i+1}", 500.0, 1.800, 1.100)
            c1, c2, c3, c4 = st.columns([1.2, 0.8, 0.8, 0.8])
            with c1:
                name = st.text_input(f"Nom panneau #{i+1}", value=name_d, key=f"name_{i}_tab3")
            with c2:
                pmax = st.number_input(f"Pmax #{i+1} (W)", min_value=1.0, value=pmax_d, step=10.0, key=f"pmax_{i}_tab3")
            with c3:
                L = st.number_input(f"Longueur #{i+1} (m)", min_value=0.3, value=L_d, step=0.001, format="%.3f", key=f"L_{i}_tab3")
            with c4:
                W = st.number_input(f"Largeur #{i+1} (m)", min_value=0.3, value=W_d, step=0.001, format="%.3f", key=f"W_{i}_tab3")

            panels[name.strip() or f"PANEL{i+1}"] = {"Pmax": float(pmax), "dims": (float(L), float(W))}

        # Dé-duplique proprement (Panel, Panel_1, …)
        dedup, seen = {}, set()
        for i, (nm, spec) in enumerate(panels.items(), start=1):
            base = nm or f"PANEL{i}"
            n = base
            k = 1
            while n in seen:
                n = f"{base}_{k}"
                k += 1
            dedup[n] = spec
            seen.add(n)
        panels = dedup
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # Construire blocs journaliers
    blocks, labels_for_blocks = [], []
    for lab, d in zip(ordered_labels, selected_dates):
        day_block = year_df[year_df.index.normalize() == d]
        if not day_block.empty:
            blocks.append(day_block)
            labels_for_blocks.append(season_fr[lab])

    if len(blocks) < 2:
        st.error("Jours distincts insuffisants pour tracer la figure saisonnière.")
        st.stop()

    df_sel = pd.concat(blocks, axis=0)
    df_sel["time_series_h"] = np.arange(1, len(df_sel) + 1)

    # Séparateurs
    boundary_idx, cum = [], 0
    for blk in blocks[:-1]:
        cum += len(blk)
        boundary_idx.append(cum)

    # Puissance surfacique instantanée (W/m²)
    for name, p in panels.items():
        area = p["dims"][0] * p["dims"][1]
        eta  = p["Pmax"] / (1000.0 * area)
        df_sel[name] = eta * df_sel[tilt_col]

    # Figure saisonnière
    fig = plt.figure(figsize=(11, 8))
    ax1 = fig.add_subplot(2, 1, 1)
    if horiz_col is not None:
        ax1.plot(df_sel["time_series_h"], df_sel[horiz_col], linewidth=2, label="Irradiance horizontale globale")
    ax1.plot(df_sel["time_series_h"], df_sel[tilt_col], linewidth=2, label="Irradiance sur panneaux inclinés")
    for x in boundary_idx: ax1.axvline(x)
    start = 0
    for i, blk in enumerate(blocks):
        end = start + len(blk)
        center = (start + end) / 2
        ax1.text(center, ax1.get_ylim()[1] * 0.95, labels_for_blocks[i], ha="center", va="top", fontsize=10, fontweight="bold")
        start = end
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.18), ncol=2 if horiz_col is not None else 1, frameon=True)
    ax1.set_ylabel("Rayonnement solaire (W/m²)")
    ax1.set_xticks([])

    ax2 = fig.add_subplot(2, 1, 2)
    for idx, (name, _) in enumerate(panels.items()):
        linestyle = "--" if idx % 2 else "-"
        ax2.plot(df_sel["time_series_h"], df_sel[name], linewidth=2, linestyle=linestyle, label=name)
    for x in boundary_idx: ax2.axvline(x)
    ax2.set_ylabel("Puissance de sortie (W/m²)")
    ax2.set_xlabel("Série temporelle (h)")
    ax2.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    st.info(f"Année utilisée : **{best_year}** · Jours sélectionnés : " +
            ", ".join([d.strftime('%Y-%m-%d') for d in selected_dates]))

    # Énergie annuelle
    annual_density_kWh_m2, annual_module_kWh = {}, {}
    for name, p in panels.items():
        area = p["dims"][0] * p["dims"][1]
        eta  = p["Pmax"] / (1000.0 * area)
        power_density = eta * year_df[tilt_col]              # W/m² horaire
        energy_density_kWh_m2 = power_density.sum() / 1000.0 # Wh/m² -> kWh/m²
        annual_density_kWh_m2[name] = energy_density_kWh_m2
        annual_module_kWh[name] = energy_density_kWh_m2 * area

    labels = list(panels.keys())
    vals_module  = [annual_module_kWh[k] for k in labels]
    vals_density = [annual_density_kWh_m2[k] for k in labels]

    fig4 = plt.figure(figsize=(10, 8))
    ax4a = fig4.add_subplot(2, 1, 1)
    bars1 = ax4a.bar(labels, vals_module)
    ax4a.set_ylabel("Énergie annuelle (kWh/an)")
    ax4a.set_title("(a) Module unique)")
    for b, v in zip(bars1, vals_module):
        ax4a.text(b.get_x() + b.get_width()/2, b.get_height()*1.01, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    ax4b = fig4.add_subplot(2, 1, 2)
    bars2 = ax4b.bar(labels, vals_density)
    ax4b.set_ylabel("Énergie annuelle (kWh/m²·an)")
    ax4b.set_title("(b) Par unité de surface")
    ax4b.set_xlabel("Panneau")
    for b, v in zip(bars2, vals_density):
        ax4b.text(b.get_x() + b.get_width()/2, b.get_height()*1.01, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    st.pyplot(fig4, clear_figure=True)

    st.subheader("Résumé annuel estimé")
    st.dataframe(
        pd.DataFrame({
            "Panneau": labels,
            "Énergie module (kWh/an)": [round(x, 1) for x in vals_module],
            "Énergie surfacique (kWh/m²·an)": [round(x, 1) for x in vals_density],
        }),
        use_container_width=True
    )

# # ---------------- TAB 4 : Équipe ----------------
# with tab4:
#     st.subheader("👥 Équipe du projet")
#     st.caption("Cliquez pour ouvrir les profils LinkedIn.")

#     TEAM = [
#         {"name": "Mahmoud Abdi",      "linkedin": "https://www.linkedin.com/in/mahamoud-abdi-abdillahi/"},
#         {"name": "Moustapha Ali",     "linkedin": "https://www.linkedin.com/in/moustaphalifarah/"},
#         {"name": "Aboubaker Mohamed", "linkedin": "https://www.linkedin.com/in/aboubaker-mohamed-abdi-010114273/"},
#     ]

#     if not TEAM:
#         st.info("Aucun membre défini. Renseigne la liste TEAM ci-dessus.")
#     else:
#         per_row = 3
#         for i in range(0, len(TEAM), per_row):
#             row = TEAM[i:i+per_row]
#             cols = st.columns(len(row))
#             for col, member in zip(cols, row):
#                 with col:
#                     st.markdown(f"**{member.get('name','(Sans nom)')}**")
#                     url = (member.get("linkedin") or "").strip()
#                     if url:
#                         st.link_button("LinkedIn", url)
#                     else:
#                         st.caption("Lien LinkedIn non fourni")

# ---------------- TAB 4 : Équipe ----------------
# # ---------------- TAB 4 : Équipe ----------------
# import os, base64, mimetypes

# with tab4:
#     st.subheader("👥 Équipe du projet")
#     st.caption("Cliquez pour ouvrir les profils LinkedIn.")

#     # Helper: local file -> data: URI (so <img src="..."> works in st.markdown)
#     def _as_data_uri(path: str) -> str | None:
#         path = (path or "").strip()
#         if not path or not os.path.exists(path):
#             return None
#         mime = mimetypes.guess_type(path)[0] or "image/jpeg"
#         with open(path, "rb") as f:
#             b64 = base64.b64encode(f.read()).decode("ascii")
#         return f"data:{mime};base64,{b64}"

#     def _resolve_avatar(src: str | None) -> str | None:
#         """Accepts data: URIs, http(s) URLs, or local repo paths."""
#         if not src:
#             return None
#         s = src.strip()
#         if s.startswith(("data:image/", "http://", "https://")):
#             return s
#         return _as_data_uri(s)

#     # ✅ Use repo-relative paths (works now via data URIs)
#     TEAM = [
#         {
#             "name": "Mahmoud Abdi",
#             "linkedin": "https://www.linkedin.com/in/mahamoud-abdi-abdillahi/",
#             "avatar": "photo/moud.jpg",
#         },
#         {
#             "name": "Moustapha Ali",
#             "linkedin": "https://www.linkedin.com/in/moustaphalifarah/",
#             "avatar": "photo/mous.jpg",
#         },
#         {
#             "name": "Aboubaker Mohamed",
#             "linkedin": "https://www.linkedin.com/in/aboubaker-mohamed-abdi-010114273/",
#             # ⚠️ change to a real file if you add it; e.g. "photo/aboubaker.jpg"
#             "avatar": "photo/j.jpg",
#         },
#     ]

#     def linkedin_button(name: str, url: str, avatar: str | None):
#         img_src = _resolve_avatar(avatar)
#         if not img_src:
#             # graceful fallback
#             img_src = "https://static.streamlit.io/examples/dice.jpg"

#         html = f"""
#         <a href="{url}" target="_blank" style="text-decoration:none;">
#           <div style="
#             display:inline-flex; align-items:center; gap:10px;
#             padding:8px 12px; border:1px solid #ddd; border-radius:10px;">
#             <img src="{img_src}" alt="{name}" 
#                  style="width:26px;height:26px;border-radius:50%;object-fit:cover;">
#             <span style="font-weight:600;">LinkedIn</span>
#           </div>
#         </a>
#         """
#         st.markdown(html, unsafe_allow_html=True)

#     if not TEAM:
#         st.info("Aucun membre défini. Renseigne la liste TEAM ci-dessus.")
#     else:
#         per_row = 3
#         for i in range(0, len(TEAM), per_row):
#             row = TEAM[i:i+per_row]
#             cols = st.columns(len(row))
#             for col, m in zip(cols, row):
#                 with col:
#                     st.markdown(f"**{m.get('name','(Sans nom)')}**")
#                     url = (m.get("linkedin") or "").strip()
#                     avatar = (m.get("avatar") or "").strip()
#                     if url:
#                         linkedin_button(m.get("name",""), url, avatar)
#                     else:
#                         st.caption("Lien LinkedIn non fourni")

# ---------------- TAB 4 : Équipe (style cartes) ----------------
import os, base64, mimetypes

with tab4:
    st.subheader("👨‍💼💡 Meet the Team")

    # ---------- helpers ----------
    def _as_data_uri(path: str) -> str | None:
        """Fichier local -> data:URI (pour <img src="..."> dans st.markdown)."""
        if not path or not os.path.exists(path):
            return None
        mime = mimetypes.guess_type(path)[0] or "image/jpeg"
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime};base64,{b64}"

    def _resolve_img(src: str | None) -> str:
        """Accepte data:, http(s) ou chemin local. Fallback propre."""
        if not src:
            return "https://static.streamlit.io/examples/dice.jpg"
        s = src.strip()
        if s.startswith(("data:image/", "http://", "https://")):
            return s
        return _as_data_uri(s) or "https://static.streamlit.io/examples/dice.jpg"

    # petit logo LinkedIn en SVG encodé (pas d'appel externe)
    _linkedin_svg = """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 448 512">
    <rect width="448" height="448" rx="48" ry="48" fill="#0A66C2"/>
    <path fill="#fff" d="M100.3 448H7V148.9h93.3V448zM53.7 108.1C24 108.1 0 84 0 54.3S24 0.6 53.7 0.6s53.7 24.1 53.7 53.7-24.1 53.8-53.7 53.8zM447.9 448h-93.1V302.4c0-34.7-0.7-79.3-48.3-79.3-48.3 0-55.7 37.7-55.7 76.6V448h-93.1V148.9H248v40.8h1.3c13.9-26.4 47.9-54.3 98.6-54.3 105.4 0 124.9 69.4 124.9 159.6V448z"/>
    </svg>
    """.strip()
    _linkedin_data_uri = "data:image/svg+xml;base64," + base64.b64encode(_linkedin_svg.encode("utf-8")).decode("ascii")

    # ---------- données équipe ----------
    TEAM = [
        {
            "name": "Mahmoud Abdi",
            "linkedin": "https://www.linkedin.com/in/mahamoud-abdi-abdillahi/",
            "avatar": "photo/moud.jpg",
        },
        {
            "name": "Moustapha Ali",
            "linkedin": "https://www.linkedin.com/in/moustaphalifarah/",
            "avatar": "photo/mous.jpg",
        },
        {
            "name": "Aboubaker Mohamed",
            "linkedin": "https://www.linkedin.com/in/aboubaker-mohamed-abdi-010114273/",
            "avatar": "photo/j.jpg",  # ⬅️ mets le vrai fichier si différent
        },
        # Exemple d’un 4e membre pour avoir une ligne complète :
        # {"name": "Muktar Abdinasir", "linkedin": "https://...", "avatar": "photo/muktar.jpg"},
    ]

    def member_card(name: str, avatar: str | None, linkedin: str | None):
        img = _resolve_img(avatar)
        ln = (linkedin or "").strip()

        html = f"""
        <div style="text-align:center;margin-bottom:28px;">
          <img src="{img}" alt="{name}" 
               style="width:100%;max-width:260px;aspect-ratio:1/1;
                      object-fit:cover;border-radius:16px;
                      box-shadow:0 4px 16px rgba(0,0,0,.08);" />
          <div style="margin-top:10px;font-weight:600;color:#333;">{name}</div>
          {f'<a href="{ln}" target="_blank" aria-label="LinkedIn">'
             f'<img src="{_linkedin_data_uri}" style="width:34px;height:34px;margin-top:10px;"/></a>'
            if ln else '<div style="height:34px;margin-top:10px;opacity:.5;">(LinkedIn non fourni)</div>'}
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

    # ---------- rendu en grille ----------
    if not TEAM:
        st.info("Aucun membre défini. Renseigne la liste TEAM ci-dessus.")
    else:
        per_row = 4  # comme ta capture
        for i in range(0, len(TEAM), per_row):
            row = TEAM[i:i+per_row]
            cols = st.columns(len(row), gap="large")
            for col, m in zip(cols, row):
                with col:
                    member_card(m.get("name","(Sans nom)"),
                                m.get("avatar"),
                                m.get("linkedin"))

