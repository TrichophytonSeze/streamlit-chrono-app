import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date
from dateutil.relativedelta import relativedelta

# --- Configuration page ---
st.set_page_config(layout="wide")
st.title("Chronologie des Traitements (Python/Streamlit)")

# --- Message info utilisateur ---
st.info(
    """La couleur indique la prise de la dci (répétée si nécessaire), "
    la luminosité (**clair/foncé**) indique la **dose** relative à chaque dci.  
    Possibilité d'exporter (icône **appareil photo** en haut à droite du graphique).  
    Infos (posologie, dates) disponibles au **survol** de la case."""
)

# --- Sidebar ---
uploaded_file = st.sidebar.file_uploader("Choisir fichier Excel/CSV", type=["xlsx", "csv"])

st.sidebar.markdown("---")
st.sidebar.markdown("### Filtre calendrier (par défaut)")

# Calendrier par défaut : 3 derniers mois
today = pd.Timestamp.today().normalize()
default_end = today
default_start = default_end - relativedelta(months=3)

use_calendar = st.sidebar.checkbox("Utiliser le calendrier", value=True)
date_start_input = st.sidebar.date_input("Date de début", value=default_start)
date_end_input = st.sidebar.date_input("Date de fin", value=default_end)

st.sidebar.markdown("---")
st.sidebar.markdown("### Filtre relatif")
number = st.sidebar.number_input("Nombre", min_value=1, value=3, step=1)
unit = st.sidebar.selectbox("Unité", ["jour", "semaine", "mois", "année"])

st.sidebar.markdown("---")
st.sidebar.markdown("### Options d'affichage")
height_per_line_px = st.sidebar.slider("Hauteur par barre (px)", 20, 150, 75)
show_case_text = st.sidebar.checkbox("Afficher le texte dans les cases", value=True)
case_text_font_size = st.sidebar.slider("Taille texte dans les cases", 8, 20, 10)
yaxis_font_size = st.sidebar.slider("Taille texte DCI (axe Y)", 8, 20, 12)

if uploaded_file is not None:
    try:
        # --- Lecture fichier ---
        if uploaded_file.name.endswith(".csv"):
            try:
                df = pd.read_csv(uploaded_file, sep=';')
            except:
                df = pd.read_csv(uploaded_file, sep=',')
        else:
            df = pd.read_excel(uploaded_file)

        # Nettoyage colonnes
        df.columns = df.columns.str.lower().str.replace('[^a-z0-9_]', '', regex=True).str.replace(' ', '_', regex=True)
        required_cols = ['dci', 'dose', 'frequence', 'date_debut', 'date_fin', 'unite']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Colonnes manquantes. Requises : {', '.join(required_cols)}")
            st.stop()

        # Conversion types
        df['date_debut'] = pd.to_datetime(df['date_debut'], errors='coerce')
        df['date_fin'] = pd.to_datetime(df['date_fin'], errors='coerce')
        df['dose'] = pd.to_numeric(df['dose'], errors='coerce')
        df.dropna(subset=['date_debut'], inplace=True)

        # --- Calcul dates filtrées ---
        if use_calendar:
            date_start_ts = pd.Timestamp(date_start_input)
            date_end_ts = pd.Timestamp(date_end_input)
        else:
            date_end_ts = today  # TOUJOURS aujourd'hui pour le filtre relatif
            if unit == "jour":
                date_start_ts = today - pd.Timedelta(days=number)
            elif unit == "semaine":
                date_start_ts = today - pd.Timedelta(weeks=number)
            elif unit == "mois":
                date_start_ts = today - relativedelta(months=number)
            elif unit == "année":
                date_start_ts = today - relativedelta(years=number)

        # Inclure le dernier jour de la période
        date_end_ts_inclusive = date_end_ts + pd.Timedelta(days=1)

        # --- Filtrer DCI présentes dans la période ---
        df_filtered = df[(df['date_fin'] >= date_start_ts) & (df['date_debut'] <= date_end_ts)].copy()
        if df_filtered.empty:
            st.warning("Aucune donnée dans la période sélectionnée.")
            st.stop()

        # Tronquer dates pour affichage
        df_filtered['date_debut_plot'] = df_filtered['date_debut'].apply(lambda d: max(d, date_start_ts))

        # Ajouter 1 jour pour inclure le dernier jour complet
        df_filtered['date_fin_plot'] = df_filtered['date_fin'].apply(lambda d: min(d + pd.Timedelta(days=1), date_end_ts + pd.Timedelta(days=1)))

        # --- Normalisation dose pour opacité ---
        df_filtered['Min_Dose_DCI'] = df_filtered.groupby('dci')['dose'].transform('min')
        df_filtered['Max_Dose_DCI'] = df_filtered.groupby('dci')['dose'].transform('max')
        df_filtered['Dose_Range'] = df_filtered['Max_Dose_DCI'] - df_filtered['Min_Dose_DCI']
        df_filtered['Dose_Normalized'] = df_filtered.apply(
            lambda row: 1.0 if pd.isna(row['Dose_Range']) or row['Dose_Range'] == 0 else (
                0.4 + 0.6 * (row['dose'] - row['Min_Dose_DCI']) / row['Dose_Range']
            ),
            axis=1
        )

        # --- Couleurs ---
        base_colors = ["#FFD966", "#F4B183", "#FF61C3", "#93C5FD", "#6FCF97"]
        unique_dci = df_filtered['dci'].unique()
        color_map = {dci: base_colors[i % len(base_colors)] for i, dci in enumerate(unique_dci)}
        df_filtered['color'] = df_filtered['dci'].map(color_map)

        # --- Posologie hover complète ---
        df_filtered['Posologie'] = df_filtered.apply(
            lambda row: (
                f"{int(row['dose']) if pd.notna(row['dose']) and row['dose'] == int(row['dose']) else row['dose']} "
                f"{row['unite'] if pd.notna(row['unite']) else ''} "
                f"{row['frequence'] if pd.notna(row['frequence']) else ''}"
            ).strip(),
            axis=1
        )

        # --- Graphique ---
        fig = px.timeline(
            df_filtered,
            x_start="date_debut_plot",
            x_end="date_fin_plot",
            y="dci",
            color="dci",
            color_discrete_map=color_map,
            hover_name="dci",
            title="Chronologie des traitements"
        )

        # Hauteur par ligne
        N = df_filtered['dci'].nunique()
        fig_height = N * height_per_line_px

        # Layout unique
        fig.update_layout(
            height=fig_height,
            width=1200,
            margin=dict(t=120, b=50, l=50, r=50),
            showlegend=False,
            title=dict(x=0.5, xanchor='center'),
            yaxis=dict(tickfont=dict(size=yaxis_font_size))
        )

        fig.update_xaxes(
            range=[date_start_ts, date_end_ts],
            tickformat="%d %b %Y",
            side="top"
        )

        # --- Annotations texte dans les cases ---
        if show_case_text:
            for _, row in df_filtered.iterrows():
                # x_mid recalculé en fonction de la période affichée
                x_mid = row['date_debut_plot'] + (row['date_fin_plot'] - row['date_debut_plot']) / 2

                dose_val = row.get('dose')
                dose_text = "" if pd.isna(dose_val) else (
                    str(int(dose_val)) if dose_val == int(dose_val) else str(dose_val))
                unite_val = "" if pd.isna(row.get('unite')) else str(row['unite'])
                freq_val = "" if pd.isna(row.get('frequence')) else str(row['frequence'])
                text = f"{dose_text} {unite_val} {freq_val}".strip()

                # Largeur de la barre calculée sur la période affichée
                bar_width_days = (row['date_fin_plot'] - row['date_debut_plot']).days + 1
                bar_width_px = bar_width_days * 10  # estimation simple

                if text and bar_width_px >= len(text) * case_text_font_size:
                    fig.add_annotation(
                        x=x_mid,
                        y=row['dci'],
                        text=text,
                        showarrow=False,
                        font=dict(size=case_text_font_size, color="black"),
                        align="center"
                    )

        # --- Hover complet ---
        for trace in fig.data:
            dci_name = trace.name
            trace_df = df_filtered[df_filtered['dci'] == dci_name]
            trace.marker.opacity = trace_df['Dose_Normalized'].tolist()
            trace.width = 0.8
            trace.customdata = list(
                zip(
                    trace_df['Posologie'],
                    trace_df['date_debut'].dt.strftime("%d %b %Y"),
                    trace_df['date_fin'].dt.strftime("%d %b %Y")
                )
            )
            trace.hovertemplate = (
                "<b>%{y}</b><br>"
                "Période: %{customdata[1]} à %{customdata[2]}<br>"
                "Posologie: %{customdata[0]}<extra></extra>"
            )

        # Calculer la durée totale en jours
        total_days = (date_end_ts - date_start_ts).days

        # Déterminer la fréquence selon la durée totale
        if total_days <= 14:
            freq = 'D'  # quotidien
        elif total_days <= 120:
            freq = 'W'  # hebdomadaire
        elif total_days <= 730:  # ~2 ans
            freq = 'M'  # mensuel
        else:
            freq = '3M'  # tous les 3 mois

        # Générer les dates des ticks
        tick_dates = pd.date_range(start=date_start_ts, end=date_end_ts, freq=freq)

        # Mettre à jour l'axe X
        fig.update_xaxes(
            tickvals=tick_dates,
            ticktext=[d.strftime("%d %b") for d in tick_dates],
            side="top",
            showgrid=False
        )

        # Ajouter des lignes verticales légères pour chaque tick
        for td in tick_dates:
            fig.add_shape(
                type="line",
                x0=td, x1=td,
                y0=-0.5, y1=len(df_filtered['dci'].unique()) - 0.5,
                line=dict(color="lightgrey", width=1),
                layer="below"
            )

        # Choix du format avec clé unique
        export_format = st.selectbox(
            "Choisir le format d'export :",
            ["SVG", "PNG"],
            key="export_format_selectbox"
        )

        # Affichage du graphique Plotly avec barre d'outils
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar": True,
                "modeBarButtonsToAdd": ["toImage"],
                "toImageButtonOptions": {
                    "format": export_format.lower(),
                    "filename": "timeline",
                    "height": fig_height,
                    "width": 1200,
                    "scale": 2
                }
            }
        )

    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier : {e}")
