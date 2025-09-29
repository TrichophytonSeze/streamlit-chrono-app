import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date
from dateutil.relativedelta import relativedelta

# --- Configuration page ---
st.set_page_config(layout="wide")
st.title("Chronologie des Traitements (Python/Streamlit)")

st.info(
    """La couleur indique la prise de la dci (répétée si nécessaire),
    la luminosité (**clair/foncé**) indique la **dose** relative à chaque dci.  
    Possibilité d'exporter (icône **appareil photo** en haut à droite du graphique).  
    Infos (posologie, dates) disponibles au **survol** de la case."""
)

# --- Sidebar ---
uploaded_file = st.sidebar.file_uploader("Choisir fichier Excel/CSV", type=["xlsx", "csv"])

st.sidebar.markdown("---")
st.sidebar.markdown("### Calendrier (par défaut)")

today = pd.Timestamp.today().normalize()
default_end = today
default_start = default_end - relativedelta(months=3)

use_calendar = st.sidebar.checkbox("Utiliser le calendrier", value=True)
date_start_input = st.sidebar.date_input("Date de début", value=default_start)
date_end_input = st.sidebar.date_input("Date de fin", value=default_end)

st.sidebar.markdown("---")
st.sidebar.markdown("### Derniers X jours/semaines... ")
number = st.sidebar.number_input("Nombre", min_value=1, value=14, step=1)
unit = st.sidebar.selectbox("Unité", ["jour", "semaine", "mois", "année"])

st.sidebar.markdown("---")
st.sidebar.markdown("### Options d'affichage")
sort_option = st.sidebar.selectbox(
    "Ordre des dci",
    ["Alphabétique", "Chronologique (date début)"],
    index=1
)
height_per_line_px = st.sidebar.slider("Hauteur par barre (px)", 20, 150, 75)
show_case_text = st.sidebar.checkbox("Afficher la posologie (dci)", value=True)
show_ei_text = st.sidebar.checkbox("Afficher le texte dans les cases des EI", value=True)
case_text_font_size = st.sidebar.slider("Taille texte dans les cases", 8, 20, 10)
yaxis_font_size = st.sidebar.slider("Taille texte DCI (axe Y)", 8, 20, 12)

st.sidebar.markdown("---")
st.sidebar.markdown("### Marqueur de date personnalisé")
show_ref_date = st.sidebar.checkbox("Afficher cette date", value=False)
offset_days = st.sidebar.number_input("Décalage en jours", min_value=0, max_value=3650, value=0, step=1, help="Appliqué à la date de référence ci-dessous")


if uploaded_file is not None:
    try:
        # --- Lecture fichier ---
        if uploaded_file.name.endswith(".csv"):
            try:
                df = pd.read_csv(uploaded_file, sep=';')
            except Exception:
                df = pd.read_csv(uploaded_file, sep=',')
        else:
            df = pd.read_excel(uploaded_file)

        df.columns = df.columns.str.lower().str.replace('[^a-z0-9_]', '', regex=True).str.replace(' ', '_', regex=True)
        required_cols = ['dci', 'dose', 'frequence', 'date_debut', 'date_fin', 'unite', 'ei']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Colonnes manquantes. Requises : {', '.join(required_cols)}")
            st.stop()

        if "ei" not in df.columns:
            df["ei"] = "Non"

        df['date_debut'] = pd.to_datetime(df['date_debut'], errors='coerce')
        df['date_fin'] = pd.to_datetime(df['date_fin'], errors='coerce')
        df['dose'] = pd.to_numeric(df['dose'], errors='coerce')
        df.dropna(subset=['date_debut'], inplace=True)

        # --- Calcul dates filtrées ---
        if use_calendar:
            date_start_ts = pd.Timestamp(date_start_input)
            date_end_ts = pd.Timestamp(date_end_input)
        else:
            date_end_ts = today
            if unit == "jour":
                date_start_ts = today - pd.Timedelta(days=number)
            elif unit == "semaine":
                date_start_ts = today - pd.Timedelta(weeks=number)
            elif unit == "mois":
                date_start_ts = today - relativedelta(months=number)
            elif unit == "année":
                date_start_ts = today - relativedelta(years=number)

        fill_value = date_end_ts if use_calendar else today
        df['date_fin'] = df['date_fin'].fillna(fill_value)
        display_end_inclusive = date_end_ts + pd.Timedelta(days=1)

        df_filtered = df[(df['date_fin'] >= date_start_ts) & (df['date_debut'] <= date_end_ts)].copy()
        if df_filtered.empty:
            st.warning("Aucune donnée dans la période sélectionnée.")
            st.stop()

        df_filtered['date_debut_plot'] = df_filtered['date_debut'].where(df_filtered['date_debut'] >= date_start_ts, date_start_ts)
        df_filtered['date_fin_plot'] = (df_filtered['date_fin'] + pd.Timedelta(days=1)).where(
            df_filtered['date_fin'] + pd.Timedelta(days=1) <= display_end_inclusive,
            display_end_inclusive
        )

        # --- EI et tri DCI ---
        df_filtered['is_ei'] = df_filtered['ei'].fillna("").astype(str).str.strip().str.lower() == "oui"
        df_filtered['dose'] = df_filtered['dose'].fillna("")
        df_filtered['frequence'] = df_filtered['frequence'].fillna("")
        df_filtered['unite'] = df_filtered['unite'].fillna("")

        # EI et non-EI séparés
        ei_df = df_filtered[df_filtered['is_ei']].copy()
        non_ei_df = df_filtered[~df_filtered['is_ei']].copy()

        # Mettre la date par défaut avec date du début du premier EI (s'il existe)
        default_ref_date = ei_df['date_debut'].min().date() if not ei_df.empty else today.date()
        ref_date_input = st.sidebar.date_input(
            "Date de référence",
            value=default_ref_date,
            help="Date du début du premier EI si présent"
        )

        # Tri DCI non-EI
        if sort_option == "Chronologique (date début)":
            # Tri par date_debut puis date_fin
            sorted_df = non_ei_df.groupby('dci').agg(
                date_debut_min=('date_debut', 'min'),
                date_fin_max=('date_fin', 'max')
            ).sort_values(by=['date_debut_min', 'date_fin_max']).reset_index()
            dci_order = sorted_df['dci'].tolist()
        else:  # Alphabétique
            dci_order = sorted(non_ei_df['dci'].unique())

        ei_names = ei_df['dci'].unique().tolist()
        y_order = ["⚠️ " + ei for ei in ei_names] + dci_order
        df_filtered['y_position'] = df_filtered.apply(lambda r: "⚠️ " + r['dci'] if r['is_ei'] else r['dci'], axis=1)

        # S'assurer que 'dose' est un float pour éviter les erreurs de type après filtrage
        df_filtered['dose'] = pd.to_numeric(df_filtered['dose'], errors='coerce')

        # --- Normalisation dose pour DCI ---
        df_filtered['Min_Dose_DCI'] = df_filtered.groupby('dci')['dose'].transform('min')
        df_filtered['Max_Dose_DCI'] = df_filtered.groupby('dci')['dose'].transform('max')

        # On utilise .fillna(0) uniquement pour le calcul de la plage
        min_doses = df_filtered['Min_Dose_DCI'].fillna(0)
        max_doses = df_filtered['Max_Dose_DCI'].fillna(0)


        #df_filtered['Dose_Range'] = df_filtered['Max_Dose_DCI'] - df_filtered['Min_Dose_DCI']
        df_filtered['Dose_Range'] = max_doses - min_doses

        df_filtered['Dose_Normalized'] = df_filtered.apply(
            lambda row: 1.0 if pd.isna(row['Dose_Range']) or row['Dose_Range'] == 0 else 0.4 + 0.6 * (
                        row['dose'] - row['Min_Dose_DCI']) / row['Dose_Range'],
            axis=1
        )

        # --- Couleurs ---
        base_colors = ["#FFD966", "#F4B183", "#93C5FD", "#6FCF97", "#FF61C3", "#A0E7E5", "#C7A0FF"]
        meds = non_ei_df['dci'].unique()
        color_map = {dci: base_colors[i % len(base_colors)] for i, dci in enumerate(meds)}
        for ei in ei_names:
            color_map[ei] = "#f4a6c1"  # rose clair pour EI

        # --- Posologie texte ---
        # --- Posologie texte ---
        # S'assurer que 'dose' est un float pour éviter les erreurs de type après filtrage
        df_filtered['dose'] = pd.to_numeric(df_filtered['dose'], errors='coerce')

        df_filtered['Posologie'] = df_filtered.apply(
            lambda r: (
                # Si la dose est NaN (manquante), on affiche juste l'unité et la fréquence
                f"{r['unite']} {r['frequence']}".strip()
                if pd.isna(r['dose']) else
                # Si la dose est un entier (ex: 50.0), on affiche 50
                f"{int(r['dose'])} {r['unite']} {r['frequence']}".strip()
                if r['dose'].is_integer() else
                # Sinon, on affiche le nombre flottant (ex: 0.75)
                f"{r['dose']} {r['unite']} {r['frequence']}".strip()
            ),
            axis=1
        )

        # --- Graphique ---
        fig = px.timeline(
            df_filtered,
            x_start="date_debut_plot",
            x_end="date_fin_plot",
            y="y_position",
            color="dci",
            color_discrete_map=color_map,
            hover_name="dci",
            category_orders={"y_position": y_order},
            title="Chronologie des traitements et EI"
        )

        # --- Annotations texte dans les cases (posologie) ---
        if show_case_text:
            fig_width_px = fig.layout.width if (fig.layout and fig.layout.width) else 1200
            total_days = (date_end_ts - date_start_ts).days
            px_per_day = fig_width_px / total_days if total_days > 0 else fig_width_px

            for _, row in df_filtered.iterrows():
                # Si EI et que l'utilisateur ne veut pas le texte dans les cases, on skip
                if row['is_ei'] and not show_ei_text:
                    continue

                start = row['date_debut_plot']
                end = row['date_fin_plot']
                if pd.isna(start) or pd.isna(end) or end <= start:
                    continue

                # Milieu de la barre affichée
                x_mid = start + (end - start) / 2
                y_pos = row['y_position']

                # Texte posologie
                dose_val = row.get('dose')  # Contient la valeur float ou NaN

                # Utiliser la même logique pour gérer les NaN et les entiers/flottants
                if pd.isna(dose_val):
                    dose_text = ""
                elif dose_val.is_integer():
                    dose_text = str(int(dose_val))  # Affiche '50' au lieu de '50.0'
                else:
                    dose_text = str(dose_val)  # Affiche '0.75'

                unite_val = "" if pd.isna(row.get('unite')) else str(row['unite'])
                freq_val = "" if pd.isna(row.get('frequence')) else str(row['frequence'])
                text = f"{dose_text} {unite_val} {freq_val}".strip()

                # Largeur de la barre affichée en pixels
                bar_width_days = (end - start).days
                bar_width_px = max(0, bar_width_days) * px_per_day
                text_px_est = len(text) * case_text_font_size * 0.6 if text else 0

                # Affiche le texte seulement si ça rentre
                if text and bar_width_px >= text_px_est:
                    fig.add_annotation(
                        x=x_mid,
                        y=y_pos,
                        text=text,
                        showarrow=False,
                        font=dict(size=case_text_font_size, color="black"),
                        align="center",
                        xanchor="center",
                        yanchor="middle"
                    )

        N = df_filtered['y_position'].nunique()
        fig_height = max(300, N * height_per_line_px)
        fig.update_layout(height=fig_height, width=1200, margin=dict(t=120, b=50, l=50, r=50),
                          showlegend=False, yaxis_title="dci",
                          title=dict(x=0.5, xanchor='center'),
                          yaxis=dict(tickfont=dict(size=yaxis_font_size)))
        fig.update_xaxes(range=[date_start_ts, display_end_inclusive], tickformat="%d %b %Y", side="top")

        # --- Ligne séparation EI / DCI ---
        n_ei = len(ei_names)
        if n_ei > 0:
            y_line = N - n_ei - 0.5
            fig.add_shape(type="line", x0=date_start_ts, x1=display_end_inclusive, y0=y_line, y1=y_line,
                          line=dict(color="grey", width=1, dash="dot"), xref="x", yref="y", layer="above")

        # --- Barre verticale EI + symboles début/fin ---
        # --- Rectangle rose à partir du dernier EI uniquement ---
        # --- Barre verticale EI + rectangle rose derrière les DCI + symboles début/fin EI ---
        if not ei_names:
            pass
        else:
            # Période globale des EI
            ei_start = ei_df['date_debut_plot'].min()
            ei_end = ei_df['date_fin_plot'].max()

            # Limites Y pour englober uniquement les DCI (sous les EI)
            y0 = 0 - 0.5  # première ligne DCI (juste sous les EI)
            y1 = N - len(ei_names) - 0.5  # dernière ligne DCI

            # Rectangle rose derrière les DCI
            fig.add_shape(
                type="rect",
                x0=ei_start,
                x1=ei_end,
                y0=y0,
                y1=y1,
                fillcolor="rgba(244,166,193,0.2)",
                line=dict(width=0),
                layer="below",
                xref="x",
                yref="y"
            )

            # Symboles début/fin pour chaque EI
            for ei in ei_names:
                ei_rows = df_filtered[df_filtered['dci'] == ei]
                if ei_rows.empty:
                    continue
                ei_start_i = ei_rows['date_debut_plot'].min()
                ei_end_i = ei_rows['date_fin_plot'].max()
                start_date_str = ei_start_i.strftime("%d %b %Y")
                end_date_str = ei_end_i.strftime("%d %b %Y")

                # Récupérer le nom de l'EI
                ei_name_display = ei  # Le nom est directement dans la variable de la boucle
                # Créer la customdata spécifique pour les deux points (Début et Fin)
                custom_data_points = [
                    [ei_name_display, f"Début: {start_date_str}"],  # Data pour le point de début
                    [ei_name_display, f"Fin: {end_date_str}"]  # Data pour le point de fin
                ]

                fig.add_scatter(
                    x=[ei_start_i, ei_end_i],
                    y=[N - 0.5, N - 0.5],
                    mode="markers",
                    marker=dict(symbol="diamond", size=12, color="#f4a6c1"),
                    showlegend=False,
                    customdata=custom_data_points,
                    hovertemplate="<b>EI: %{customdata[0]}</b><br>%{customdata[1]}<extra></extra>"
                )

        # Barre verticale pour aider à voir les dci
        if show_ref_date:
            try:
                # Calcul de la date finale (date + offset)
                ref_date_ts = pd.Timestamp(ref_date_input) - pd.Timedelta(days=int(offset_days))
                # On trace une ligne verticale fine rouge couvrant tout l'axe Y (comportement identique aux EI)
                fig.add_shape(
                    type="line",
                    x0=ref_date_ts,
                    x1=ref_date_ts,
                    y0=-0.5,
                    y1=N - 0.5,
                    line=dict(color="#fc6b03", width=2, dash="dot"),
                    xref="x",
                    yref="y",
                    layer="above"
                )
                # Optionnel : petit marqueur en haut (décommenter si tu veux)
                # fig.add_scatter(x=[ref_date_ts], y=[N - 0.4], mode="markers", marker=dict(symbol="line-ns", size=8, color="red"), showlegend=False, hoverinfo="skip")
            except Exception as e:
                # En cas d'erreur (mauvaise conversion), on ignore proprement
                st.warning(f"Impossible de tracer la date de référence : {e}")

        # --- Annotations texte, hover et hachure EI ---
        for trace in fig.data:
            trace_df = df_filtered[df_filtered['dci'] == trace.name]
            if trace_df.empty: continue
            if trace.name in ei_names:
                trace.marker.opacity = 1.0
                trace.marker.color = "#f4a6c1"
                trace.marker.pattern = dict(shape="/")  # hachure EI
            else:
                try:
                    trace.marker.opacity = trace_df['Dose_Normalized'].tolist()
                except:
                    trace.marker.opacity = float(trace_df['Dose_Normalized'].mean())
            trace.width = 0.8
            trace.customdata = list(zip(
                trace_df['Posologie'],
                trace_df['date_debut'].dt.strftime("%d %b %Y"),
                trace_df['date_fin'].dt.strftime("%d %b %Y")
            ))
            trace.hovertemplate = "<b>%{y}</b><br>Période: <b>%{customdata[1]} - %{customdata[2]}</b><br>Posologie: <b>%{customdata[0]}</b><extra></extra>"

        # --- Ticks axe X ---
        total_days = (date_end_ts - date_start_ts).days
        if total_days <= 14:
            freq = 'D'
        elif total_days <= 120:
            freq = 'W'
        elif total_days <= 730:
            freq = 'M'
        else:
            freq = '3M'
        tick_dates = pd.date_range(start=date_start_ts, end=date_end_ts, freq=freq)
        fig.update_xaxes(tickvals=tick_dates, ticktext=[d.strftime("%d %b") for d in tick_dates], side="top", showgrid=False)
        for td in tick_dates:
            fig.add_shape(type="line", x0=td, x1=td, y0=-0.5, y1=N-0.5,
                          line=dict(color="lightgrey", width=1), layer="below", xref="x", yref="y")

        # --- Export / affichage ---
        export_format = st.selectbox("Choisir le format d'export :", ["SVG", "PNG"], key="export_format_selectbox")
        st.plotly_chart(fig, use_container_width=True, config={
            "displayModeBar": True,
            "modeBarButtonsToAdd": ["toImage"],
            "toImageButtonOptions": {"format": export_format.lower(), "filename": "timeline",
                                     "height": fig_height, "width": 1200, "scale": 2}
        })

    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier : {e}")
