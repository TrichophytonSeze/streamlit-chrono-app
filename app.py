import streamlit as st
import pandas as pd
import plotly.express as px
import io
import plotly.io as pio

# Configuration de la page
st.set_page_config(layout="wide")

st.title("Chronologie des Traitements (Python/Streamlit)")
st.markdown("La luminosité de la couleur est **relative à la Dose MIN/MAX de chaque molécule**.")

# Zone de téléchargement de fichier
uploaded_file = st.sidebar.file_uploader(
    "1. Choisir le fichier Excel (.xlsx) ou CSV (.csv)",
    type=["xlsx", "csv"]
)

# Instructions sidebar
st.sidebar.markdown("""
**Noms de colonnes requis (en minuscule et sans accent) :**
- dci
- dose
- frequence
- date_debut
- date_fin
- unite
""")

if uploaded_file is not None:

    try:
        # --- LECTURE DU FICHIER ---
        if uploaded_file.name.endswith('.csv'):
            try:
                df = pd.read_csv(uploaded_file, sep=';')
            except:
                df = pd.read_csv(uploaded_file, sep=',')
        else:
            df = pd.read_excel(uploaded_file)

        # --- NETTOYAGE DES COLONNES ---
        df.columns = df.columns.str.lower().str.replace('[^a-z0-9_]', '', regex=True).str.replace(' ', '_', regex=True)

        # --- VÉRIFICATION DES COLONNES ---
        required_cols = ['dci', 'dose', 'frequence', 'date_debut', 'date_fin', 'unite']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Erreur: Colonnes manquantes. Veuillez utiliser : {', '.join(required_cols)}")
            st.stop()

        # --- CONVERSION DES COLONNES ---
        df['date_debut'] = pd.to_datetime(df['date_debut'], errors='coerce')
        df['date_fin'] = pd.to_datetime(df['date_fin'], errors='coerce')
        df['dose'] = pd.to_numeric(df['dose'], errors='coerce')

        # Supprimer les lignes sans date_debut
        df.dropna(subset=['date_debut'], inplace=True)
        if df.empty:
            st.warning("Le fichier ne contient aucune donnée de traitement valide.")
            st.stop()

        # --- TRAITEMENTS EN COURS ---
        df['en_cours'] = df['date_fin'].isna()
        df['date_fin'] = df['date_fin'].fillna(pd.Timestamp.today().normalize())
        df['date_fin_plot'] = df['date_fin'] + pd.Timedelta(days=1)

        # --- TEXTE POUR HOVER ---
        df['date_debut_str'] = df['date_debut'].dt.strftime("%d %b %Y")
        df['date_fin_str'] = df['date_fin'].dt.strftime("%d %b %Y")
        df.loc[df['en_cours'], 'date_fin_str'] = "En cours"

        # Remplacer les valeurs manquantes par "?" pour dose et fréquence
        df['dose_str'] = df['dose'].fillna("?")
        df['frequence_str'] = df['frequence'].fillna("?")
        df['unite_str'] = df['unite'].fillna("")  # vide si non précisé

        # Concaténation : dose + unité + fréquence
        df['Posologie'] = (df['dose_str'].astype(str) + " " + df['unite_str'].astype(str) + " " + df['frequence_str'].astype(str)).str.strip()

        # --- NORMALISATION DE LA DOSE POUR OPACITÉ ---
        df['Min_Dose_DCI'] = df.groupby('dci')['dose'].transform('min')
        df['Max_Dose_DCI'] = df.groupby('dci')['dose'].transform('max')
        df['Dose_Range'] = df['Max_Dose_DCI'] - df['Min_Dose_DCI']
        df['Dose_Normalized'] = df.apply(
            lambda row: 1.0 if pd.isna(row['Dose_Range']) or row['Dose_Range'] == 0 else (
                0.4 + 0.6 * (row['dose'] - row['Min_Dose_DCI']) / row['Dose_Range']
            ),
            axis=1
        )

        # --- COULEURS FIXES ET CIRCULAIRES ---
        #base_colors = ["#F4B183", "#B48EAD", "#FF61C3", "#01C19F"]  # orange, violet, rose, turquoise
        base_colors = [
            "#FFD966",  # jaune clair
            "#F4B183",  # orange pastel
            "#FF61C3",  # rose vif mais doux
            "#93C5FD",  # bleu clair
            "#6FCF97"  # vert pastel
        ]
        unique_dci = df['dci'].unique()
        color_map = {dci: base_colors[i % len(base_colors)] for i, dci in enumerate(unique_dci)}
        df['color'] = df['dci'].map(color_map)

        # --- GRAPHIQUE GANTT ---
        fig = px.timeline(
            df,
            x_start="date_debut",
            x_end="date_fin_plot",
            y="dci",
            color='dci',  # couleur discrète
            color_discrete_map=color_map,
            hover_name="dci",
            hover_data={"dci": False},
            title="Chronologie des Traitements (Luminosité Relative)"
        )

        # Hauteur par ligne
        N = df['dci'].nunique()
        height_per_line_px = 75
        fig_height = N * height_per_line_px

        fig.update_layout(
            height=fig_height,
            width=1200,
            margin=dict(t=120, b=50, l=50, r=50),
            showlegend=False,
            title=dict(x=0.5, xanchor='center')
        )

        # --- ANNOTATIONS DE DOSE ---
        for _, row in df.iterrows():
            if pd.notna(row['dose']):
                x_mid = row['date_debut'] + (row['date_fin_plot'] - row['date_debut']) / 2
                unit = row['unite_str']
                fig.add_annotation(
                    x=x_mid,
                    y=row['dci'],
                    text=f"{row['dose_str']} {unit}".strip(),
                    showarrow=False,
                    font=dict(color="black", size=10),
                    align="center",
                    yshift=0
                )

        # --- APPLIQUER OPACITÉ ---
        for trace in fig.data:
            dci_name = trace.name
            trace_df = df[df['dci'] == dci_name]
            trace.marker.opacity = trace_df['Dose_Normalized'].tolist()
            trace.width = 0.8
            trace.customdata = list(
                zip(
                    trace_df['Posologie'],
                    trace_df['date_debut_str'],
                    trace_df['date_fin_str']
                )
            )
            trace.hovertemplate = (
                "<b>%{y}</b><br>" +
                "Période: %{customdata[1]} à %{customdata[2]}<br>" +
                "Posologie: %{customdata[0]}<extra></extra>"
            )

        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(
            side="top",
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.1)'
        )

        st.info(
            "La couleur indique le DCI (orange/violet/rose/turquoise, répétée si nécessaire), et la luminosité (**clair/foncé**) indique la **Dose** relative à chaque DCI."
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- BOUTON EXPORT PDF ---
        if st.button("Télécharger PDF"):
            pdf_buffer = io.BytesIO()
            pio.write_image(fig, pdf_buffer, format='pdf', width=1200, height=fig_height)
            pdf_buffer.seek(0)
            st.download_button(
                label="Télécharger le PDF",
                data=pdf_buffer,
                file_name="timeline.pdf",
                mime="application/pdf"
            )

    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier : {e}")
