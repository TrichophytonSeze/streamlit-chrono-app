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

if uploaded_file is not None:

    # --- LECTURE ROBUSTE DU FICHIER AVEC PANDAS ---
    try:
        if uploaded_file.name.endswith('.csv'):
            try:
                df = pd.read_csv(uploaded_file, sep=';')
            except:
                df = pd.read_csv(uploaded_file, sep=',')
        else:
            df = pd.read_excel(uploaded_file)

        # --- NETTOYAGE DES NOMS DE COLONNES ---
        df.columns = df.columns.str.lower().str.replace('[^a-z0-9_]', '', regex=True).str.replace(' ', '_', regex=True)

        # --- VÉRIFICATION ET CONVERSION DES DONNÉES ---
        required_cols = ['dci', 'dose', 'frequence', 'date_debut', 'date_fin']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Erreur: Colonnes manquantes. Veuillez utiliser : {', '.join(required_cols)}")
            st.stop()

        df['date_debut'] = pd.to_datetime(df['date_debut'], errors='coerce')
        df['date_fin'] = pd.to_datetime(df['date_fin'], errors='coerce')
        df['dose'] = pd.to_numeric(df['dose'], errors='coerce')

        df.dropna(subset=['date_debut', 'date_fin', 'dose'], inplace=True)

        if df.empty:
            st.warning("Le fichier ne contient aucune donnée de traitement valide.")
            st.stop()

        # --- ÉTAPE CLÉ : NORMALISATION DE LA DOSE PAR GROUPE DE DCI ---
        df['Min_Dose_DCI'] = df.groupby('dci')['dose'].transform('min')
        df['Max_Dose_DCI'] = df.groupby('dci')['dose'].transform('max')

        df['Dose_Range'] = df['Max_Dose_DCI'] - df['Min_Dose_DCI']

        # Correction de l'opacité : Assurer un MINIMUM DE VISIBILITÉ (0.3)
        # La Dose la plus faible de chaque DCI sera 0.3 (clair), la Dose la plus élevée sera 1.0 (foncé).
        df['Dose_Normalized'] = df.apply(
            lambda row: 1.0 if row['Dose_Range'] == 0 else (
                        0.3 + 0.7 * (row['dose'] - row['Min_Dose_DCI']) / row['Dose_Range']),
            axis=1
        )

        df['date_fin_plot'] = df['date_fin'] + pd.Timedelta(days=1)

        # --- CRÉATION DES COLONNES DE DATES FORMATTÉES ---
        df["date_debut_str"] = df["date_debut"].dt.strftime("%d %b %Y")
        df["date_fin_str"] = df["date_fin"].dt.strftime("%d %b %Y")

        # --- CRÉATION DU TEXTE AU SURVOL (Hover_Text) ---
        df['Hover_Text'] = (
                "<b>Dose:</b> " + df['dose'].astype(str) + "mg<br>" +
                "<b>Fréquence:</b> " + df['frequence'].astype(str)
        )

        # --- CRÉATION DU DIAGRAMME DE GANTT (Plotly.express) ---
        fig = px.timeline(
            df,
            x_start="date_debut",
            x_end="date_fin_plot",
            y="dci",
            color="dci",
            color_discrete_sequence=px.colors.qualitative.Dark24,
            hover_name="dci",
            hover_data={"dci": False},  # on garde hover_data minimal
            title="Chronologie des Traitements (Luminosité Relative)"
        )

        # Nombre de lignes (DCI uniques)
        N = df['dci'].nunique()

        # Hauteur souhaitée par ligne (en pixels)
        # 1cm ≈ 37.8 pixels
        height_per_line_px = 75  # par exemple 2 cm ≈ 75 pixels

        # Hauteur totale
        fig_height = N * height_per_line_px

        # Mettre à jour le layout
        fig.update_layout(
            height=fig_height,
            width= 1200,
            margin=dict(t=120, b=50, l=50, r=50),  # t = marge top
            showlegend=False,
            # S'assurer que le titre est centré pour un meilleur look
            title=dict(x=0.5, xanchor='center')
        )

        # --- AJOUTER ANNOTATIONS DE DOSE AU MILIEU DES INTERVALLES ---
        for _, row in df.iterrows():
            x_mid = row['date_debut'] + (row['date_fin_plot'] - row['date_debut']) / 2
            fig.add_annotation(
                x=x_mid,
                y=row['dci'],
                text=str(int(row['dose'])) + " mg",
                showarrow=False,
                font=dict(color="black", size=10),
                align="center",
                yshift=0
            )

        # --- APPLIQUER LA LUMINOSITÉ (OPACITÉ) ET FIXER L'HOVER TEMPLATE ---

        for trace in fig.data:
            dci_name = trace.name
            trace_df = df[df['dci'] == dci_name]

            # Application de l'opacité (luminosité)
            trace.marker.opacity = trace_df['Dose_Normalized'].tolist()

            trace.width = 0.8 # valeur entre 0 et 1 ; 1 = barre occupe toute la hauteur de la catégorie

            # Ajout du texte de survol
            trace.customdata = list(
                zip(
                    trace_df['Hover_Text'],
                    trace_df['date_debut_str'],
                    trace_df['date_fin_str']
                )
            )

            # CORRECTION DU HOVER TEMPLATE : suppression des formats exotiques et des parenthèses
            # Les dates sont gérées par les variables xstart et xend
            trace.hovertemplate = (
                    "<b>%{y}</b><br>" +
                    "Période: %{customdata[1]} à %{customdata[2]}<br>" +
                    "%{customdata[0]}<extra></extra>"
            )

        # Mise en forme du Layout pour le Gantt
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(side="top",
                         showgrid=True,  # Affiche la grille
                         gridwidth=1,  # Épaisseur de la ligne de grille
                         gridcolor='rgba(0,0,0,0.1)'  # Couleur gris très clair (10% d'opacité)
                         )

        st.info(
            "La teinte de la barre indique le **DCI**, et sa luminosité (**clair/foncé**) indique la **Dose** (relative à la dose MIN/MAX de ce même DCI).")

        # Affichage du graphique
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

else:
    st.sidebar.markdown("""
        **Noms de colonnes requis (en minuscule et sans accent) :**
        - dci
        - dose
        - frequence
        - date_debut
        - date_fin
    """)
    st.info("Veuillez charger un fichier pour commencer l'analyse.")