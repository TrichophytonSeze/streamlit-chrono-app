import streamlit as st
import pandas as pd
import plotly.express as px

# Configuration de la page
st.set_page_config(layout="wide")

st.title("Chronologie des Traitements (Python/Streamlit)")

# Zone de téléchargement de fichier
uploaded_file = st.sidebar.file_uploader(
    "1. Choisir le fichier Excel (.xlsx) ou CSV (.csv)",
    type=["xlsx", "csv"]
)

if uploaded_file is not None:

    # --- LECTURE ROBUSTE DU FICHIER AVEC PANDAS ---
    try:
        # Tente de lire le fichier en fonction de son extension/mime type
        if uploaded_file.name.endswith('.csv'):
            # Lecture CSV : essaie le point-virgule (souvent le cas en France), sinon la virgule
            try:
                df = pd.read_csv(uploaded_file, sep=';')
            except:
                df = pd.read_csv(uploaded_file, sep=',')
        else:
            # Lecture Excel (openpyxl est utilisé en arrière-plan)
            df = pd.read_excel(uploaded_file)

        # --- NETTOYAGE DES NOMS DE COLONNES (Simplification) ---
        # Comme nous l'avons convenu : tout en minuscule et sans accent
        df.columns = df.columns.str.lower().str.replace('[^a-z0-9_]', '', regex=True).str.replace(' ', '_')

        # --- VÉRIFICATION ET CONVERSION DES DONNÉES ---
        required_cols = ['dci', 'dose', 'frequence', 'date_debut', 'date_fin']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Erreur: Colonnes manquantes. Veuillez utiliser : {', '.join(required_cols)}")
            st.stop()

        # Conversion des dates et des doses
        df['date_debut'] = pd.to_datetime(df['date_debut'], errors='coerce')
        df['date_fin'] = pd.to_datetime(df['date_fin'], errors='coerce')
        df['dose'] = pd.to_numeric(df['dose'], errors='coerce')

        # Filtrer les lignes non valides
        df.dropna(subset=['date_debut', 'date_fin', 'dose'], inplace=True)

        if df.empty:
            st.warning("Le fichier ne contient aucune donnée de traitement valide.")
            st.stop()

        # --- CRÉATION DU DIAGRAMME DE GANTT (Plotly.express) ---

        # Créer le champ de texte au survol
        df['Hover_Text'] = (
                "<b>" + df['dci'].astype(str) + "</b> (" +
                df['dose'].astype(str) + "mg, " +
                df['frequence'].astype(str) + ")"
        )

        # Création du Gantt (Barres Chronologiques)
        fig = px.timeline(
            df,
            x_start="date_debut",
            x_end="date_fin",
            y="dci",
            color="dose",  # Couleur par Dose
            color_continuous_scale=px.colors.sequential.Viridis,  # Palette de couleur contrastée
            hover_name="dci",
            hover_data={"dose": True, "frequence": True, "date_debut": True, "date_fin": True, "dci": False},
            title="Chronologie des Traitements (Intensité de Dose)",
        )

        # Mise en forme du Layout pour le Gantt
        fig.update_yaxes(autorange="reversed")  # Affiche la première DCI en haut
        fig.update_xaxes(side="top")  # Place l'axe des dates en haut

        # Affichage du graphique
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier : {e}")

else:
    st.info("Veuillez charger un fichier pour commencer l'analyse.")