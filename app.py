import streamlit as st
import pandas as pd
import plotly.express as px
import io
from datetime import date
from dateutil.relativedelta import relativedelta
import numpy as np
from io import BytesIO

# --- Configuration et Initialisation ---
st.set_page_config(layout="wide")
st.title("Chronologie des Traitements (Python/Streamlit)")

# Initialisation de l'état de session
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df_dci_ei' not in st.session_state:
    st.session_state.df_dci_ei = pd.DataFrame()
if 'df_lab' not in st.session_state:
    st.session_state.df_lab = pd.DataFrame()
if 'ref_date_input' not in st.session_state:
    st.session_state.ref_date_input = date.today()


# --- Fonctions de Navigation et Utilitaires ---

def go_to_visualisation():
    """Fonction de rappel pour passer à l'étape de visualisation."""
    if not st.session_state.df_dci_ei.empty:
        st.session_state.data_loaded = True
    else:
        st.error("Veuillez charger un fichier DCI/EI valide pour continuer.")


def go_to_upload():
    """Fonction de rappel pour revenir à l'étape de chargement."""
    st.session_state.data_loaded = False


# --- UTILS POUR FICHIERS DE DÉMONSTRATION ---

def get_dci_excel_test_file():
    """Crée un DataFrame de démonstration DCI/EI et le renvoie sous forme binaire."""
    data = {
        'DCI': ['DCI A', 'DCI A', 'DCI B', 'EI X', 'DCI C', 'DCI D'],
        'Dose': [50, 100, 20, np.nan, 75, 10],
        'Unite': ['mg', 'mg', 'mg', 'UI', 'mg', 'mg'],
        'Frequence': ['1/j', '1/j', '2/j', 'ponctuelle', '2/s', '1/j'],
        'Date_Debut': [
            pd.Timestamp(date.today()) - pd.Timedelta(days=60),
            pd.Timestamp(date.today()) - pd.Timedelta(days=30),
            pd.Timestamp(date.today()) - pd.Timedelta(days=15),
            pd.Timestamp(date.today()) - pd.Timedelta(days=40),
            pd.Timestamp(date.today()) - pd.Timedelta(days=5),
            pd.Timestamp(date.today()) - pd.Timedelta(days=50)
        ],
        'Date_Fin': [
            pd.Timestamp(date.today()) - pd.Timedelta(days=35),
            pd.Timestamp(date.today()),
            pd.NaT,
            pd.Timestamp(date.today()) - pd.Timedelta(days=38),
            pd.NaT,
            pd.Timestamp(date.today()) - pd.Timedelta(days=20)
        ],
        'EI': ['Non', 'Non', 'Non', 'Oui', 'Non', 'Non']
    }
    df_test = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_test.to_excel(writer, index=False, sheet_name='Données')
    return output.getvalue()


def get_lab_excel_test_file():
    """Crée un DataFrame de démonstration Labo (format croisé) et le renvoie sous forme binaire."""
    today_str = pd.Timestamp(date.today()).strftime("%d.%m.%Y")
    data = {
        'nom': ['Hémoglobine', 'Sodium', 'Créatinine'],
        'seuil': ['120 - 160 g/l', '135 - 145 mmol/l', '60 - 110 µmol/l'],
        today_str: [150, 140, 85],
        (pd.Timestamp(date.today()) - pd.Timedelta(days=7)).strftime("%d.%m.%Y"): [115, 133, 120],
        (pd.Timestamp(date.today()) - pd.Timedelta(days=14)).strftime("%d.%m.%Y"): [155, 142, 70]
    }
    df_test = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_test.to_excel(writer, index=False, sheet_name='Labo')
    return output.getvalue()


def pivot_lab_data(df_lab: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme le DataFrame Labo (format croisé) en format long pour Plotly Timeline.
    """
    if df_lab.empty:
        return pd.DataFrame()

    id_vars = ['nom', 'seuil']

    # Nettoyage des colonnes Labo
    df_lab.columns = df_lab.columns.str.lower().str.replace('[^a-z0-9_.]', '', regex=True)

    if not all(col in df_lab.columns for col in id_vars):
        return pd.DataFrame()

    value_vars = [col for col in df_lab.columns if col not in id_vars]
    if not value_vars:
        return pd.DataFrame()

    # Transformation du format "large" au format "long"
    df_long = df_lab.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name='date_mesure',
        value_name='valeur'
    ).dropna(subset=['valeur'])

    # Conversion des colonnes
    df_long['date_mesure'] = pd.to_datetime(df_long['date_mesure'], format='%d.%m.%Y', errors='coerce')
    df_long['valeur'] = pd.to_numeric(df_long['valeur'], errors='coerce')

    # Nettoyage final
    df_long.rename(columns={'nom': 'dci'}, inplace=True)
    df_long.dropna(subset=['date_mesure', 'valeur'], inplace=True)

    # Préparation pour la fusion
    df_long['dose'] = df_long['valeur']
    df_long['unite'] = df_long['seuil'].astype(str).str.extract(r'([a-z\/]+)$', expand=False).fillna('')
    df_long['frequence'] = "Mesure unique"
    df_long['ei'] = "Non"
    df_long['is_lab'] = True

    # Pour l'affichage "Timeline", on utilise la date_mesure comme date_debut et date_fin
    df_long['date_debut'] = df_long['date_mesure']
    df_long['date_fin'] = df_long['date_mesure'] + pd.Timedelta(days=0.1)  # Durée minimale pour être visible

    final_cols = ['dci', 'dose', 'frequence', 'date_debut', 'date_fin', 'unite', 'ei', 'is_lab']
    return df_long[final_cols]


# --- Fonction pour la page 1: Upload ---
def show_upload_page():
    st.header("1. Chargement des données DCI/EI et Labo 💾")

    # --- UPLOAD DCI/EI ---
    st.markdown("##### Fichier 1 : Traitements (DCI/EI) ")
    col_dci_upload, col_dci_example = st.columns([2, 1])

    with col_dci_upload:
        uploaded_file_dci = st.file_uploader("Choisir fichier Excel/CSV (DCI/EI)", type=["xlsx", "csv"],
                                             key="uploaded_dci_ei")

    with col_dci_example:
        dci_excel_data = get_dci_excel_test_file()
        if dci_excel_data is not None:
            st.download_button(
                label="Télécharger exemple DCI (.xlsx)",
                data=dci_excel_data,
                file_name="test_chrono_dci.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    st.markdown("---")

    # --- UPLOAD LABO ---
    st.markdown("##### Fichier 2 (Optionnel) : Résultats de Laboratoire ")
    col_lab_upload, col_lab_example = st.columns([2, 1])

    with col_lab_upload:
        uploaded_file_lab = st.file_uploader("Choisir fichier Excel/CSV (Labo - Format croisé)", type=["xlsx", "csv"],
                                             key="uploaded_lab")

    with col_lab_example:
        lab_excel_data = get_lab_excel_test_file()
        if lab_excel_data is not None:
            st.download_button(
                label="Télécharger exemple Labo (.xlsx)",
                data=lab_excel_data,
                file_name="test_chrono_labo.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    st.markdown("---")

    # --- Traitement des fichiers après l'upload ---

    # 1. Traitement DCI/EI
    if uploaded_file_dci:
        try:
            df_dci = pd.read_csv(uploaded_file_dci, sep=';') if uploaded_file_dci.name.endswith(".csv") and ';' in \
                                                                pd.read_csv(uploaded_file_dci, sep=';',
                                                                            nrows=1).columns[0] else pd.read_csv(
                uploaded_file_dci, sep=',') if uploaded_file_dci.name.endswith(".csv") else pd.read_excel(
                uploaded_file_dci)

            df_dci.columns = df_dci.columns.str.lower().str.replace('[^a-z0-9_]', '', regex=True).str.replace(' ', '_',
                                                                                                              regex=True)
            required_cols = ['dci', 'dose', 'frequence', 'date_debut', 'date_fin', 'unite', 'ei']

            if not all(col in df_dci.columns for col in required_cols):
                st.warning(f"Fichier DCI/EI invalide : Colonnes manquantes. Requises : {', '.join(required_cols)}")
                st.session_state.df_dci_ei = pd.DataFrame()
            else:
                df_dci['is_lab'] = False
                st.session_state.df_dci_ei = df_dci.copy()

                ei_df_for_default = df_dci[df_dci['ei'].fillna("").astype(str).str.strip().str.lower() == "oui"]
                ei_df_for_default['date_debut'] = pd.to_datetime(ei_df_for_default['date_debut'], errors='coerce')
                default_ref_date = ei_df_for_default[
                    'date_debut'].min().date() if not ei_df_for_default.empty and not pd.isna(
                    ei_df_for_default['date_debut'].min()) else date.today()
                st.session_state.ref_date_input = default_ref_date

                st.success(f"Fichier DCI/EI chargé : {len(df_dci)} lignes.")
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier DCI/EI : {e}")
            st.session_state.df_dci_ei = pd.DataFrame()
    else:
        st.session_state.df_dci_ei = pd.DataFrame()

    # 2. Traitement Labo
    if uploaded_file_lab:
        try:
            df_lab = pd.read_csv(uploaded_file_lab, sep=';') if uploaded_file_lab.name.endswith(".csv") and ';' in \
                                                                pd.read_csv(uploaded_file_lab, sep=';',
                                                                            nrows=1).columns[0] else pd.read_csv(
                uploaded_file_lab, sep=',') if uploaded_file_lab.name.endswith(".csv") else pd.read_excel(
                uploaded_file_lab)

            st.session_state.df_lab = pivot_lab_data(df_lab)

            if st.session_state.df_lab.empty:
                st.warning(
                    "Le fichier Labo est vide ou le format est incorrect. Vérifiez les colonnes 'nom', 'seuil' et les colonnes de date.")
            else:
                st.success(f"Fichier Labo chargé : {len(st.session_state.df_lab)} mesures.")

        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier Labo : {e}")
            st.session_state.df_lab = pd.DataFrame()
    else:
        st.session_state.df_lab = pd.DataFrame()

    # Bouton de validation
    if not st.session_state.df_dci_ei.empty:
        st.button("Démarrer la Chronologie ▶️", on_click=go_to_visualisation, type="primary")
    else:
        st.info("Chargez votre fichier DCI/EI pour continuer.")


# --- Fonction pour la page 2: Visualisation ---
def show_visualization_page():
    st.button("Retour au chargement ⏪", on_click=go_to_upload, key="back_button_viz")

    st.info(
        """La couleur indique la période d'administration de la dci/labo,
        la luminosité (**clair/foncé**) indique la **dose/valeur** relative à chaque dci/paramètre.  
        Les Effets Indésirables (EI) apparaissent en haut avec le symbole **⚠️**.  
        Infos (posologie, dates) disponibles au **survol** de la case.
        Possibilité d'exporter (icône **appareil photo** en haut à droite du graphique)."""
    )

    # --- 1. Fusionner les données DCI/EI et Labo ---
    df_dci = st.session_state.df_dci_ei.copy()
    df_lab = st.session_state.df_lab.copy()

    if df_dci.empty:
        st.error("Aucune donnée DCI/EI n'a été trouvée. Retour au chargement.")
        go_to_upload()
        return

    # Si les données Labo existent, les ajouter au DataFrame principal
    if not df_lab.empty:
        df = pd.concat([df_dci, df_lab], ignore_index=True)
    else:
        df = df_dci.copy()

    # --- 2. Sidebar pour les filtres d'affichage ---
    with st.sidebar:
        st.markdown("#### Options d'affichage")

        today = pd.Timestamp.today().normalize()
        default_end = today
        default_start = default_end - relativedelta(months=3)

        st.markdown("#### Période d'affichage")
        use_calendar = st.checkbox("Utiliser le calendrier", value=True)
        date_start_input = st.date_input("Date de début", value=default_start)
        date_end_input = st.date_input("Date de fin", value=default_end)

        st.markdown("#### Derniers X jours/semaines... ")
        number = st.number_input("Nombre", min_value=1, value=14, step=1)
        unit = st.selectbox("Unité", ["jour", "semaine", "mois", "année"])

        st.markdown("---")
        sort_option = st.selectbox(
            "Ordre des lignes",
            #["Alphabétique", "Chronologique (date début)"],
            ["Chronologique (date début)"],
            index=0
        )
        height_per_line_px = st.slider("Hauteur par barre (px)", 20, 150, 75)
        show_case_text = st.checkbox("Afficher la posologie/valeur", value=True)
        show_ei_text = st.checkbox("Afficher le texte dans les cases des EI", value=True)
        case_text_font_size = st.slider("Taille texte dans les cases", 8, 20, 10)
        yaxis_font_size = st.slider("Taille texte DCI/Labo (axe Y)", 8, 20, 12)
        xaxis_tick_font_size = st.slider("Taille police dates (axe X)", 8, 20, 10)

        st.markdown("---")
        st.markdown("#### Marqueur de date personnalisé")

        ref_date_input = st.date_input(
            "Date de référence",
            value=st.session_state.ref_date_input,
            key="ref_date_viz_input",
            help="Date du début du premier EI si présent"
        )
        show_ref_date = st.checkbox("Afficher cette date", value=False)
        offset_days = st.number_input("Décalage en jours", min_value=0, max_value=3650, value=0, step=1,
                                      help="Appliqué à la date de référence ci-dessous")

    # --- 3. Traitement et Filtration ---
    try:
        df['date_debut'] = pd.to_datetime(df['date_debut'], errors='coerce')
        df['date_fin'] = pd.to_datetime(df['date_fin'], errors='coerce')
        df['dose'] = pd.to_numeric(df['dose'], errors='coerce')
        df.dropna(subset=['date_debut'], inplace=True)

        # ... (Calcul dates filtrées)
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

        fill_value = date_end_ts
        df.loc[df['is_lab'] == False, 'date_fin'] = df.loc[df['is_lab'] == False, 'date_fin'].fillna(fill_value)

        display_end_inclusive = date_end_ts + pd.Timedelta(days=1)

        df_filtered = df[(df['date_fin'] >= date_start_ts) & (df['date_debut'] <= date_end_ts)].copy()
        if df_filtered.empty:
            st.warning("Aucune donnée dans la période sélectionnée.")
            st.stop()

        df_filtered['date_debut_plot'] = df_filtered['date_debut'].where(df_filtered['date_debut'] >= date_start_ts,
                                                                         date_start_ts)
        df_filtered['date_fin_plot'] = (df_filtered['date_fin'] + pd.Timedelta(days=1)).where(
            df_filtered['date_fin'] + pd.Timedelta(days=1) <= display_end_inclusive,
            display_end_inclusive
        )

        # --- EI, DCI et Labo ---
        df_filtered['is_ei'] = df_filtered['ei'].fillna("").astype(str).str.strip().str.lower() == "oui"
        df_filtered['is_lab'] = df_filtered['is_lab'].fillna(False)

        for col in ['dose', 'frequence', 'unite']:
            df_filtered[col] = df_filtered[col].apply(lambda x: x if pd.notna(x) else "")

        ei_df = df_filtered[df_filtered['is_ei']].copy()
        dci_only_df = df_filtered[~df_filtered['is_ei'] & ~df_filtered['is_lab']].copy()
        lab_only_df = df_filtered[df_filtered['is_lab']].copy()  # CORRECTION: Sélection des labos

        # Tri de la catégorie
        if sort_option == "Chronologique (date début)":
            # Tri des DCI : le plus récent en premier
            sorted_dci = dci_only_df.groupby('dci').agg(
                date_debut_min=('date_debut', 'min'),
                date_fin_max=('date_fin', 'max')
            ).sort_values(by=['date_debut_min', 'date_fin_max'], ascending=False).reset_index()
            dci_order = sorted_dci['dci'].tolist()

            # Tri des Labo : le plus récent en premier
            sorted_lab = lab_only_df.groupby('dci').agg(
                date_debut_min=('date_debut', 'min'),
                date_fin_max=('date_fin', 'max')
            ).sort_values(by=['date_debut_min', 'date_fin_max'], ascending=False).reset_index()
            lab_order = ["🧪 " + lab for lab in sorted_lab['dci'].tolist()]
        else:
            dci_order = sorted(dci_only_df['dci'].unique())
            lab_order = ["🧪 " + lab for lab in sorted(lab_only_df['dci'].unique())]

        ei_names = ei_df['dci'].unique().tolist()

        # Ordre des catégories souhaité (Haut vers Bas du graphique) : EI > Labo > DCI
        y_order_desired = dci_order + lab_order + ["⚠️ " + ei for ei in ei_names]

        # Création de la position Y unique
        df_filtered['y_position'] = df_filtered.apply(
            lambda r: ("⚠️ " + r['dci'] if r['is_ei'] else
                       "🧪 " + r['dci'] if r['is_lab'] else
                       r['dci']),
            axis=1
        )

        df_filtered['dose'] = pd.to_numeric(df_filtered['dose'], errors='coerce')

        # --- Normalisation dose/valeur ---
        non_ei_df_clean = df_filtered[~df_filtered['is_ei']].dropna(subset=['dose']).copy()
        df_filtered['Dose_Normalized'] = 1.0

        if not non_ei_df_clean.empty:
            min_max_doses = non_ei_df_clean.groupby('dci')['dose'].agg(['min', 'max']).reset_index()
            min_max_doses.columns = ['dci', 'Min_Dose_DCI', 'Max_Dose_DCI']

            df_filtered = df_filtered.merge(min_max_doses, on='dci', how='left')

            non_ei_mask = ~df_filtered['is_ei']
            df_filtered.loc[non_ei_mask, 'Dose_Range'] = df_filtered['Max_Dose_DCI'] - df_filtered['Min_Dose_DCI']

            df_filtered.loc[non_ei_mask, 'Dose_Normalized'] = df_filtered.loc[non_ei_mask].apply(
                lambda row: 1.0 if pd.isna(row['Dose_Range']) or row['Dose_Range'] == 0 else 0.4 + 0.6 * (
                        row['dose'] - row['Min_Dose_DCI']) / row['Dose_Range'],
                axis=1
            )
            df_filtered.drop(columns=['Min_Dose_DCI', 'Max_Dose_DCI', 'Dose_Range'], inplace=True, errors='ignore')

        df_filtered['Dose_Normalized'] = df_filtered['Dose_Normalized'].fillna(1.0)

        # --- Couleurs ---
        base_colors = ["#FFD966", "#F4B183", "#93C5FD", "#6FCF97", "#FF61C3", "#A0E7E5", "#C7A0FF"]
        meds = dci_only_df['dci'].unique()
        labs = lab_only_df['dci'].unique()

        # Création du color_map basé sur la y_position (la catégorie de la trace)
        color_map = {}
        # Couleurs DCI
        for i, dci in enumerate(meds):
            color_map[dci] = base_colors[i % len(base_colors)]

        # Couleurs Labo
        lab_color = "#C3DDE9"
        for lab in labs:
            color_map["🧪 " + lab] = lab_color

            # Couleurs EI
        ei_color = "#f4a6c1"
        for ei in ei_names:
            color_map["⚠️ " + ei] = ei_color

        # --- Posologie texte ---
        df_filtered['Posologie'] = df_filtered.apply(
            lambda r: (
                f"{r['unite']} {r['frequence']}".strip()
                if pd.isna(r['dose']) else
                f"{int(r['dose'])} {r['unite']} {r['frequence']}".strip()
                if not pd.isna(r['dose']) and pd.to_numeric(r['dose'], errors='coerce').is_integer() else
                f"{r['dose']} {r['unite']} {r['frequence']}".strip()
            ),
            axis=1
        )

        # --- NOUVEAU : Tri Final du DataFrame pour garantir l'ordre vertical des barres ---

        # 1. Créer une colonne catégorielle pour le tri. Utiliser l'ordre y_order (bas vers haut).
        y_cat_type = pd.CategoricalDtype(categories=y_order_desired, ordered=True)
        df_filtered['y_position'] = df_filtered['y_position'].astype(y_cat_type)

        # 2. Trier le DataFrame filtré. Tri 'y_position' descendant (pour que le premier élément de y_order soit en bas).
        df_filtered = df_filtered.sort_values(by=['y_position', 'date_debut_plot'],
                                              ascending=[False, True]).reset_index(drop=True)

        # 3. Reconvertir y_position en string
        df_filtered['y_position'] = df_filtered['y_position'].astype(str)

        # --- Graphique ---
        fig = px.timeline(
            df_filtered,
            x_start="date_debut_plot",
            x_end="date_fin_plot",
            y="y_position",
            # color="y_position" pour que le regroupement des barres se fasse par la catégorie affichée
            color="y_position",
            color_discrete_map=color_map,
            hover_name="y_position",
            # category_orders pour le tri des libellés Y
            category_orders={"y_position": y_order_desired},
            title="Chronologie des traitements, EI et résultats de laboratoire"
        )

        # --- Annotations texte dans les cases ---
        if show_case_text:
            fig_width_px = fig.layout.width if (fig.layout and fig.layout.width) else 1200
            total_days = (date_end_ts - date_start_ts).days
            px_per_day = fig_width_px / total_days if total_days > 0 else fig_width_px

            for _, row in df_filtered.iterrows():
                if row['is_ei'] and not show_ei_text: continue

                start = row['date_debut_plot']
                end = row['date_fin_plot']
                if pd.isna(start) or pd.isna(end) or end <= start: continue

                x_mid = start + (end - start) / 2
                y_pos = row['y_position']

                # Texte Labo : Afficher la valeur et l'unité
                if row['is_lab'] and pd.notna(row['dose']):
                    dose_val = row['dose']
                    if dose_val.is_integer():
                        dose_text = str(int(dose_val))
                    else:
                        dose_text = str(dose_val)
                    unite_val = str(row['unite']).strip()
                    text = f"{dose_text} {unite_val}".strip()
                # Texte DCI/EI : Posologie classique
                else:
                    dose_val = row['dose']
                    if pd.isna(dose_val) or dose_val == "":
                        dose_text = ""
                    elif isinstance(dose_val, (int, float)) and dose_val.is_integer():
                        dose_text = str(int(dose_val))
                    else:
                        dose_text = str(dose_val)

                    unite_val = str(row['unite']) if str(row['unite']) != "" else ""
                    freq_val = str(row['frequence']) if str(row['frequence']) != "" else ""
                    text = f"{dose_text} {unite_val} {freq_val}".strip()

                bar_width_days = (end - start).days
                bar_width_px = max(0, bar_width_days) * px_per_day
                text_px_est = len(text) * case_text_font_size * 0.6 if text else 0

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

        # Définir les valeurs des ticks de l'axe Y pour masquer les valeurs flottantes
        tick_indices = list(range(N))

        fig.update_layout(
            height=fig_height,
            width=1200,
            margin=dict(t=120, b=50, l=50, r=50),
            showlegend=False,
            title=dict(x=0.5, xanchor='center'),
            yaxis=dict(autorange="reversed")  # corrige l’ordre haut/bas
        )
        fig.update_xaxes(range=[date_start_ts, display_end_inclusive],
                         tickformat="%d %b %Y",
                         side="top",
                         tickfont=dict(size=xaxis_tick_font_size))

        # --- Calculs pour les séparations et zones ---
        N = len(y_order_desired)  # nombre total de lignes
        n_ei = len(ei_names)  # nombre de EI
        n_lab = len(lab_order)  # nombre de labos

        # --- Ligne séparation EI / Lab ---
        if n_ei > 0:
            # EI en haut → ligne juste en dessous des EI
            y_line_ei_sep = n_ei - 0.5
            fig.add_shape(
                type="line",
                x0=date_start_ts,
                x1=display_end_inclusive,
                y0=y_line_ei_sep,
                y1=y_line_ei_sep,
                line=dict(color="grey", width=1, dash="dot"),
                xref="x",
                yref="y",
                layer="above"
            )

        # --- Ligne séparation Lab / DCI ---
        if n_lab > 0 and (len(dci_only_df) > 0 or n_ei > 0):
            y_line_lab_sep = n_ei + n_lab - 0.5  # ligne sous les labos
            fig.add_shape(
                type="line",
                x0=date_start_ts,
                x1=display_end_inclusive,
                y0=y_line_lab_sep,
                y1=y_line_lab_sep,
                line=dict(color="lightgrey", width=1, dash="dot"),
                xref="x",
                yref="y",
                layer="below"
            )

        # --- Rectangle rose (zone EI) ---
        if n_ei > 0:
            # Période des EI
            ei_start = ei_df['date_debut_plot'].min()
            ei_end = ei_df['date_fin_plot'].max()

            # Nombre total de lignes
            N = df_filtered['y_position'].nunique()
            n_ei = len(ei_names)  # nombre de lignes EI
            print(n_ei)
            # Rectangle rose pour les EI
            fig.add_shape(
                type="rect",
                x0=ei_start,
                x1=ei_end,
                y0=n_ei-0.5,  # sous les EI
                y1=N-0.5,  # jusqu'au haut du bloc DCI/Labo
                fillcolor="rgba(244,166,193,0.2)",
                line=dict(width=0),
                layer="below",
                xref="x",
                yref="y"
            )

        # --- Mise à jour Plotly Trace (Opacité/Hachure) ---
        for trace in fig.data:
            # trace.name contient maintenant la y_position (ex: "⚠️ EI X", "🧪 Hémoglobine")

            # Utilisation de trace.name pour le filtre (doit correspondre à y_position)
            trace_df = df_filtered[df_filtered['y_position'] == trace.name]
            if trace_df.empty: continue

            # --- Cas EI ---
            if "⚠️" in trace.name:
                trace.marker.opacity = 1.0
                trace.marker.color = ei_color
                trace.marker.pattern = dict(shape="/")

            # --- Cas DCI/Labo (Opacité/Hover) ---
            else:
                try:
                    trace.marker.opacity = trace_df['Dose_Normalized'].tolist()
                except:
                    trace.marker.opacity = float(trace_df['Dose_Normalized'].mean())

            trace.width = 0.8
            # customdata est basé sur le trace_df filtré par y_position, garantissant le bon contenu
            trace.customdata = list(zip(
                trace_df['dci'],  # 0: Nom pur (pour le hover)
                trace_df['Posologie'],  # 1: Posologie/Valeur
                trace_df['date_debut'].dt.strftime("%d %b %Y"),  # 2: Date Début
                trace_df['date_fin'].dt.strftime("%d %b %Y")  # 3: Date Fin
            ))

            # Hover personnalisé
            is_lab_trace = trace_df['is_lab'].any()
            if is_lab_trace:
                trace.hovertemplate = "<b>🧪 Labo: %{customdata[0]}</b><br>Mesure: <b>%{customdata[1]}</b><br>Date: <b>%{customdata[2]}</b><extra></extra>"
            else:
                trace.hovertemplate = "<b>%{customdata[0]}</b><br>Période: <b>%{customdata[2]} - %{customdata[3]}</b><br>Posologie: <b>%{customdata[1]}</b><extra></extra>"

        # --- Barre verticale date de référence ---
        if show_ref_date:
            try:
                ref_date_ts = pd.Timestamp(ref_date_input) + pd.Timedelta(days=int(offset_days))
                fig.add_shape(type="line", x0=ref_date_ts, x1=ref_date_ts, y0=-0.5, y1=N - 0.5,
                              line=dict(color="#fc6b03", width=2, dash="dot"), xref="x", yref="y", layer="above")
            except Exception as e:
                st.warning(f"Impossible de tracer la date de référence : {e}")

        # --- Ticks axe X et quadrillage ---
        total_days = (date_end_ts - date_start_ts).days
        if total_days <= 14:
            freq = 'D'
        elif total_days <= 120:
            freq = 'W'
        elif total_days <= 730:
            freq = 'M'
        else:
            freq = '3M'

        tick_dates = pd.date_range(start=date_start_ts.floor('D'), end=date_end_ts.ceil('D'), freq=freq)
        fig.update_xaxes(tickvals=tick_dates, ticktext=[d.strftime("%d %b") for d in tick_dates], showgrid=False)
        for td in tick_dates:
            fig.add_shape(type="line", x0=td, x1=td, y0=-0.5, y1=N - 0.5,
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
        st.error(f"Erreur lors de la création du graphique : {e}")
        st.exception(e)


# --- Logique d'exécution principale ---

if st.session_state.data_loaded:
    show_visualization_page()
else:
    show_upload_page()