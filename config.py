import streamlit as st
import os

# Chemins des ressources
LLAMA_MODEL_PATH = "./models/Llama-3.2-3B-Instruct-Q5_K_M.gguf"
UNIFIED_MODEL_PATH = "./models/best1.pt"



# Liste des PDFs pour la base de connaissances
PDF_PATHS = [
    "./PDFs/Breast Imaging_ The Requisites.pdf",
    "./PDFs/brochure_cancer_sein.pdf",
    "./PDFs/cancer du sein .pdf",
    "./PDFs/Guide Thérapeutique.pdf",
    
]

# Liste des URLs pour la base de connaissances
URLS = [
    "https://www.fondation-arc.org/cancer/cancer-sein/facteurs-risque-cancer",
    "https://www.e-cancer.fr/Patients-et-proches/Les-cancers/Cancer-du-sein/Les-facteurs-de-risque",
    "https://www.ameli.fr/assure/sante/themes/cancer-sein/symptomes-diagnostic",
    "https://www.contrelecancer.ma/en/",
 
]

# Configuration de la page Streamlit
def set_page_config():
    """
    Configure les paramètres de la page Streamlit
    """
    st.set_page_config(
        page_title="Chatbot de Détection de Cancer",
        page_icon="🏥",
        layout="wide"
    )

# Seuil de confiance pour la classification des tumeurs malignes/bénignes
MALIGNANCY_THRESHOLD = 0.70

