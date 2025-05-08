import streamlit as st
import PyPDF2
import urllib.request
from bs4 import BeautifulSoup
import re

@st.cache_data
def extract_text_from_pdfs(pdf_paths):
    """
    Extrait le texte de plusieurs fichiers PDF.
    
    Args:
        pdf_paths (list): Liste des chemins vers les fichiers PDF
        
    Returns:
        str: Texte extrait des PDFs
    """
    all_text = ""
    for pdf_path in pdf_paths:
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    all_text += page.extract_text() + "\n\n"
        except Exception as e:
            st.error(f"Erreur lors de la lecture du PDF {pdf_path}: {e}")
    return all_text

@st.cache_data
def extract_text_from_urls(urls):
    """
    Extrait le texte de plusieurs URLs.
    
    Args:
        urls (list): Liste des URLs
        
    Returns:
        str: Texte extrait des URLs
    """
    all_text = ""
    for url in urls:
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            req = urllib.request.Request(url, headers=headers)
            response = urllib.request.urlopen(req)
            html = response.read()
            
            soup = BeautifulSoup(html, 'html.parser')
            # Suppression des scripts et styles
            for script in soup(["script", "style"]):
                script.extract()
            
            # Extraction du texte
            text = soup.get_text()
            
            # Nettoyage du texte
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            all_text += text + "\n\n"
        except Exception as e:
            st.error(f"Erreur lors de l'extraction du texte depuis {url}: {e}")
    return all_text

def extract_relevant_knowledge(query, knowledge_base, language):
    """
    Extrait les parties pertinentes de la base de connaissances en fonction de la requête.
    
    Args:
        query (str): Requête de l'utilisateur
        knowledge_base (str): Base de connaissances complète
        language (str): Code de langue détecté
        
    Returns:
        str: Extrait pertinent de la base de connaissances
    """
    # Si la base de connaissances est courte, on la retourne entièrement
    if len(knowledge_base) < 1000:
        return knowledge_base
    
    # Sinon, on extrait des segments pertinents basés sur des mots clés
    query_lower = query.lower()
    relevant_segments = []
    
    # Identification des mots clés selon la langue
    keywords = []
    
    if language == "fr":
        if any(word in query_lower for word in ["symptome", "symptôme", "signe"]):
            keywords = ["symptôme", "signe", "caractéristique", "manifestation", "indication"]
        elif any(word in query_lower for word in ["traitement", "soigner", "guérir"]):
            keywords = ["traitement", "thérapie", "soin", "guérison", "chirurgie", "radiothérapie", "chimiothérapie"]
        elif any(word in query_lower for word in ["risque", "facteur", "prévention"]):
            keywords = ["risque", "facteur", "prévention", "dépistage", "prédisposition"]
        elif any(word in query_lower for word in ["diagnostic", "détection", "test"]):
            keywords = ["diagnostic", "détection", "test", "examen", "mammographie", "biopsie"]
    elif language == "en":
        if any(word in query_lower for word in ["symptom", "sign"]):
            keywords = ["symptom", "sign", "characteristic", "manifestation", "indication"]
        elif any(word in query_lower for word in ["treatment", "cure", "heal"]):
            keywords = ["treatment", "therapy", "care", "healing", "surgery", "radiation", "chemotherapy"]
        elif any(word in query_lower for word in ["risk", "factor", "prevention"]):
            keywords = ["risk", "factor", "prevention", "screening", "predisposition"]
        elif any(word in query_lower for word in ["diagnosis", "detection", "test"]):
            keywords = ["diagnosis", "detection", "test", "examination", "mammography", "biopsy"]
    elif language == "ar":
        if any(word in query_lower for word in ["عرض", "علامة", "أعراض"]):
            keywords = ["عرض", "علامة", "خصائص", "أعراض", "مؤشر"]
        elif any(word in query_lower for word in ["علاج", "شفاء"]):
            keywords = ["علاج", "شفاء", "رعاية", "جراحة", "إشعاع", "كيماوي"]
        elif any(word in query_lower for word in ["خطر", "عامل", "وقاية"]):
            keywords = ["خطر", "عامل", "وقاية", "فحص", "استعداد"]
        elif any(word in query_lower for word in ["تشخيص", "كشف", "فحص"]):
            keywords = ["تشخيص", "كشف", "فحص", "تصوير", "خزعة"]
    
    # Si aucun mot clé spécifique, utiliser des mots généraux sur le cancer du sein
    if not keywords:
        if language == "fr":
            keywords = ["cancer", "sein", "tumeur", "maligne", "bénigne"]
        elif language == "en":
            keywords = ["cancer", "breast", "tumor", "malignant", "benign"]
        elif language == "ar":
            keywords = ["سرطان", "ثدي", "ورم", "خبيث", "حميد"]
        else:
            keywords = ["cancer", "sein", "breast", "tumor", "tumeur"]
    
    # Diviser la base de connaissances en paragraphes
    paragraphs = re.split(r'\n\s*\n', knowledge_base)
    
    # Sélectionner les paragraphes contenant au moins un mot clé
    for paragraph in paragraphs:
        paragraph_lower = paragraph.lower()
        if any(keyword in paragraph_lower for keyword in keywords):
            relevant_segments.append(paragraph)
    
    # Si aucun segment pertinent trouvé, prendre les 3 premiers paragraphes
    if not relevant_segments and paragraphs:
        relevant_segments = paragraphs[:3]
    
    # Limiter la taille totale des segments
    combined_segments = "\n\n".join(relevant_segments)
    if len(combined_segments) > 1500:
        return combined_segments[:1500] + "..."
    
    return combined_segments
