import streamlit as st
from streamlit_chat import message
from PIL import Image
import numpy as np
import torch
import time
import uuid
import logging
import inspect
import asyncio

# Importation des modules personnalisés
from modules.detection import load_model, predict_with_yolo, display_results
from modules.chat import get_bot_response, load_llama_model
from modules.knowledge import extract_text_from_pdfs, extract_text_from_urls
# Importation directe depuis utils.py
from modules.utils import (
    detect_language,
)
from utils import CustomStreamingCallbackHandler,CustomStreamingCallbackHandlerWHatsapp,timer_decorator
# Importation du gestionnaire de callbacks existant
from modules.callbacks import callback_manager, on_tumor_detection, on_message_received, on_error_occurred
from config import (
    LLAMA_MODEL_PATH, UNIFIED_MODEL_PATH, PDF_PATHS, URLS, 
    set_page_config
)

# Configuration de la page Streamlit
set_page_config()

# Configuration du logger pour supprimer les messages de Streamlit
logger = logging.getLogger('streamlit')
logger.setLevel(logging.ERROR)

def load_resources_silently():
    """
    Fonction pour charger silencieusement tous les modèles et ressources nécessaires
    """
    # Chargement du modèle YOLO
    try:
        st.session_state.model = load_model(UNIFIED_MODEL_PATH)
        if not st.session_state.model:
            logger.error("Échec du chargement du modèle YOLO.")
            on_error_occurred("model_loading_error", "Échec du chargement du modèle YOLO", "detection")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle YOLO: {e}")
        on_error_occurred("model_loading_error", str(e), "detection")
    
    # Chargement du modèle LLama
    try:
        # Vérifier l'existence du fichier
        import os
        if not os.path.exists(LLAMA_MODEL_PATH):
            logger.error(f"Le fichier modèle Llama n'existe pas: {LLAMA_MODEL_PATH}")
            on_error_occurred("model_loading_error", f"Fichier introuvable: {LLAMA_MODEL_PATH}", "chat")
            st.session_state.llama_model = None
            st.session_state.use_llama = False
        else:
            st.session_state.llama_model = load_llama_model(LLAMA_MODEL_PATH)
            if not st.session_state.llama_model:
                logger.warning("Échec du chargement du modèle LLama.")
                st.session_state.use_llama = False
                on_error_occurred("model_loading_error", "Échec du chargement du modèle LLama", "chat")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle LLama: {e}")
        st.session_state.use_llama = False
        on_error_occurred("model_loading_error", str(e), "chat")
    
    # Extraction des connaissances
    try:
        pdf_text = extract_text_from_pdfs(PDF_PATHS)
    except Exception as e:
        logger.error(f"Erreur lors du traitement des PDFs: {e}")
        pdf_text = ""
        on_error_occurred("knowledge_extraction_error", str(e), "knowledge_pdf")
    
    try:
        url_text = extract_text_from_urls(URLS)
    except Exception as e:
        logger.error(f"Erreur lors du traitement des URLs: {e}")
        url_text = ""
        on_error_occurred("knowledge_extraction_error", str(e), "knowledge_url")
    
    # Combiner les textes disponibles
    st.session_state.knowledge_base = pdf_text + "\n\n" + url_text

def main():
    st.title("🏥 Chatbot de Détection de Cancer du Sein")
    
    # Initialisation des variables d'état de session
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if 'model' not in st.session_state:
        st.session_state.model = None
    
    if 'llama_model' not in st.session_state:
        st.session_state.llama_model = None
        
    if 'knowledge_base' not in st.session_state:
        st.session_state.knowledge_base = ""
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
        
    if 'detected_condition' not in st.session_state:
        st.session_state.detected_condition = None
    
    if 'use_llama' not in st.session_state:
        # Toujours utiliser LLama s'il est disponible
        st.session_state.use_llama = True
    
    if 'conf_threshold' not in st.session_state:
        st.session_state.conf_threshold = 0.25
    
    # Initialiser le gestionnaire de streaming adapté à WhatsApp
    if 'streaming_handler' not in st.session_state:
        st.session_state.streaming_handler = CustomStreamingCallbackHandlerWHatsapp(mode='sentence')
        
    if 'resources_loaded' not in st.session_state or not st.session_state.resources_loaded:
        # Chargement silencieux des modèles et ressources au démarrage
        with st.spinner("Initialisation de l'application..."):
            load_resources_silently()
            st.session_state.resources_loaded = True
    
    # Barre latérale pour les options
    sidebar_section()
    
    # Mise en page en deux colonnes
    col1, col2 = st.columns([3, 2])
    
    # Colonne pour le chat
    with col1:
        chat_section()
    
    # Colonne pour l'analyse d'image
    with col2:
        image_analysis_section()

def sidebar_section():
    """Section pour la barre latérale avec les configurations"""
    with st.sidebar:
        st.header("Configuration")
        
        # Seuil de confiance pour YOLO
        st.session_state.conf_threshold = st.slider(
            "Seuil de confiance pour la détection:", 
            0.0, 1.0, 0.25, 0.05,
            key="conf_threshold_sidebar"
        )
        
        # Section des callbacks (optionnelle)
        with st.expander("Options avancées"):
            if st.button("Sauvegarder l'historique"):
                save_path = f"session_history_{st.session_state.session_id}.json"
                callback_manager.save_history(save_path)
                st.success(f"Historique sauvegardé dans {save_path}")
            
            if st.button("Effacer l'historique"):
                callback_manager.clear_history()
                st.success("Historique effacé")
            
            # Options du mode de streaming
            streaming_mode = st.radio(
                "Mode de streaming:",
                ["sentence", "paragraph"],
                index=0 if st.session_state.streaming_handler.mode == "sentence" else 1
            )
            
            if streaming_mode != st.session_state.streaming_handler.mode:
                st.session_state.streaming_handler = CustomStreamingCallbackHandlerWHatsapp(mode=streaming_mode)
                st.success(f"Mode de streaming modifié: {streaming_mode}")
        
        st.markdown("---")
        st.markdown("### À propos")
        st.info(
            "Cette application utilise un modèle pour détecter les tumeurs mammaires "
            "et intègre un chatbot pour répondre à vos questions sur le cancer du sein "
            "en se basant sur des sources médicales fiables."
        )
        
        if st.button("Réinitialiser la conversation"):
            st.session_state.messages = []
            st.session_state.uploaded_image = None
            st.session_state.detected_condition = None
            
            # Déclencher un callback pour la fin de session
            callback_manager.trigger(
                "on_session_end", 
                session_id=st.session_state.session_id,
                session_duration=time.time() - st.session_state.get('session_start_time', time.time())
            )
            
            # Nouvelle session
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.session_start_time = time.time()

def chat_section():
    """Section pour l'interface de chat"""
    st.subheader("💬 Conversation")
    
    # Affichage des messages
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            message(msg["content"], is_user=True, key=f"msg_{i}")
        else:
            message(msg["content"], is_user=False, key=f"msg_{i}")
    
    # Zone de saisie du message
    with st.form(key="message_form", clear_on_submit=True):
        user_input = st.text_input("Tapez votre message ici:", key="user_input")
        submit_button = st.form_submit_button("Envoyer")
    
    # Traitement du message
    if submit_button and user_input:
        # Ajouter le message de l'utilisateur
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Vérifier la disponibilité du modèle LLama
        use_llama_now = st.session_state.get('llama_model') is not None
        
        # Configuration pour l'affichage progressif de la réponse
        response_placeholder = st.empty()
        current_response = [""]
        
        # Fonction de callback pour afficher la réponse progressivement
        def update_response(text_chunk):
            current_response[0] += text_chunk
            response_placeholder.markdown(current_response[0])
        
        # Enregistrer le callback dans le gestionnaire de streaming
        st.session_state.streaming_handler.register_callback(update_response)
        
        # Générer la réponse avec un indicateur de chargement
        with st.spinner("Réponse en cours..."):
            try:
                detected_language = detect_language(user_input)
                
                # Essayer d'utiliser la version modifiée de get_bot_response qui accepte streaming_callback
                try:
                    response = get_bot_response(
                        user_input, 
                        st.session_state.llama_model if use_llama_now else None,
                        st.session_state.knowledge_base,
                        st.session_state.detected_condition,
                        streaming_callback=st.session_state.streaming_handler
                    )
                except TypeError:
                    # Si la fonction n'accepte pas le paramètre streaming_callback
                    logger.warning("La fonction get_bot_response ne supporte pas le streaming")
                    response = get_bot_response(
                        user_input, 
                        st.session_state.llama_model if use_llama_now else None,
                        st.session_state.knowledge_base,
                        st.session_state.detected_condition
                    )
                    # Afficher manuellement la réponse complète
                    update_response(response)
                
                # Déclencher le callback de message
                on_message_received(
                    user_message=user_input,
                    bot_response=response,
                    detected_language=detected_language
                )
            except Exception as e:
                logger.error(f"Erreur lors de la génération de la réponse: {e}")
                import traceback
                logger.error(traceback.format_exc())
                
                # Utiliser une réponse de secours en cas d'erreur
                from modules.utils import get_fallback_response
                response = get_fallback_response(user_input, detected_language)
                update_response(response)
                on_error_occurred("response_generation_error", str(e), "chat")
        
        # Ajouter la réponse du bot
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Forcer le rafraîchissement
        st.rerun()

def image_analysis_section():
    """Section pour l'analyse d'image"""
    st.subheader("🔍 Analyse d'Image")
    
    # Upload d'image
    uploaded_file = st.file_uploader("Choisissez une image mammographique", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.session_state.uploaded_image = image
        
        # Vérifier les modèles disponibles
        if st.session_state.model is not None:
            with st.spinner("Analyse en cours..."):
                try:
                    # Résultats YOLO - utilisation du seuil stocké dans session_state
                    yolo_results = predict_with_yolo(
                        st.session_state.model, 
                        image, 
                        st.session_state.conf_threshold
                    )
                    
                    # Vérifier si des tumeurs ont été détectées
                    has_detections = yolo_results and len(yolo_results) > 0 and len(yolo_results[0].boxes) > 0
                    
                    # Si des tumeurs sont détectées, déclencher les callbacks
                    if has_detections:
                        # Récupérer la détection avec la plus haute confiance
                        boxes = yolo_results[0].boxes
                        highest_conf_idx = torch.argmax(boxes.conf).item()
                        highest_conf = float(boxes.conf[highest_conf_idx])
                        
                        # Générer un ID unique pour cette image
                        image_id = f"img_{int(time.time())}_{uploaded_file.name}"
                        
                        # Déterminer si la tumeur est maligne selon le seuil
                        from config import MALIGNANCY_THRESHOLD
                        is_malignant = highest_conf > MALIGNANCY_THRESHOLD
                        
                        # Déclencher le callback de détection
                        on_tumor_detection(
                            image_id=image_id,
                            confidence=highest_conf,
                            is_malignant=is_malignant,
                            image_path=uploaded_file.name
                        )
                    
                    # Afficher les résultats
                    display_results(yolo_results, image)
                    
                    # Ajouter automatiquement un message du système concernant la détection
                    if st.session_state.detected_condition:
                        # Détecter la langue utilisée dans la conversation
                        user_lang = "fr"  # Par défaut en français
                        if st.session_state.messages:
                            last_user_msg = next((msg["content"] for msg in reversed(st.session_state.messages) 
                                                if msg["role"] == "user"), None)
                            if last_user_msg:
                                user_lang = detect_language(last_user_msg)
                        
                        # Préparer la réponse dans la langue détectée
                        if user_lang == "en":
                            if st.session_state.detected_condition == "cancer":
                                response = "I have analyzed your image and detected a tumor classified as 'malignant'. This classification suggests a potential cancer. I recommend consulting a doctor as soon as possible for a professional evaluation."
                            else:
                                response = "I have analyzed your image and detected a tumor classified as 'benign'. This classification suggests a non-cancerous tumor, but medical follow-up is still recommended."
                        elif user_lang == "ar":
                            if st.session_state.detected_condition == "cancer":
                                response = "لقد قمت بتحليل صورتك واكتشفت ورمًا مصنفًا على أنه 'خبيث'. يشير هذا التصنيف إلى احتمال وجود سرطان. أوصي باستشارة الطبيب في أقرب وقت ممكن للحصول على تقييم مهني."
                            else:
                                response = "لقد قمت بتحليل صورتك واكتشفت ورمًا مصنفًا على أنه 'حميد'. يشير هذا التصنيف إلى ورم غير سرطاني، ولكن لا يزال يُنصح بالمتابعة الطبية."
                        else:  # Français par défaut
                            if st.session_state.detected_condition == "cancer":
                                response = "J'ai analysé votre image et j'ai détecté une tumeur classée comme 'maligne'. Cette classification suggère un cancer potentiel. Je vous recommande de consulter un médecin dès que possible pour une évaluation professionnelle."
                            else:
                                response = "J'ai analysé votre image et j'ai détecté une tumeur classée comme 'bénigne'. Cette classification suggère une tumeur non cancéreuse, mais un suivi médical est toujours recommandé."
                        
                        # Ajouter la réponse du système à la conversation
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        # Déclencher le callback de message pour cette réponse automatique
                        on_message_received(
                            user_message="[IMAGE ANALYSIS]",
                            bot_response=response,
                            detected_language=user_lang
                        )
                except Exception as e:
                    st.error(f"Erreur lors de l'analyse d'image: {e}")
                    on_error_occurred("image_analysis_error", str(e), "detection")
        else:
            st.warning("Le modèle de détection n'est pas disponible. Veuillez réessayer dans quelques instants.")
            st.image(image, caption="Image téléchargée (en attente d'analyse)", use_container_width=True)

if __name__ == "__main__":
    # Stocker le temps de début de session
    if 'session_start_time' not in st.session_state:
        st.session_state.session_start_time = time.time()
    
    main()