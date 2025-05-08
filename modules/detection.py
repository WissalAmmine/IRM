import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image
import uuid
import time
from config import MALIGNANCY_THRESHOLD

# Importation conditionnelle pour éviter les erreurs circulaires
try:
    from modules.callbacks import callback_manager
except ImportError:
    callback_manager = None

@st.cache_resource
def load_model(model_path):
    """
    Charge le modèle YOLO à partir du chemin spécifié.
    
    Args:
        model_path (str): Chemin vers le fichier du modèle YOLO
        
    Returns:
        YOLO: Modèle YOLO chargé ou None en cas d'erreur
    """
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle YOLO: {e}")
        return None

def predict_with_yolo(model, image, conf_threshold=0.25):
    """
    Effectue une prédiction sur une image avec le modèle YOLO.
    
    Args:
        model (YOLO): Modèle YOLO
        image (PIL.Image): Image à analyser
        conf_threshold (float): Seuil de confiance
        
    Returns:
        list: Résultats de la prédiction ou None en cas d'erreur
    """
    try:
        # Convertir l'image en RGB si nécessaire
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Faire la prédiction
        results = model.predict(image, conf=conf_threshold)
        
        # Déclencher un callback pour la prédiction
        if callback_manager and len(results) > 0 and len(results[0].boxes) > 0:
            callback_manager.trigger(
                "on_detection",
                detection_count=len(results[0].boxes),
                confidence_scores=[float(box.conf[0]) for box in results[0].boxes],
                prediction_time=time.time(),
                threshold_used=conf_threshold
            )
        
        return results
    except Exception as e:
        st.error(f"Erreur lors de la prédiction YOLO: {e}")
        # Déclencher un callback d'erreur si disponible
        if callback_manager:
            callback_manager.trigger(
                "on_error",
                error_type="prediction_error",
                error_message=str(e),
                module="detection"
            )
        return None

def display_results(yolo_results, image):
    """
    Affiche les résultats de l'analyse YOLO.
    
    Args:
        yolo_results (list): Résultats de la prédiction YOLO
        image (PIL.Image): Image analysée
    """
    st.subheader("Résultats de l'analyse")
    
    # Générer un ID unique pour cette analyse
    analysis_id = f"analysis_{uuid.uuid4().hex[:8]}"
    
    # Afficher l'image originale
    st.image(image, caption="Image originale", use_container_width=True)
    
    # Afficher les résultats du modèle YOLO
    st.subheader("🔍 Résultats de l'analyse")
    
    if yolo_results and len(yolo_results) > 0:
        # Afficher l'image avec les annotations YOLO
        res = yolo_results[0]
        boxes = res.boxes
        
        if len(boxes) > 0:
            # Créer une nouvelle image avec les annotations
            annotated_img = res.plot()
            st.image(annotated_img, caption="Image avec détections", use_container_width=True)
            
            # Variables pour les statistiques
            malignant_count = 0
            benign_count = 0
            all_confidences = []
            
            # Afficher les détails des détections
            for i, box in enumerate(boxes):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                original_class_name = res.names[cls]
                all_confidences.append(conf)
                
                # Classification basée sur le seuil de confiance
                if conf > MALIGNANCY_THRESHOLD:
                    adjusted_class_name = "cancer"  # Malin
                    display_name = "MALIGNE"
                    color = "🔴"
                    malignant_count += 1
                else:
                    adjusted_class_name = "normal"  # Bénin
                    display_name = "BÉNIGNE"
                    color = "🟢"
                    benign_count += 1
                
                st.write(f"{color} Détection {i+1}: Tumeur {display_name} (Confiance: {conf:.2f})")
                
                '''# Ajouter une explication de la classification
                if adjusted_class_name != original_class_name:
                    if adjusted_class_name == "cancer":
                        st.write(f"   ℹ️ Reclassé comme MALIGNE car le score de confiance ({conf:.2f}) est supérieur à {MALIGNANCY_THRESHOLD}")
                    else:
                        st.write(f"   ℹ️ Reclassé comme BÉNIGNE car le score de confiance ({conf:.2f}) est inférieur à {MALIGNANCY_THRESHOLD}")
                '''
                # Déclencher un callback pour chaque classification
                if callback_manager:
                    callback_manager.trigger(
                        "on_classification",
                        detection_id=f"{analysis_id}_det{i}",
                        confidence=conf,
                        original_class=original_class_name,
                        adjusted_class=adjusted_class_name,
                        is_malignant=(conf > MALIGNANCY_THRESHOLD)
                    )
            
            # Stocker la condition détectée pour le chatbot basée sur la détection avec la plus haute confiance
            highest_conf_idx = torch.argmax(boxes.conf).item()
            highest_conf = float(boxes.conf[highest_conf_idx])
            
            # Utiliser le seuil pour déterminer la classification finale
            if highest_conf > MALIGNANCY_THRESHOLD:
                st.session_state.detected_condition = "cancer"  # Malin
            else:
                st.session_state.detected_condition = "normal"  # Bénin
            
            # Déclencher un callback avec les statistiques globales
            if callback_manager:
                callback_manager.trigger(
                    "on_analysis_complete",
                    analysis_id=analysis_id,
                    total_detections=len(boxes),
                    malignant_count=malignant_count,
                    benign_count=benign_count,
                    average_confidence=sum(all_confidences) / len(all_confidences) if all_confidences else 0,
                    highest_confidence=highest_conf,
                    final_classification=st.session_state.detected_condition
                )
        else:
            st.info("Aucune tumeur n'a été détectée dans cette image.")
            st.session_state.detected_condition = None
            
            # Déclencher un callback pour absence de détection
            if callback_manager:
                callback_manager.trigger(
                    "on_analysis_complete",
                    analysis_id=analysis_id,
                    total_detections=0,
                    result="no_detection"
                )
    else:
        st.info("Aucune tumeur n'a été détectée dans cette image.")
        st.session_state.detected_condition = None
        
        # Déclencher un callback pour absence de résultats
        if callback_manager:
            callback_manager.trigger(
                "on_analysis_complete",
                analysis_id=analysis_id,
                total_detections=0,
                result="no_results"
            )
    
    # Afficher une conclusion
    st.subheader("Conclusion")
    
    if yolo_results and len(yolo_results) > 0 and len(yolo_results[0].boxes) > 0:
        # Extraire les résultats de YOLO pour la conclusion
        boxes = yolo_results[0].boxes
        highest_conf_idx = torch.argmax(boxes.conf).item()
        highest_conf = float(boxes.conf[highest_conf_idx])
        
        # Utiliser le seuil pour la conclusion finale
        if highest_conf > MALIGNANCY_THRESHOLD:
            st.error("⚠️ Le modèle détecte une tumeur MALIGNE avec une confiance de {:.1f}%. Caractéristiques possibles: masse irrégulière avec spicules. Veuillez consulter un médecin rapidement.".format(highest_conf * 100))
            conclusion = "maligne"
        else:
            st.success("✅ Le modèle détecte une tumeur BÉNIGNE avec une confiance de {:.1f}%. Caractéristiques possibles: masse ronde à contours lisses. Un suivi médical régulier est recommandé.".format(highest_conf * 100))
            conclusion = "bénigne"
            
        # Déclencher un callback pour la conclusion
        if callback_manager:
            callback_manager.trigger(
                "on_conclusion",
                analysis_id=analysis_id,
                conclusion=conclusion,
                confidence=highest_conf,
                threshold_used=MALIGNANCY_THRESHOLD
            )
    else:
        st.info("Aucune tumeur n'a été détectée dans cette image. Cependant, si vous avez des préoccupations, consultez un professionnel de santé.")
        
        # Déclencher un callback pour la conclusion sans détection
        if callback_manager:
            callback_manager.trigger(
                "on_conclusion",
                analysis_id=analysis_id,
                conclusion="aucune_détection",
                confidence=0,
                threshold_used=MALIGNANCY_THRESHOLD
            )
