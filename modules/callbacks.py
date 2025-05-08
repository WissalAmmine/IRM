"""
Module pour gérer les callbacks et événements dans l'application.
"""
import logging
from typing import Dict, Any, Callable, List, Optional
import time
import json
import os

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app_events.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Types de callbacks disponibles
CALLBACK_TYPES = [
    "on_detection",       # Déclenché quand une tumeur est détectée
    "on_classification",  # Déclenché quand une tumeur est classifiée
    "on_message",         # Déclenché quand un message est échangé
    "on_error",           # Déclenché en cas d'erreur
    "on_startup",         # Déclenché au démarrage de l'application
    "on_session_end",     # Déclenché à la fin d'une session
]

class CallbackManager:
    """Gestionnaire de callbacks pour l'application."""
    
    def __init__(self):
        """Initialise le gestionnaire de callbacks."""
        self.callbacks: Dict[str, List[Callable]] = {
            callback_type: [] for callback_type in CALLBACK_TYPES
        }
        self.history: List[Dict[str, Any]] = []
        self.max_history = 100  # Nombre maximum d'entrées dans l'historique
        
        # Enregistre le démarrage du gestionnaire
        logger.info("Gestionnaire de callbacks initialisé")
    
    def register(self, event_type: str, callback: Callable) -> None:
        """
        Enregistre un callback pour un type d'événement spécifique.
        
        Args:
            event_type: Type d'événement (doit être dans CALLBACK_TYPES)
            callback: Fonction à appeler lors de l'événement
        """
        if event_type not in CALLBACK_TYPES:
            raise ValueError(f"Type d'événement inconnu: {event_type}. "
                            f"Les types valides sont: {', '.join(CALLBACK_TYPES)}")
        
        self.callbacks[event_type].append(callback)
        logger.debug(f"Callback enregistré pour l'événement: {event_type}")
    
    def trigger(self, event_type: str, **kwargs) -> None:
        """
        Déclenche tous les callbacks enregistrés pour un type d'événement.
        
        Args:
            event_type: Type d'événement à déclencher
            **kwargs: Arguments à passer aux callbacks
        """
        if event_type not in CALLBACK_TYPES:
            logger.warning(f"Tentative de déclenchement d'un événement inconnu: {event_type}")
            return
        
        # Ajoute l'événement à l'historique
        event_data = {
            "type": event_type,
            "timestamp": time.time(),
            "data": kwargs
        }
        self.history.append(event_data)
        
        # Limite la taille de l'historique
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        # Log l'événement
        logger.info(f"Événement déclenché: {event_type}")
        
        # Exécute tous les callbacks pour cet événement
        for callback in self.callbacks[event_type]:
            try:
                callback(**kwargs)
            except Exception as e:
                logger.error(f"Erreur dans le callback pour {event_type}: {str(e)}")
    
    def save_history(self, file_path: str = "callback_history.json") -> None:
        """
        Sauvegarde l'historique des événements dans un fichier JSON.
        
        Args:
            file_path: Chemin du fichier où sauvegarder l'historique
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(self.history, f, indent=2)
            logger.info(f"Historique sauvegardé dans {file_path}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de l'historique: {str(e)}")
    
    def clear_history(self) -> None:
        """Efface l'historique des événements."""
        self.history = []
        logger.info("Historique des événements effacé")

# Instance globale du gestionnaire de callbacks
callback_manager = CallbackManager()

# Fonctions d'aide pour les événements courants
def on_tumor_detection(image_id: str, confidence: float, is_malignant: bool, 
                        image_path: Optional[str] = None) -> None:
    """
    Déclenche l'événement de détection de tumeur.
    
    Args:
        image_id: Identifiant de l'image
        confidence: Score de confiance de la détection
        is_malignant: Si la tumeur est classifiée comme maligne
        image_path: Chemin vers l'image analysée (optionnel)
    """
    callback_manager.trigger(
        "on_detection",
        image_id=image_id,
        confidence=confidence,
        is_malignant=is_malignant,
        image_path=image_path,
        classification_type="maligne" if is_malignant else "bénigne"
    )

def on_message_received(user_message: str, bot_response: str, 
                        detected_language: str = "fr") -> None:
    """
    Déclenche l'événement de réception de message.
    
    Args:
        user_message: Message de l'utilisateur
        bot_response: Réponse du chatbot
        detected_language: Langue détectée dans le message
    """
    callback_manager.trigger(
        "on_message",
        user_message=user_message,
        bot_response=bot_response,
        detected_language=detected_language,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
    )

def on_error_occurred(error_type: str, error_message: str, 
                     module: str = "unknown") -> None:
    """
    Déclenche l'événement d'erreur.
    
    Args:
        error_type: Type d'erreur
        error_message: Message d'erreur
        module: Module où l'erreur est survenue
    """
    callback_manager.trigger(
        "on_error",
        error_type=error_type,
        error_message=error_message,
        module=module,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
    )

# Exportation des fonctions et classes principales
__all__ = [
    'callback_manager', 
    'CallbackManager', 
    'on_tumor_detection', 
    'on_message_received', 
    'on_error_occurred',
    'CALLBACK_TYPES'
]
