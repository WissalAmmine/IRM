import streamlit as st
import re
import numpy as np
from llama_cpp import Llama

from modules.utils import (
    detect_language, 
    clean_llama_response, 
    generate_llama_prompt, 
    get_fallback_response
)
from modules.knowledge import extract_relevant_knowledge

# Liste de mots-clés liés à la santé dans différentes langues
HEALTH_KEYWORDS = {
    "fr": [
        "cancer", "sein", "tumeur", "mammaire", "santé", "médecin", "médical", "hôpital", 
        "maladie", "traitement", "symptôme", "douleur", "dépistage", "diagnostic", 
        "médicament", "chirurgie", "thérapie", "radiothérapie", "chimiothérapie",
        "bénin", "malin", "métastase", "cellule", "biopsie", "mammographie", "échographie",
        "prévention", "risque", "hormonal", "récidive", "guérison", "consulter"
    ],
    "en": [
        "cancer", "breast", "tumor", "health", "doctor", "medical", "hospital", 
        "disease", "treatment", "symptom", "pain", "screening", "diagnosis", 
        "medicine", "surgery", "therapy", "radiotherapy", "chemotherapy",
        "benign", "malignant", "metastasis", "cell", "biopsy", "mammography", "ultrasound",
        "prevention", "risk", "hormonal", "recurrence", "recovery", "consult"
    ],
    "ar": [
        "سرطان", "ثدي", "ورم", "صحة", "طبيب", "طبي", "مستشفى", 
        "مرض", "علاج", "عرض", "ألم", "فحص", "تشخيص", 
        "دواء", "جراحة", "معالجة", "علاج إشعاعي", "علاج كيميائي",
        "حميد", "خبيث", "نقيلة", "خلية", "خزعة", "تصوير الثدي", "تصوير بالموجات فوق الصوتية",
        "وقاية", "خطر", "هرموني", "انتكاس", "شفاء", "استشارة"
    ]
}

# Messages pour les sujets non liés à la santé
OFF_TOPIC_MESSAGES = {
    "fr": "Je suis spécialisé dans les questions relatives au cancer du sein et à la santé. Je ne peux pas répondre à cette question qui semble en dehors de mon domaine d'expertise. Si vous avez des questions sur le cancer du sein, ses symptômes, traitements ou prévention, je serai ravi de vous aider.",
    "en": "I specialize in questions related to breast cancer and health. I cannot answer this question as it appears to be outside my area of expertise. If you have questions about breast cancer, its symptoms, treatments, or prevention, I would be happy to help.",
    "ar": "أنا متخصص في الأسئلة المتعلقة بسرطان الثدي والصحة. لا يمكنني الإجابة على هذا السؤال لأنه يبدو خارج مجال خبرتي. إذا كان لديك أسئلة حول سرطان الثدي وأعراضه وعلاجاته أو الوقاية منه، فسأكون سعيدًا بمساعدتك."
}

# Salutations reconnues dans différentes langues
GREETINGS = {
    "fr": ["bonjour", "salut", "bonsoir", "coucou", "hello"],
    "en": ["hi", "hello", "hey", "good morning", "good evening"],
    "ar": ["مرحبا", "أهلا", "السلام عليكم", "السلام عليكم ورحمة الله"]
}

# Réponses aux salutations
GREETING_RESPONSES = {
    "fr": "Bonjour ! Comment puis-je vous aider concernant le cancer du sein ?",
    "en": "Hello! How can I assist you regarding breast cancer?",
    "ar": "مرحبًا! كيف يمكنني مساعدتك بشأن سرطان الثدي؟"
}

@st.cache_resource
def load_llama_model(model_path):
    """
    Charge le modèle LLama à partir du chemin spécifié.
    
    Args:
        model_path (str): Chemin vers le fichier du modèle LLama
        
    Returns:
        Llama: Modèle LLama chargé ou None en cas d'erreur
    """
    try:
        from langchain_core.callbacks import CallbackManager
        from utils import CustomStreamingCallbackHandler
        
        custom_callback = CustomStreamingCallbackHandler()
        callback_manager = CallbackManager([custom_callback])
        return Llama(
            model_path=model_path,
            n_ctx=2048,  # Contexte limité à 2048 tokens
            n_batch=256,  # Taille de batch pour l'inférence
            verbose=True,  # Désactiver les logs verbeux
            n_gpu_layers=-1,  # Utiliser le GPU si disponible (-1 = automatique)
            f16_kv=True,
            streaming=True,
            callbacks=callback_manager
        )
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle LLama: {e}")
        return None

def clean_text(text: str) -> str:
    """
    Nettoie le texte en le mettant en minuscules et en supprimant la ponctuation.
    
    Args:
        text (str): Texte à nettoyer
        
    Returns:
        str: Texte nettoyé
    """
    return re.sub(r'[^\w\s\u0600-\u06FF]', '', text.lower()).strip()

def detect_greeting_language(text: str) -> str | None:
    """
    Détecte si le texte commence par une salutation et renvoie la langue.
    
    Args:
        text (str): Texte à analyser
        
    Returns:
        str | None: Code de langue ('fr', 'en', 'ar') ou None si pas de salutation
    """
    cleaned = clean_text(text)
    
    # Vérifier si le texte commence par l'une des salutations connues
    for lang, greets in GREETINGS.items():
        for g in greets:
            if cleaned.startswith(g):
                return lang
    
    return None

def is_health_related(text: str, language: str="fr") -> bool:
    """
    Vérifie si le texte est lié à la santé en se basant sur des mots-clés.
    
    Args:
        text (str): Texte à analyser
        language (str): Langue du texte (fr, en, ar)
        
    Returns:
        bool: True si le texte est lié à la santé, False sinon
    """
    txt = text.lower()
    
    # Vérifier les mots-clés dans la langue principale
    for kw in HEALTH_KEYWORDS.get(language, HEALTH_KEYWORDS["fr"]):
        if kw in txt:
            return True
    
    # Vérifier les mots-clés en français et en anglais quelle que soit la langue
    for lang in ("fr", "en"):
        if lang != language:
            for kw in HEALTH_KEYWORDS[lang]:
                if kw in txt:
                    return True
    
    return False

def get_llama_response(llama_model, query: str, knowledge_base: str) -> str:
    """
    Obtient une réponse du modèle LLama.
    
    Args:
        llama_model (Llama): Modèle LLama
        query (str): Requête de l'utilisateur
        knowledge_base (str): Base de connaissances
        
    Returns:
        str: Réponse générée
    """
    # Détection de la langue de la question
    language = detect_language(query)
    
    # Vérifier si la question est liée à la santé
    if not is_health_related(query, language):
        return OFF_TOPIC_MESSAGES.get(language, OFF_TOPIC_MESSAGES["fr"])
        
    # Extraction d'information pertinente
    know = extract_relevant_knowledge(query, knowledge_base, language)
    
    # Génération du prompt
    prompt = generate_llama_prompt(query, language, know, use_correction=True)
    
    # Limiter la taille du prompt si nécessaire
    if len(prompt) > 100000:
        st.warning(f"Prompt trop long, troncature…")
        prompt = generate_llama_prompt(query, language, know[:2000]+"…", use_correction=True)
    
    try:
        # Génération de la réponse
        resp = llama_model.create_completion(
            prompt,
            max_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.2,
            stop=["<|user|>", "<|system|>", "<|assistant|>", "[INST]", "[/INST]", "</s>", "<s>"],
            echo=False
        )
        
        # Extraction et nettoyage de la réponse
        text = resp["choices"][0]["text"].strip()
        
        # Vérifier si la réponse est vide ou trop courte
        if not text or len(text) < 10:
            return get_fallback_response(query, language)
            
        # Nettoyer la réponse
        return clean_llama_response(text, query)
        
    except Exception as e:
        st.error(f"Erreur génération LLaMA: {e}")
        return get_fallback_response(query, language)

def get_fallback_responses(query: str, language: str) -> str:
    """
    Fournit des réponses prédéfinies basées sur la catégorie de la question.
    
    Args:
        query (str): Question de l'utilisateur
        language (str): Langue détectée
        
    Returns:
        str: Réponse appropriée
    """
    # Dictionnaire de réponses prédéfinies
    responses = {
        "fr": {
            "cancer_info": "Le cancer du sein est une maladie où les cellules mammaires se multiplient de façon anormale. Un dépistage précoce augmente significativement les chances de guérison.",
            "benign_info": "Une tumeur bénigne n'est pas cancéreuse. Elle ne se propage pas aux tissus environnants et n'est généralement pas dangereuse pour la santé.",
            "malignant_info": "Une tumeur maligne est cancéreuse et peut envahir les tissus environnants. Un traitement médical rapide est essentiel.",
            "treatment": "Les traitements courants du cancer du sein incluent la chirurgie, la radiothérapie, la chimiothérapie, l'hormonothérapie et les thérapies ciblées.",
            "default": "Je n'ai pas d'information spécifique sur ce sujet. Consultez un professionnel de santé pour des conseils médicaux personnalisés."
        },
        "en": {
            "cancer_info": "Breast cancer is a disease where breast cells multiply abnormally. Early detection significantly increases chances of recovery.",
            "benign_info": "A benign tumor is not cancerous. It does not spread to surrounding tissues and is generally not dangerous to health.",
            "malignant_info": "A malignant tumor is cancerous and can invade surrounding tissues. Prompt medical treatment is essential.",
            "treatment": "Common breast cancer treatments include surgery, radiation therapy, chemotherapy, hormone therapy, and targeted therapies.",
            "default": "I don't have specific information on this topic. Please consult a healthcare professional for personalized medical advice."
        },
        "ar": {
            "cancer_info": "سرطان الثدي هو مرض تتكاثر فيه خلايا الثدي بشكل غير طبيعي. يزيد الكشف المبكر بشكل كبير من فرص الشفاء.",
            "benign_info": "الورم الحميد ليس سرطانيًا. لا ينتشر إلى الأنسجة المحيطة وعادة لا يشكل خطرًا على الصحة.",
            "malignant_info": "الورم الخبيث سرطاني ويمكن أن يغزو الأنسجة المحيطة. العلاج الطبي السريع ضروري.",
            "treatment": "تشمل علاجات سرطان الثدي الشائعة الجراحة والعلاج الإشعاعي والعلاج الكيميائي والعلاج الهرموني والعلاجات المستهدفة.",
            "default": "ليس لدي معلومات محددة حول هذا الموضوع. يرجى استشارة أخصائي رعاية صحية للحصول على نصائح طبية مخصصة."
        }
    }
    
    # Si la langue n'est pas prise en charge, utiliser le français
    if language not in responses:
        language = "fr"
    
    # Logique simple pour déterminer la catégorie de la question
    query_lower = query.lower()
    
    # Déterminer la catégorie de la question
    if any(word in query_lower for word in ["cancer", "sein", "mammaire", "breast", "سرطان", "ثدي"]):
        return responses[language]["cancer_info"]
    elif any(word in query_lower for word in ["bénin", "benin", "bénigne", "benign", "حميد"]):
        return responses[language]["benign_info"]
    elif any(word in query_lower for word in ["malin", "maligne", "malignes", "malignant", "خبيث"]):
        return responses[language]["malignant_info"]
    elif any(word in query_lower for word in ["traitement", "soigner", "guérir", "guérison", "treatment", "therapy", "علاج"]):
        return responses[language]["treatment"]
    else:
        return responses[language]["default"]

def get_bot_response(question: str, llama_model, knowledge_base: str, detected_condition=None) -> str:
    """
    Obtient une réponse du chatbot en fonction de la question.
    
    Args:
        question (str): Question de l'utilisateur
        llama_model (Llama): Modèle LLama (ou None)
        knowledge_base (str): Base de connaissances
        detected_condition (str, optional): Condition détectée dans l'image
        
    Returns:
        str: Réponse du chatbot
    """
    # 1. Vérifier si c'est une salutation simple
    greeting_lang = detect_greeting_language(question)
    if greeting_lang:
        return GREETING_RESPONSES.get(greeting_lang, GREETING_RESPONSES["fr"])
    
    # 2. Déterminer la langue de la question
    language = detect_language(question)
    
    # 3. Vérifier si la question est liée à la santé
    if not is_health_related(question, language):
        return OFF_TOPIC_MESSAGES.get(language, OFF_TOPIC_MESSAGES["fr"])
    
    # 4. Traiter la condition détectée dans l'image si disponible
    if detected_condition == "normal":
        return {
            "fr": "D'après l'analyse de votre image, une tumeur bénigne a été détectée. Les tumeurs bénignes ne sont généralement pas cancéreuses, mais un suivi médical est recommandé.",
            "en": "Based on the analysis of your image, a benign tumor has been detected. Benign tumors are generally not cancerous, but medical follow-up is recommended.",
            "ar": "بناءً على تحليل صورتك، تم اكتشاف ورم حميد. الأورام الحميدة ليست سرطانية عمومًا، ولكن يوصى بالمتابعة الطبية."
        }.get(language, "D'après l'analyse de votre image, une tumeur bénigne a été détectée. Les tumeurs bénignes ne sont généralement pas cancéreuses, mais un suivi médical est recommandé.")
    
    elif detected_condition == "cancer":
        return {
            "fr": "D'après l'analyse de votre image, une tumeur maligne a été détectée. Veuillez consulter un médecin dès que possible pour une évaluation professionnelle.",
            "en": "Based on the analysis of your image, a malignant tumor has been detected. Please consult a doctor as soon as possible for a professional evaluation.",
            "ar": "بناءً على تحليل صورتك، تم اكتشاف ورم خبيث. يرجى استشارة الطبيب في أقرب وقت ممكن للحصول على تقييم مهني."
        }.get(language, "D'après l'analyse de votre image, une tumeur maligne a été détectée. Veuillez consulter un médecin dès que possible pour une évaluation professionnelle.")
    
    # 5. Utiliser le modèle LLama si disponible
    if llama_model:
        return get_llama_response(llama_model, question, knowledge_base)
    
    # 6. Utiliser les réponses prédéfinies comme solution de repli
    return get_fallback_responses(question, language)
