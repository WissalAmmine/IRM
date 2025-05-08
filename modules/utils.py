import re
import langdetect
from langdetect import detect

def detect_language(text):
    """
    Détecte la langue du texte fourni.
    
    Args:
        text (str): Texte à analyser
        
    Returns:
        str: Code de langue détecté (fr, en, ar, etc.)
    """
    try:
        return langdetect.detect(text)
    except:
        return "fr"  # Par défaut, on retourne le français

def clean_llama_response(answer, query=""):
    """
    Nettoie la réponse du modèle LLama pour garder uniquement les détails 
    pertinents à la question posée.
    
    Args:
        answer (str): Réponse brute du modèle LLama
        query (str): Question originale de l'utilisateur (pour l'analyse de pertinence)
        
    Returns:
        str: Réponse nettoyée et pertinente
    """
    import re
    
    if not answer or len(answer) < 10:
        return "Je ne peux pas générer une réponse complète pour le moment."
    
    # Suppression des instructions internes et des formulations de question
    answer = re.sub(r'INSTRUCTION (CRITIQUE|INTERNE|CRITICAL).*?Question:', '', answer, flags=re.DOTALL | re.IGNORECASE)
    answer = re.sub(r'INTERNAL INSTRUCTION.*?Question:', '', answer, flags=re.DOTALL | re.IGNORECASE)
    answer = re.sub(r'تعليمات (مهمة|داخلية).*?السؤال:', '', answer, flags=re.DOTALL | re.IGNORECASE)
    answer = re.sub(r'Question corrigée et reformulée\s*:.*?\n', '', answer, flags=re.DOTALL | re.IGNORECASE)
    answer = re.sub(r'Corrected and rephrased question\s*:.*?\n', '', answer, flags=re.DOTALL | re.IGNORECASE)
    answer = re.sub(r'السؤال المصحح والمعاد صياغته\s*:.*?\n', '', answer, flags=re.DOTALL | re.IGNORECASE)
    
    # Suppression des mentions à la correction
    answer = re.sub(r'J\'ai corrigé et reformulé votre question\.', '', answer, flags=re.IGNORECASE)
    answer = re.sub(r'I have corrected and rephrased your question\.', '', answer, flags=re.IGNORECASE)
    answer = re.sub(r'لقد قمت بتصحيح وإعادة صياغة سؤالك\.', '', answer, flags=re.IGNORECASE)
    
    # Nettoyage des balises HTML et du formatage
    answer = re.sub(r'<[^>]*>', '', answer)
    answer = re.sub(r'[\*\_\~\`\|]+', '', answer)
    
    # Correction des termes médicaux erronés
    incorrect_terms = [
        "glandes salivaires", "cancéro-breast", "Léucodésques de Kallmann", "Mélancoloma",
        "tissue mammary", "moustacelles", "cancérisation", "tumor anaplastique",
        "lécithines", "cellules moustacelles", "anaplastique intraépithéliale",
        "tumors anaplastiques"
    ]
    
    for term in incorrect_terms:
        answer = answer.replace(term, "")
    
    # Extraction des mots-clés de la question pour l'analyse de pertinence
    if query:
        query_lower = query.lower()
        question_words = set(re.findall(r'\b\w+\b', query_lower))
        significant_words = {word for word in question_words if len(word) > 3}
        
        # Séparation en paragraphes
        paragraphs = re.split(r'\n\s*\n', answer)
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 20]
        
        # Évaluation de la pertinence de chaque paragraphe
        scored_paragraphs = []
        for i, para in enumerate(paragraphs):
            para_lower = para.lower()
            
            # Calcul du score basé sur la présence de mots-clés
            para_words = set(re.findall(r'\b\w+\b', para_lower))
            word_match_score = len(para_words.intersection(significant_words)) * 2
            
            # Bonus pour les paragraphes en position primaire (premier paragraphe)
            position_score = max(0, 3 - i) * 2
            
            # Pénalité pour les formules de politesse et invitations
            politeness_penalty = 0
            if re.search(r'n\'hésitez pas|je suis là|en espérant|j\'espère|pour toute question', para_lower):
                politeness_penalty = 10
                
            # Score final
            total_score = word_match_score + position_score - politeness_penalty
            scored_paragraphs.append((para, total_score))
        
        # Tri des paragraphes par score et sélection des meilleurs
        scored_paragraphs.sort(key=lambda x: x[1], reverse=True)
        best_paragraphs = [p[0] for p in scored_paragraphs[:4]]  # Garder jusqu'à 4 paragraphes
        
        # Réorganisation des paragraphes dans leur ordre d'origine pour maintenir la cohérence
        ordered_paragraphs = [p for p in paragraphs if p in best_paragraphs]
        
        # Si aucun paragraphe n'est retenu, prendre le premier paragraphe
        if not ordered_paragraphs and paragraphs:
            ordered_paragraphs = [paragraphs[0]]
            
        # Reconstitution de la réponse avec les paragraphes retenus
        answer = "\n\n".join(ordered_paragraphs)
    
    # Nettoyage final
    
    # Suppression des formules de politesse et invitations
    answer = re.sub(r'N\'hésitez[^\.\n]*?à me poser[^\.\n]*?\.', '', answer, flags=re.IGNORECASE | re.DOTALL)
    answer = re.sub(r'Nous sommes là pour t\'aider[^\.\n]*?\.', '', answer, flags=re.IGNORECASE | re.DOTALL) 
    answer = re.sub(r'Si tu n\'obtiens pas[^\.\n]*?\.', '', answer, flags=re.IGNORECASE | re.DOTALL)
    answer = re.sub(r'Vous pouvez aussi contacter[^\.\n]*?\.', '', answer, flags=re.IGNORECASE | re.DOTALL)
    answer = re.sub(r'Si j\'avais des réponses[^\.\n]*?\.', '', answer, flags=re.IGNORECASE | re.DOTALL)
    answer = re.sub(r'Je suis[^\.\n]*?à votre service[^\.\n]*?\.', '', answer, flags=re.IGNORECASE | re.DOTALL)
    answer = re.sub(r'n\'hésitez pas à me poser[^\.\n]*?\.', '', answer, flags=re.IGNORECASE | re.DOTALL)
    answer = re.sub(r'N\'hésitez pas à me demander[^\.\n]*?\.', '', answer, flags=re.IGNORECASE | re.DOTALL)
    answer = re.sub(r'Je suis disponible[^\.\n]*?\.', '', answer, flags=re.IGNORECASE | re.DOTALL)
    answer = re.sub(r'J\'espère que cette [^\.\n]*? vous a aidé[^\.\n]*?\.', '', answer, flags=re.IGNORECASE | re.DOTALL)
    answer = re.sub(r'En espérant avoir répondu à votre question[^\.\n]*?\.', '', answer, flags=re.IGNORECASE | re.DOTALL)
    answer = re.sub(r'Avez-vous d\'autres questions[^\.\n]*?\.', '', answer, flags=re.IGNORECASE | re.DOTALL)
    
    # Suppression des formules de conclusion
    answer = re.sub(r'En résumé,[^\.]*?(?=\. [A-Z]|$)', '', answer, flags=re.IGNORECASE | re.DOTALL)
    answer = re.sub(r'En conclusion,[^\.]*?(?=\. [A-Z]|$)', '', answer, flags=re.IGNORECASE | re.DOTALL)
    
    # Suppression des suggestions de consultations médicales
    answer = re.sub(r'Il est (donc |)important que vous consultiez un médecin[^\.]*?\.', '', answer, flags=re.IGNORECASE | re.DOTALL)
    answer = re.sub(r'vous devriez consulter un médecin[^\.]*?\.', '', answer, flags=re.IGNORECASE | re.DOTALL)
    answer = re.sub(r'consultez un professionnel de santé[^\.]*?\.', '', answer, flags=re.IGNORECASE | re.DOTALL)
    
    # Suppression des mentions d'importance
    answer = re.sub(r'Il est (aussi |également |)important de noter[^\.]*?\.', '', answer, flags=re.IGNORECASE | re.DOTALL)
    
    # Suppression des mentions de complexité du sujet
    answer = re.sub(r'il s\'agit (donc |)d\'un sujet (très |)vaste et complexe[^\.]*?\.', '', answer, flags=re.IGNORECASE | re.DOTALL)
    answer = re.sub(r'Il n\'est pas possible dans ce contexte[^\.]*?\.', '', answer, flags=re.IGNORECASE | re.DOTALL)
    
    # Nettoyage des espaces et retours à la ligne
    answer = re.sub(r'\n{3,}', '\n\n', answer)
    answer = re.sub(r' {2,}', ' ', answer)
    
    # Suppression des préfixes standard
    answer = re.sub(r'^(Réponse|Answer|إجابة)\s*:\s*', '', answer, flags=re.IGNORECASE)
    
    # Nettoyage final
    answer = answer.strip()
    
    return answer

def generate_llama_prompt(query, language, knowledge_base="", use_correction=True):
    """
    Génère un prompt pour obtenir une réponse détaillée mais strictement pertinente.
    
    Args:
        query (str): Requête de l'utilisateur
        language (str): Code de langue détecté
        knowledge_base (str, optional): Base de connaissances pertinente
        use_correction (bool): Si True, inclut l'étape de correction de la requête
        
    Returns:
        str: Prompt formaté pour LLama
    """
    if language == "fr":
        system_message = """
        Tu es un assistant médical spécialisé dans le cancer du sein. RÉPONDS UNIQUEMENT ET DIRECTEMENT À CE QUI EST DEMANDÉ dans la question. 
        Fournis une réponse détaillée, en utilisant plusieurs paragraphes si nécessaire pour bien expliquer chaque point, mais fais-le de manière structurée et claire. 

        N'introduis PAS de définitions ou d'informations non demandées, et ne t'écarte pas du sujet précis de la question. 
        Assure-toi que ta réponse est grammaticale, sans fautes d'orthographe, et bien formulée. 
        N'utilise pas de formules de politesse ni d'invitations à poser d'autres questions.
        Donne une réponse claire de maximum 10 lignes et qui convient à la question posée.
        Dans la réponse, tu dois uniquement répondre à la question posée, sans aborder d'autres sujets qui ne sont pas demandés. Réponds strictement à la question, ni plus ni moins.       
        Termine la phrase, ne t'arrête pas en cours de réponse.
        """
    elif language == "en":
        system_message = """
        You are a medical assistant specializing in breast cancer. ANSWER ONLY AND DIRECTLY WHAT IS ASKED in the question. 
        Provide a detailed response, using multiple paragraphs when necessary to explain each point clearly and effectively. 

        DO NOT introduce definitions or information that wasn't asked for and don't deviate from the specific topic of the question. 
        Ensure your answer is grammatically correct, free from spelling mistakes, and well-structured. 
        Don't use politeness formulas or invitations to ask other questions.
        Provide a clear response with a maximum of 10 lines that fits the question being asked.
        In your answer, you should only respond to the question being asked, without addressing any other topics. Respond strictly to the question, no more, no less.       
        Finish the sentence, do not stop halfway.
        """
    elif language == "ar":
        system_message = """
        أنت مساعد طبي متخصص في سرطان الثدي. أجب فقط ومباشرة على ما هو مطلوب في السؤال. 
        قدم إجابة مفصلة باستخدام عدة فقرات عند الحاجة لشرح كل نقطة بوضوح وفعالية. 

        لا تقدم تعريفات أو معلومات غير مطلوبة ولا تنحرف عن الموضوع المحدد للسؤال. 
        تأكد من أن إجابتك صحيحة من الناحية النحوية وخالية من الأخطاء الإملائية ومهيكلة بشكل جيد. 
        لا تستخدم صيغ المجاملة أو الدعوات لطرح أسئلة أخرى.
        قدم إجابة واضحة لا تتجاوز 10 أسطر تتناسب مع السؤال المطروح.
        في إجابتك، يجب أن تقتصر على الإجابة عن السؤال المطروح دون التطرق إلى مواضيع أخرى. أجب بدقة على السؤال، لا أكثر ولا أقل.       
        أكمل الجملة، لا تتوقف في منتصف الجواب.
        """
    else:
        system_message = """
        Tu es un assistant médical spécialisé dans le cancer du sein. RÉPONDS UNIQUEMENT ET DIRECTEMENT À CE QUI EST DEMANDÉ dans la question. 
        Fournis une réponse détaillée, en utilisant plusieurs paragraphes si nécessaire pour bien expliquer chaque point, mais fais-le de manière structurée et claire. 

        N'introduis PAS de définitions ou d'informations non demandées, et ne t'écarte pas du sujet précis de la question. 
        Assure-toi que ta réponse est grammaticale, sans fautes d'orthographe, et bien formulée. 
        N'utilise pas de formules de politesse ni d'invitations à poser d'autres questions.
        Donne une réponse claire de maximum 10 lignes et qui convient à la question posée.
        Dans la réponse, tu dois uniquement répondre à la question posée, sans aborder d'autres sujets qui ne sont pas demandés. Réponds strictement à la question, ni plus ni moins.       
        Termine la phrase, ne t'arrête pas en cours de réponse.
        """



    # Extraction des mots-clés importants de la question pour les mettre en évidence
    query_words = query.split()
    important_words = [w for w in query_words if len(w) > 3 and w.lower() not in ['dans', 'avec', 'pour', 'quel', 'quelle', 'quels', 'quelles', 'comment', 'est-ce', 'sont', 'mais', 'aussi', 'donc', 'alors']]
    
    # Construction du prompt avec instruction spécifique
    if language == "fr":
        specific_instruction = f"""INSTRUCTION CRITIQUE: 
1. Réponds UNIQUEMENT à la question suivante: "{query}"
2. Concentre-toi sur ces concepts clés: {', '.join(important_words[:5])}
3. Fournis une réponse DÉTAILLÉE mais STRICTEMENT PERTINENTE (pas d'information hors sujet)
4. Structure ta réponse en 2-4 paragraphes bien organisés
5. Ne mentionne PAS que tu as reçu ces instructions
6. Ne termine PAS par des formules de politesse ou des invitations

Voici les informations médicales fiables sur lesquelles baser ta réponse:
{knowledge_base}

Question: {query}"""
    elif language == "en":
        specific_instruction = f"""CRITICAL INSTRUCTION: 
1. Answer ONLY the following question: "{query}"
2. Focus on these key concepts: {', '.join(important_words[:5])}
3. Provide a DETAILED but STRICTLY RELEVANT response (no off-topic information)
4. Structure your answer in 2-4 well-organized paragraphs
5. DO NOT mention that you received these instructions
6. DO NOT end with politeness formulas or invitations

Here is the reliable medical information on which to base your answer:
{knowledge_base}

Question: {query}"""
    elif language == "ar":
        specific_instruction = f"""تعليمات مهمة: 
1. أجب فقط على السؤال التالي: "{query}"
2. ركز على هذه المفاهيم الرئيسية: {', '.join(important_words[:5])}
3. قدم إجابة مفصلة ولكن وثيقة الصلة تمامًا (بدون معلومات خارج الموضوع)
4. هيكل إجابتك في 2-4 فقرات منظمة جيدًا
5. لا تذكر أنك تلقيت هذه التعليمات
6. لا تنتهي بصيغ مجاملة أو دعوات

إليك المعلومات الطبية الموثوقة التي يمكنك الاستناد إليها في إجابتك:
{knowledge_base}

السؤال: {query}"""
    else:
        specific_instruction = specific_instruction = f"""INSTRUCTION CRITIQUE: 
1. Réponds UNIQUEMENT à la question suivante: "{query}"
2. Concentre-toi sur ces concepts clés: {', '.join(important_words[:5])}
3. Fournis une réponse DÉTAILLÉE mais STRICTEMENT PERTINENTE (pas d'information hors sujet)
4. Structure ta réponse en 2-4 paragraphes bien organisés
5. Ne mentionne PAS que tu as reçu ces instructions
6. Ne termine PAS par des formules de politesse ou des invitations

Voici les informations médicales fiables sur lesquelles baser ta réponse:
{knowledge_base}

Question: {query}"""
    
    # Format final pour le prompt LLama
    prompt = f"<s>[INST] {system_message}\n\n{specific_instruction} [/INST]"
    
    return prompt

def get_fallback_response(query, language):
    """
    Génère une réponse de repli basée sur des règles simples et la langue détectée.
    
    Args:
        query (str): Requête de l'utilisateur
        language (str): Code de langue détecté
        
    Returns:
        str: Réponse de repli
    """
    query_lower = query.lower()
    
    # Réponses en français
    if language == "fr":
        if any(word in query_lower for word in ["bonjour", "salut", "coucou"]):
            return "Bonjour! Je suis votre assistant spécialisé dans le cancer du sein. Comment puis-je vous aider aujourd'hui?"
        elif any(word in query_lower for word in ["merci", "remercie"]):
            return "Je vous en prie! N'hésitez pas si vous avez d'autres questions."
        elif any(word in query_lower for word in ["symptome", "symptôme", "signe"]):
            return "Les symptômes courants du cancer du sein incluent une bosse ou un épaississement dans le sein, un changement de taille ou de forme du sein, des modifications de la peau du sein (rougeur, fossettes), un écoulement du mamelon et une douleur dans le sein ou le mamelon. Il est important de consulter un médecin si vous remarquez l'un de ces signes."
        elif any(word in query_lower for word in ["traitement", "soigner", "guérir"]):
            return "Les traitements du cancer du sein peuvent inclure la chirurgie (tumorectomie ou mastectomie), la radiothérapie, la chimiothérapie, l'hormonothérapie et les thérapies ciblées. Le plan de traitement est personnalisé en fonction du stade du cancer, de son type et des caractéristiques de la patiente."
        elif any(word in query_lower for word in ["risque", "facteur", "prévention"]):
            return "Les facteurs de risque du cancer du sein incluent l'âge, les antécédents familiaux, les mutations génétiques (BRCA1 et BRCA2), l'exposition aux œstrogènes, le surpoids après la ménopause, la consommation d'alcool et l'inactivité physique. La prévention passe par un mode de vie sain et un dépistage régulier."
        else:
            return "Je ne peux pas générer une réponse complète à cette question pour le moment. Je vous suggère de reformuler votre question ou de consulter un professionnel de santé pour obtenir des informations précises sur le cancer du sein."
    
    # Réponses en anglais
    elif language == "en":
        if any(word in query_lower for word in ["hello", "hi", "hey"]):
            return "Hello! I'm your breast cancer specialist assistant. How can I help you today?"
        elif any(word in query_lower for word in ["thank", "thanks"]):
            return "You're welcome! Feel free to ask if you have any other questions."
        elif any(word in query_lower for word in ["symptom", "sign"]):
            return "Common breast cancer symptoms include a lump or thickening in the breast, change in breast size or shape, changes to the skin of the breast (redness, dimpling), nipple discharge, and pain in the breast or nipple. It's important to consult a doctor if you notice any of these signs."
        elif any(word in query_lower for word in ["treatment", "cure", "heal"]):
            return "Breast cancer treatments may include surgery (lumpectomy or mastectomy), radiation therapy, chemotherapy, hormone therapy, and targeted therapies. The treatment plan is personalized based on the stage of cancer, its type, and the patient's characteristics."
        elif any(word in query_lower for word in ["risk", "factor", "prevention"]):
            return "Breast cancer risk factors include age, family history, genetic mutations (BRCA1 and BRCA2), estrogen exposure, being overweight after menopause, alcohol consumption, and physical inactivity. Prevention involves a healthy lifestyle and regular screening."
        else:
            return "I cannot generate a complete answer to this question at the moment. I suggest rephrasing your question or consulting a healthcare professional for accurate information about breast cancer."
    
    # Réponses en arabe
    elif language == "ar":
        if any(word in query_lower for word in ['صباح الخير ',"مرحبا", "سلام", "أهلا"]):
            return "مرحباً! أنا مساعدك المتخصص في سرطان الثدي. كيف يمكنني مساعدتك اليوم؟"
        elif any(word in query_lower for word in ["شكرا"]):
            return "على الرحب والسعة! لا تتردد في السؤال إذا كان لديك أي استفسارات أخرى."
        elif any(word in query_lower for word in ["عرض", "علامة", "أعراض"]):
            return "تشمل أعراض سرطان الثدي الشائعة وجود كتلة أو سماكة في الثدي، تغير في حجم أو شكل الثدي، تغيرات في جلد الثدي (احمرار، نقر)، إفرازات من الحلمة، وألم في الثدي أو الحلمة. من المهم استشارة الطبيب إذا لاحظت أياً من هذه العلامات."
        elif any(word in query_lower for word in ["علاج", "شفاء"]):
            return "قد تشمل علاجات سرطان الثدي الجراحة (استئصال الورم أو استئصال الثدي)، والعلاج الإشعاعي، والعلاج الكيميائي، والعلاج الهرموني، والعلاجات المستهدفة. يتم تخصيص خطة العلاج بناءً على مرحلة السرطان، ونوعه، وخصائص المريضة."
        elif any(word in query_lower for word in ["خطر", "عامل", "وقاية"]):
            return "تشمل عوامل خطر الإصابة بسرطان الثدي العمر، والتاريخ العائلي، والطفرات الجينية (BRCA1 و BRCA2)، والتعرض للإستروجين، وزيادة الوزن بعد انقطاع الطمث، واستهلاك الكحول، وقلة النشاط البدني. تتضمن الوقاية نمط حياة صحي وفحص منتظم."
        else:
            return "لا يمكنني تقديم إجابة كاملة على هذا السؤال في الوقت الحالي. أقترح إعادة صياغة سؤالك أو استشارة أخصائي رعاية صحية للحصول على معلومات دقيقة حول سرطان الثدي."
    
    # Réponse par défaut (français)
    else:
        return "Je ne peux pas générer une réponse complète à cette question pour le moment. Je vous suggère de reformuler votre question ou de consulter un professionnel de santé pour obtenir des informations précises sur le cancer du sein."
