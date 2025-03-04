import logging
import re
from spellchecker import SpellChecker
import flair
import numpy as np
from nltk.corpus import wordnet
from nltk.stem.snowball import FrenchStemmer
from sentence_transformers import util
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import base64

from ...services.model_loader import paraphrase_model, nlp, nlp_flair, spell, bert_large

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_text(text):
    """
    Nettoie un texte en supprimant HTML, scripts, espaces et caractères spéciaux.
    """
    try:
        if not isinstance(text, str):
            raise TypeError("L'entrée doit être une chaîne de caractères.")

        text = re.sub(r'<[^>]+>', '', text)  # Suppression des balises HTML
        text = re.sub(r'<script.*?>.*?</script>', '', text, flags=re.DOTALL)
        text = re.sub(r'<style.*?>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'\s+', ' ', text).strip()  # Normalisation des espaces
        text = re.sub(r'[\x00-\x1F\x7F]', '', text)  # Suppression des caractères invisibles
        text = re.sub(r'[^\w\s,.!?;:\'\"()\[\]{}-]', '', text)  # Nettoyage des caractères spéciaux

        return text

    except Exception as e:
        logger.error(f"❌ Erreur lors du nettoyage du texte : {e}")
        return text

import re

def extract_phone_numbers(text):
    """Extrait les numéros de téléphone sous forme valide."""
    return re.findall(r'\b0[1-9](?:[ .-]?[0-9]{2}){4}\b', text)  # Amélioré pour mieux capturer les formats communs


def extract_emails(text):
    """Extrait les emails valides, y compris ceux avec des caractères spéciaux et des sous-domaines."""
    # Regex améliorée pour accepter plus de formats d'email, y compris les sous-domaines et caractères spéciaux
    return re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,7}\b', text)

def extract_money_amounts(text):
    """Extrait les montants financiers sous format €."""
    return re.findall(r'\b\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})? ?€\b', text)  # Capture les formats avec séparateurs


def extract_dates(text):
    """Extrait les dates sous différents formats courants, incluant les dates relatives."""
    # Regex améliorée pour inclure plusieurs formats de date : jour/mois/année, mois/jour/année, etc.
    return re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b', text)

def extract_percentages(text):
    """Extrait les pourcentages sous différents formats, y compris les espaces et les symboles entre le nombre et %."""
    # Regex améliorée pour inclure les pourcentages sous différentes formes (espaces et chiffres avec plusieurs décimales)
    return re.findall(r'\b\d{1,3}(?:[.,]\d+)? ?%\b', text)

def tokenize_text(text):
    """
    Tokenise un texte en conservant la structure grammaticale et les entités nommées.
    """
    try:
        if not isinstance(text, str):
            raise TypeError("L'entrée doit être une chaîne de caractères.")

        # Chargement du modèle seulement si nécessaire
        # nlp = spacy.load("fr_core_news_md")
        doc = nlp(text)
        tokens = [token.text for token in doc]
        return tokens

    except Exception as e:
        logger.error(f"❌ Erreur lors de la tokenisation du texte : {e}")
        return text.split()  # Fallback simple


def remove_stopwords(tokens):
    """
    Supprime les stopwords d'une liste de tokens tout en conservant les entités nommées.
    """
    try:
        if not isinstance(tokens, list):
            raise TypeError("L'entrée doit être une liste de tokens.")

        # Chargement du modèle seulement si nécessaire
        # nlp = spacy.load("fr_core_news_md")
        doc = nlp(" ".join(tokens))
        filtered_tokens = [token.text for token in doc if not token.is_stop or token.ent_type_]
        return filtered_tokens

    except Exception as e:
        logger.error(f"❌ Erreur lors de la suppression des stopwords : {e}")
        return tokens


def stem_text(tokens):
    """
    Applique la stemmatisation à une liste de tokens.
    """
    try:
        if not isinstance(tokens, list):
            raise TypeError("L'entrée doit être une liste de tokens.")

        # Chargement du stemmer seulement si nécessaire
        stemmer = FrenchStemmer()
        return [stemmer.stem(token) for token in tokens]

    except Exception as e:
        logger.error(f"❌ Erreur lors de la stemmatisation : {e}")
        return tokens


def lemmatize_text(tokens):
    """
    Applique la lemmatisation à une liste de tokens.
    """
    try:
        if not isinstance(tokens, list):
            raise TypeError("L'entrée doit être une liste de tokens.")

        # Chargement du modèle seulement si nécessaire
        # nlp = spacy.load("fr_core_news_md")
        doc = nlp(" ".join(tokens))
        return [token.lemma_ if not token.ent_type_ else token.text for token in doc]

    except Exception as e:
        logger.error(f"❌ Erreur lors de la lemmatisation : {e}")
        return tokens


def extract_named_entities_spacy(text):
    """
    Extrait les entités nommées en utilisant SpaCy.
    """
    try:
        doc = nlp(text)
        entities = {"PER": set(), "LOC": set(), "ORG": set(), "MISC": set()}

        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].add(ent.text)
            else:
                entities["MISC"].add(ent.text)

        return {key: list(values) for key, values in entities.items()}
    except Exception as e:
        logger.error(f"❌ Erreur avec SpaCy : {e}")
        return {}


def extract_named_entities_flair(text):
    """
    Extrait les entités nommées en utilisant Flair.
    """
    try:
        sentence = flair.data.Sentence(text)
        nlp_flair.predict(sentence)
        entities = {"PER": set(), "LOC": set(), "ORG": set(), "MISC": set()}

        for entity in sentence.get_spans("ner"):
            label = entity.get_label("ner").value
            if label in entities:
                entities[label].add(entity.text)
            else:
                entities["MISC"].add(entity.text)

        return {key: list(values) for key, values in entities.items()}
    except Exception as e:
        logger.error(f"❌ Erreur avec Flair : {e}")
        return {}


def extract_named_entities_combined(text):
    """
    Combine les résultats de SpaCy et Flair pour améliorer la précision de l'extraction des entités.
    """
    try:
        entities_spacy = extract_named_entities_spacy(text)
        entities_flair = extract_named_entities_flair(text)

        combined_entities = {key: set(entities_spacy.get(key, []) + entities_flair.get(key, [])) for key in
                             set(entities_spacy) | set(entities_flair)}
        return {key: list(values) for key, values in combined_entities.items()}
    except Exception as e:
        logger.error(f"❌ Erreur lors de la combinaison des entités : {e}")
        return {}


def encode_image_to_base64(image_path: str) -> str:
    """
    Convertit une image en base64.
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
    except Exception as e:
        logger.error(f"Erreur lors de la conversion de l'image en base64 : {e}")
        return ""


def convert_numpy_types(data):
    """
    Convertit récursivement les objets numpy (par exemple numpy.float32) en types Python natifs.
    """
    if isinstance(data, np.generic):
        return data.item()
    elif isinstance(data, dict):
        return {k: convert_numpy_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(i) for i in data]
    else:
        return data


# def extract_entities_advanced(
#     text: str,
#     min_score: float = 0.5
# ) -> dict:
#     """
#     Effectue une classification zéro-shot sur un texte à l'aide de facebook/bart-large-mnli
#     afin de déterminer son thème parmi des catégories administratives du gouvernement français.
#
#     :param text: Le texte à analyser.
#     :param min_score: Score minimal pour conserver un label.
#     :return: Un dictionnaire contenant :
#              - "cleaned_text": le texte nettoyé,
#              - "entities": liste de labels + scores (après filtrage par min_score),
#              - "raw_output": la sortie brute du pipeline zero-shot-classification.
#     :raises ValueError: Si le texte fourni est vide.
#     """
#     if not text or not text.strip():
#         logger.error("Le texte fourni est vide.")
#         raise ValueError("Le texte fourni est vide.")
#
#     # Nettoyage du texte : suppression des espaces multiples
#     cleaned_text = re.sub(r'\s+', ' ', text).strip()
#
#     try:
#         # Chargement du pipeline de classification zéro-shot
#         classifier = pipeline(
#             "zero-shot-classification",
#             model="facebook/bart-large-mnli"
#         )
#
#         # Labels pour des thématiques administratives françaises
#         candidate_labels = [
#             "Fiscalité",         # Impôts, taxes, TVA, déclarations fiscales
#             "Logement",          # Aides au logement, HLM
#             "Transport",         # Permis de conduire, carte grise, transports en commun
#             "État civil",        # Naissance, mariage, décès
#             "Sécurité",          # Police, gendarmerie, défense
#             "Justice",           # Tribunaux, procès, lois
#             "Travail",           # Contrats, droit du travail, emploi
#             "Retraite",          # Pensions, réformes des retraites
#             "Santé",             # Sécurité sociale, assurance maladie
#             "Éducation",         # École, université, formation
#             "Allocations",       # Allocations familiales, RSA, chômage
#             "Élections",         # Carte électorale, vote
#             "Entreprise",        # Auto-entrepreneur, création d'entreprise
#             "Environnement",     # Écologie, développement durable
#             "Culture",           # Patrimoine, musées, spectacles
#             "Sports",            # Pratiques sportives, événements sportifs
#             "Administration",    # Demandes de documents officiels, formulaires
#         ]
#
#         # Classification zéro-shot
#         raw_output = classifier(cleaned_text, candidate_labels)
#     except Exception as e:
#         logger.error(f"Erreur lors de la classification zéro-shot avec BART : {e}")
#         raise e
#
#     # Le pipeline renvoie un dict : {"sequence", "labels", "scores"}
#     # On filtre les labels selon un score minimal
#     entities = []
#     try:
#         labels = raw_output.get("labels", [])
#         scores = raw_output.get("scores", [])
#         for label, score in zip(labels, scores):
#             if score >= min_score:
#                 entities.append({
#                     "text": label,
#                     "score": float(score)
#                 })
#     except Exception as e:
#         logger.error(f"Erreur lors du traitement des labels/scores : {e}")
#
#     logger.info(f"{len(entities)} labels conservés après filtrage (min_score={min_score}).")
#
#     result = {
#         "cleaned_text": cleaned_text,
#         "entities": entities,
#         "raw_output": raw_output
#     }
#
#     # Convertir récursivement les types numpy en types natifs Python
#     result = convert_numpy_types(result)
#
#     return result

def extract_entities_advanced(
    text: str,
    min_score: float = 0.5,
    aggregation: str = "simple"
) -> dict:
    """
    Extrait les entités nommées d'un texte en français en utilisant le modèle
    Jean-Baptiste/camembert-ner via le pipeline de Hugging Face.

    :param text: Le texte en français à analyser.
    :param min_score: Score minimal pour filtrer les entités retenues.
    :param aggregation: Stratégie d'agrégation NER (ex. "simple", "first", "max", etc.)
    :return: Un dictionnaire contenant :
             - "cleaned_text": le texte nettoyé
             - "entities": liste d'entités filtrées
             - "raw_output": la sortie brute du pipeline
    :raises ValueError: Si le texte est vide.
    """

    # Vérification de la validité du texte
    if not text or not text.strip():
        logger.error("Le texte fourni est vide.")
        raise ValueError("Le texte fourni est vide.")

    # Nettoyage du texte (suppression des espaces multiples)
    cleaned_text = re.sub(r"\s+", " ", text).strip()

    try:
        # Initialisation du pipeline NER pour le français
        ner_pipeline = pipeline(
            "ner",
            model="Jean-Baptiste/camembert-ner",
            tokenizer="Jean-Baptiste/camembert-ner",
            aggregation_strategy=aggregation
        )

        raw_output = ner_pipeline(cleaned_text)
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction NER: {e}")
        raise e

    # Filtrage des entités par score minimal
    entities = []
    for ent in raw_output:
        score = float(ent.get("score", 0.0))
        if score >= min_score:
            entities.append({
                "text": ent.get("word", ""),        # ou "entity_group" selon aggregation
                "label": ent.get("entity_group", ""),
                "start": ent.get("start", None),
                "end": ent.get("end", None),
                "score": score
            })

    logger.info(f"{len(entities)} entités conservées (score >= {min_score}).")

    # Conversion récursive des types numpy en types Python natifs
    result = {
        "cleaned_text": cleaned_text,
        "entities": entities,
        "raw_output": raw_output
    }
    result = convert_numpy_types(result)

    return result

def correct_ocr_errors(text):
    """
    Applique des techniques avancées de correction des erreurs OCR basées sur le contexte linguistique et l'orthographe.
    """
    try:
        # Vérifier si le texte est None ou vide
        if text is None or not str(text).strip():
            logger.warning("Le texte fourni est None ou vide, renvoi d'une chaîne vide.")
            return ""

        # S'assurer que text est de type str
        text = str(text)

        # Suppression des artefacts fréquents : remplacements basés sur des patterns
        text = re.sub(r'(?<!\d)1(?!\d)', 'l', text)  # remplace '1' isolé par 'l'
        text = re.sub(r'(?<!\d)0(?!\d)', 'o', text)  # remplace '0' isolé par 'o'
        text = re.sub(r'(?<!\d)5(?!\d)', 's', text)  # remplace '5' isolé par 's'
        text = re.sub(r'(?<!\w)rn(?!\w)', 'm', text)  # remplace 'rn' par 'm'
        text = re.sub(r'(?<!\d)\-(?!\d)', ' ', text)  # remplace tirets non numériques par espace

        # Nettoyage : suppression d'espaces multiples
        text = re.sub(r'\s+', ' ', text).strip()

        # # Correction orthographique et vérification avec WordNet pour le français
        # words = text.split()
        # corrected_words = []
        #
        # for word in words:
        #     # Correction du mot via autocorrect (pour les mots de plus de 2 caractères)
        #     corrected_word = spell.correction(word) if len(word) > 2 else word
        #
        #     # Vérifier avec WordNet (en français) : si des synsets existent pour le mot corrigé, on le garde
        #     # Note : wordnet.synsets() attend une chaîne, donc on s'assure que corrected_word est bien une chaîne.
        #     if isinstance(corrected_word, str) and wordnet.synsets(corrected_word, lang='fra'):
        #         corrected_words.append(corrected_word)
        #     else:
        #         # Sinon, on garde le mot original ou la correction si elle est une chaîne
        #         corrected_words.append(corrected_word if isinstance(corrected_word, str) else str(corrected_word))
        #
        # corrected_text = " ".join(corrected_words)

        # Correction contextuelle supplémentaire : par exemple, transformer "l" en "1" si nécessaire
        corrected_text = re.sub(r'(?<=\w)l(?=\d)', '1', text)

        return corrected_text

    except Exception as e:
        logger.error(f"❌ Erreur lors de la correction OCR : {e}")
        return text


def get_synonym(word):
    """
    Récupère un synonyme du mot donné en utilisant WordNet.
    """
    try:
        if not isinstance(word, str):
            raise TypeError("L'entrée doit être une chaîne de caractères.")

        synsets = wordnet.synsets(word, lang="fra")
        if synsets:
            synonym = synsets[0].lemmas("fra")[0].name()
            return synonym if synonym != word else word

        return word

    except Exception as e:
        logger.error(f"❌ Erreur lors de l'extraction du synonyme pour '{word}': {e}")
        return word


def replace_synonyms_selective(text):
    """
    Remplace uniquement les mots qui ne sont pas des entités nommées.
    """
    try:
        if not isinstance(text, str):
            raise TypeError("L'entrée doit être une chaîne de caractères.")

        # Chargement du modèle seulement si nécessaire
        # nlp = spacy.load("fr_core_news_md")
        doc = nlp(text)

        new_tokens = []
        for token in doc:
            if token.ent_type_:
                new_tokens.append(token.text)
            else:
                new_tokens.append(get_synonym(token.text))

        return " ".join(new_tokens)

    except Exception as e:
        logger.error(f"❌ Erreur lors du remplacement des synonymes : {e}")
        return text


def compute_embedding_similarity_from_vectors(embedding1, embedding2):
    """
    Calcule la similarité cosinus entre deux vecteurs d'embeddings.
    """
    try:
        if not isinstance(embedding1, np.ndarray) or not isinstance(embedding2, np.ndarray):
            raise TypeError("Les entrées doivent être des vecteurs numpy.")

        return cosine_similarity([embedding1], [embedding2])[0][0]

    except Exception as e:
        logger.error(f"❌ Erreur lors du calcul de similarité cosinus : {e}")
        return 0.0


def detect_paraphrase(text1, text2):
    """
    Vérifie si deux phrases sont des paraphrases.
    """
    try:
        # Chargement du modèle seulement si nécessaire
        # paraphrase_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        embedding1, embedding2 = paraphrase_model.encode([text1, text2])
        return util.pytorch_cos_sim(embedding1, embedding2).item()

    except Exception as e:
        logger.error(f"❌ Erreur lors de la détection de paraphrase : {e}")
        return 0.0
