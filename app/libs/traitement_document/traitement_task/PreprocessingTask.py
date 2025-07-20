import logging
from ..ClassTraitement import Traitement
from ...functions.global_functions import (
    clean_text, tokenize_text, remove_stopwords,
    stem_text, lemmatize_text,
    extract_phone_numbers,
    extract_emails, extract_money_amounts, extract_dates, extract_percentages, extract_entities_advanced
)

logger = logging.getLogger(__name__)


class PreprocessingTask(Traitement):
    def __init__(self):
        """
        Initialise la tâche de prétraitement en héritant de `Traitement`.
        """
        super().__init__("Preprocessing Task", max_retries=3, retry_delay=2)
        self.data = {}

    def prepare(self, data):
        """
        Prépare les données pour le traitement.
        - Vérifie si le texte est bien une chaîne de caractères.
        - Initialise les valeurs nécessaires pour le traitement.

        :param data: dict contenant la clé "extracted_text" (texte brut).
        :return: dict avec le texte brut à traiter.
        """
        if not isinstance(data, dict) or "extracted_text" not in data:
            raise ValueError("PreprocessingTask attend un dictionnaire contenant la clé 'extracted_text'.")

        text = data["extracted_text"]
        if not isinstance(text, str) or len(text.strip()) == 0:
            raise ValueError("Le texte fourni est invalide.")

        logger.info(f"{self.name} - Texte prêt pour le prétraitement.")
        # Stocker le texte original pour traçabilité
        return {
            "document_id": data.get("document_id"),
            "deposited_path": data.get("deposited_path"),
            "extension": data.get("extension"),
            "text_file_path": data.get("text_file_path"),
            "requires_ocr": data.get("requires_ocr"),
            "original_text": text,
            "has_tables": data.get("has_tables"),
            "tables_count": data.get("tables_count"),
            "pages_with_tables": data.get("pages_with_tables"),
            "details_tables": data.get("details_tables"),
            "image_paths": data.get("image_paths", []),
            "filename": data.get("filename"),  # Préserver le nom du fichier
            "upload_date": data.get("upload_date")  # Préserver la date d'upload
        }


    def execute(self, prepared_data):
        """
        Applique le pipeline de prétraitement et renvoie un dictionnaire complet contenant :
          - document_id, deposited_path, extension, requires_ocr
          - original_text, cleaned_text, tokens, tokens_no_stopwords,
            stemmed_tokens, lemmatized_tokens, named_entities,
            synonym_replaced_text, phone_numbers, emails,
            money_amounts, dates, percentages

        Les résultats de chaque étape s'ajoutent aux clés existantes.

        :param prepared_data: dict contenant au minimum :
             "document_id", "deposited_path", "extension", "requires_ocr" et "original_text"
        :return: dict regroupant toutes les informations issues du prétraitement.
        """
        if not prepared_data or "original_text" not in prepared_data:
            raise ValueError("Données invalides pour le traitement.")

        text = prepared_data["original_text"]
        logger.info(f"{self.name} - Démarrage du prétraitement...")

        try:
            # On part d'une copie du dictionnaire de base
            result = prepared_data.copy()

            # 1️⃣ Nettoyage du texte
            cleaned_text = clean_text(text)
            result["cleaned_text"] = cleaned_text
            logger.info("✅ Texte nettoyé.")


            # 2️⃣ Tokenisation
            tokens = tokenize_text(cleaned_text)
            result["tokens"] = tokens
            logger.info("✅ Tokenisation effectuée.")

            # 3️⃣ Suppression des stopwords
            tokens_no_stopwords = remove_stopwords(tokens)
            result["tokens_no_stopwords"] = tokens_no_stopwords
            logger.info("✅ Stopwords supprimés.")

            # 4️⃣ Stemmatisation
            stemmed_tokens = stem_text(tokens_no_stopwords)
            result["stemmed_tokens"] = stemmed_tokens
            logger.info("✅ Stemmatisation effectuée.")

            # 5️⃣ Lemmatisation
            lemmatized_tokens = lemmatize_text(tokens_no_stopwords)
            result["lemmatized_tokens"] = lemmatized_tokens
            logger.info("✅ Lemmatisation effectuée.")


            # 6️⃣.1️⃣ Extraction des entités nommées (NER)
            named_entities_bert = extract_entities_advanced(cleaned_text)
            result["named_entities_bert"] = named_entities_bert
            logger.info("✅ Extraction des entités terminée bert.")


            # 8️⃣ Extraction des numéros de téléphone et emails
            result["phone_numbers"] = extract_phone_numbers(cleaned_text)
            result["emails"] = extract_emails(cleaned_text)
            logger.info("✅ Extraction des numéros de téléphone et emails effectuée.")

            # 9️⃣ Extraction des montants financiers, dates et pourcentages
            result["money_amounts"] = extract_money_amounts(cleaned_text)
            result["dates"] = extract_dates(cleaned_text)
            result["percentages"] = extract_percentages(cleaned_text)
            logger.info("✅ Extraction des montants financiers, dates et pourcentages effectuée.")

            logger.info(f"{self.name} - Prétraitement terminé avec succès !")

            # Retour final respectant le format demandé
            return {
                "document_id": result.get("document_id"),
                "deposited_path": result.get("deposited_path"),
                "extension": result.get("extension"),
                "requires_ocr": result.get("requires_ocr"),
                "original_text": result.get("original_text"),
                "text_file_path": result.get("text_file_path"),
                "cleaned_text": result.get("cleaned_text"),
                "tokens": result.get("tokens"),
                "tokens_no_stopwords": result.get("tokens_no_stopwords"),
                "stemmed_tokens": result.get("stemmed_tokens"),
                "lemmatized_tokens": result.get("lemmatized_tokens"),
                "named_entities_bert": result.get("named_entities_bert"),
                "phone_numbers": result.get("phone_numbers"),
                "emails": result.get("emails"),
                "money_amounts": result.get("money_amounts"),
                "dates": result.get("dates"),
                "percentages": result.get("percentages"),
                "has_tables": result.get("has_tables"),
                "tables_count": result.get("tables_count"),
                "pages_with_tables": result.get("pages_with_tables"),
                "details_tables": result.get("details_tables"),
                "image_paths": result.get("image_paths", []),
                "filename": result.get("filename"),
                "upload_date": result.get("upload_date")
            }

        except Exception as e:
            logger.error(f"{self.name} - Erreur lors du prétraitement : {e}")
            raise e
