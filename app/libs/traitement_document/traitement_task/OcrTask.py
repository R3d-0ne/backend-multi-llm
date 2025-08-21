# app/libs/traitement_document/traitement_task/step_002_ocr.py

import os
import logging

# Conditional imports for OCR functionality
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

from ..ClassTraitement import Traitement

logger = logging.getLogger(__name__)


class OCRTask(Traitement):
    def __init__(self):
        super().__init__("OCR Task", max_retries=3, retry_delay=2)

    def prepare(self, data):
        """
        Attend que data soit un dictionnaire contenant :
          - 'deposited_path': chemin du fichier déposé
          - 'requires_ocr': booléen indiquant si OCR est nécessaire
          - 'extension' et 'document_id'

        Si requires_ocr est False, le document est déjà en texte et sera lu directement.
        Sinon, retourne le chemin pour effectuer l'OCR.
        """
        if not isinstance(data, dict):
            raise ValueError("OCRTask attend un dictionnaire en entrée.")

        deposited_path = data.get("deposited_path")
        requires_ocr = data.get("requires_ocr", True)

        if not deposited_path or not os.path.exists(deposited_path):
            raise FileNotFoundError("Le fichier déposé est introuvable.")

        logger.info(f"{self.name} - Fichier vérifié : {deposited_path}")
        return data

    def execute(self, prepared_data):
        """
        Exécute l'OCR ou la lecture directe selon le flag 'requires_ocr'.
        - Si OCR est requis (ex. PDF), convertit le PDF en images, extrait le texte via pytesseract,
          enregistre le texte extrait dans un fichier .txt dans le dossier d'upload, et ajoute les infos.
        - Sinon, lit directement le contenu du fichier texte.

        Retourne un dictionnaire enrichi contenant :
          - 'document_id', 'deposited_path', 'extension', 'requires_ocr' (issus de l'étape précédente)
          - 'extracted_text': le texte extrait
          - 'text_file_path': chemin du fichier texte (créé pour un PDF ou identique au deposited_path si déjà en texte)
          - 'image_paths': liste des chemins des images extraites
        """
        # Copie du dictionnaire pour le résultat final
        result = prepared_data.copy()
        file_path = prepared_data["deposited_path"]

        if prepared_data["requires_ocr"]:
            logger.info(f"{self.name} - Début de la conversion du PDF en images pour OCR : {file_path}")
            
            if not PDF2IMAGE_AVAILABLE or not PYTESSERACT_AVAILABLE:
                logger.error(f"{self.name} - OCR requis mais dépendances manquantes (pdf2image: {PDF2IMAGE_AVAILABLE}, pytesseract: {PYTESSERACT_AVAILABLE})")
                # Retour avec texte vide au lieu de lever une exception
                text_file_path = file_path.replace(os.path.splitext(file_path)[1], ".txt")
                with open(text_file_path, 'w', encoding='utf-8') as f:
                    f.write("")
                prepared_data.update({
                    "extracted_text": "",
                    "text_file_path": text_file_path,
                    "image_paths": []
                })
                logger.warning(f"{self.name} - OCR ignoré, fichier texte vide créé : {text_file_path}")
                return prepared_data
            
            try:
                images = convert_from_path(file_path, dpi=300)
            except Exception as e:
                logger.error(f"{self.name} - Erreur lors de la conversion du PDF : {e}")
                raise e

            logger.info(f"{self.name} - Conversion terminée, pages extraites : {len(images)}")
            extracted_text = ""
            image_paths = []
            
            # Créer un dossier unique pour le document avec son ID
            document_dir = os.path.join(os.path.dirname(file_path), prepared_data['document_id'])
            images_dir = os.path.join(document_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            
            for idx, image in enumerate(images):
                try:
                    logger.info(f"{self.name} - Extraction OCR sur la page {idx + 1}")
                    page_text = pytesseract.image_to_string(image, lang='eng')
                    extracted_text += page_text + "\n"
                    
                    # Sauvegarder l'image dans le dossier unique du document
                    image_path = os.path.join(images_dir, f"page_{idx + 1}.png")
                    image.save(image_path, "PNG")
                    image_paths.append(image_path)
                    logger.info(f"{self.name} - Image de la page {idx + 1} sauvegardée : {image_path}")
                except Exception as e:
                    logger.error(f"{self.name} - Erreur lors de l'extraction OCR sur la page {idx + 1} : {e}")
                    raise e

            # Enregistrer le texte extrait dans le dossier unique du document
            text_file_path = os.path.join(
                document_dir,
                f"{prepared_data['document_id']}.txt"
            )
            try:
                with open(text_file_path, "w", encoding="utf-8") as f:
                    f.write(extracted_text)
                logger.info(f"{self.name} - Texte extrait sauvegardé dans : {text_file_path}")
            except Exception as e:
                logger.error(f"{self.name} - Erreur lors de l'enregistrement du fichier texte : {e}")
                raise e

            result["extracted_text"] = extracted_text
            result["text_file_path"] = text_file_path
            result["image_paths"] = image_paths

        else:
            # Lecture directe du texte (fichier déjà en format texte)
            logger.info(f"{self.name} - Lecture directe du fichier texte : {file_path}")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    extracted_text = f.read()
            except Exception as e:
                logger.error(f"{self.name} - Erreur lors de la lecture du fichier texte : {e}")
                raise e
            result["extracted_text"] = extracted_text
            result["text_file_path"] = file_path

        return {
            "document_id": result.get("document_id"),
            "deposited_path": result.get("deposited_path"),
            "extension": result.get("extension"),
            "requires_ocr": result.get("requires_ocr"),
            "extracted_text": result.get("extracted_text"),
            "text_file_path": result.get("text_file_path"),
            "image_paths": result.get("image_paths", []),
            "filename": result.get("filename"),  # Préserver le nom du fichier
            "upload_date": result.get("upload_date")  # Préserver la date d'upload
        }
