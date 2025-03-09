# app/libs/traitement_list/step_001_upload_document.py

import os
from datetime import datetime

from ..ClassTraitement import Traitement
from ....services.document_upload_service import DocumentUploadService


class UploadDocumentTask(Traitement):
    def __init__(self):
        super().__init__("Upload Document Task", max_retries=3, retry_delay=2)
        self.upload_service = DocumentUploadService()

    def prepare(self, data):
        """
        Vérifie l'existence du fichier source, détermine son extension
        et retourne un dictionnaire contenant le chemin source, l'extension,
        le nom du fichier original, la date d'upload et un flag indiquant 
        si le document nécessite un traitement OCR.

        :param data: Dict contenant le chemin du fichier source et les métadonnées
        :return: dict avec les informations nécessaires pour l'exécution
        :raises FileNotFoundError: Si le fichier n'existe pas.
        """
        if isinstance(data, dict):
            source_file = data.get('temp_file_path')
            filename = data.get('filename', '')
            upload_date = data.get('upload_date', '')
        else:
            source_file = data
            filename = os.path.basename(data)
            upload_date = datetime.now().isoformat()

        if not os.path.exists(source_file):
            raise FileNotFoundError("Le fichier source n'existe pas.")
        print(f"{self.name} - Fichier source vérifié : {source_file}")

        extension = os.path.splitext(source_file)[1].lower()
        # Pour notre cas, on considère que les fichiers texte (.txt, .json) n'ont pas besoin d'OCR
        requires_ocr = True
        if extension in [".txt", ".json"]:
            requires_ocr = False

        # On retourne un dictionnaire préparé pour l'étape d'exécution
        return {
            "source_file": source_file,
            "extension": extension,
            "requires_ocr": requires_ocr,
            "filename": filename,
            "upload_date": upload_date
        }

    def execute(self, prepared_data):
        """
        Dépose le document en appelant le service d'upload, puis récupère le chemin déposé.
        Retourne un dictionnaire comprenant l'identifiant du document, le chemin déposé,
        l'extension et le flag 'requires_ocr' pour la suite du pipeline.

        :param prepared_data: dict contenant les informations du document
        :return: dict avec toutes les informations nécessaires
        """
        source_file = prepared_data["source_file"]
        # Déposer le document et récupérer l'ID
        document_id = self.upload_service.deposit_document(source_file)
        # Récupérer le chemin complet du fichier déposé
        deposited_path = self.upload_service.get_document_path(document_id)
        print(f"{self.name} - Document déposé avec succès : {deposited_path}")

        # Retourne un dictionnaire avec toutes les informations utiles pour la suite
        return {
            "document_id": document_id,
            "deposited_path": deposited_path,
            "extension": prepared_data["extension"],
            "requires_ocr": prepared_data["requires_ocr"],
            "filename": prepared_data["filename"],
            "upload_date": prepared_data["upload_date"]
        }


