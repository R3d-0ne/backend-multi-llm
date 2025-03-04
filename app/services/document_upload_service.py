import logging
import os
import uuid
import shutil

UPLOAD_FOLDER = "data/uploads"
logger = logging.getLogger(__name__)


class DocumentUploadService:
    """
    Service de dépôt de documents.
    Permet de déposer, récupérer et supprimer un document.
    """

    def __init__(self, upload_folder: str = UPLOAD_FOLDER):
        self.collection_name = "documents"
        logger.info(f"Utilisation de la collection '{self.collection_name}' pour les documents.")
        self.upload_folder = upload_folder
        os.makedirs(self.upload_folder, exist_ok=True)

    def deposit_document(self, source_file: str) -> str:
        """
        Dépose le fichier source dans le dossier de dépôt et renvoie l'identifiant unique du document.

        :param source_file: Chemin du fichier à déposer.
        :return: Identifiant unique du document.
        :raises FileNotFoundError: Si le fichier source n'existe pas.
        :raises ValueError: Si le format du fichier n'est pas supporté.
        """
        # Vérification de l'existence du fichier source
        if not os.path.exists(source_file):
            logger.error("Le fichier source n'existe pas: %s", source_file)
            raise FileNotFoundError("Le fichier n'existe pas.")

        # Détermination de l'extension et validation du format
        extension = os.path.splitext(source_file)[1].lower()
        allowed_extensions = [".pdf", ".txt", ".json"]
        if extension not in allowed_extensions:
            logger.error("Format non supporté pour le fichier %s. Extensions autorisées: %s", source_file,
                         allowed_extensions)
            raise ValueError("Format non supporté. Formats acceptés : PDF, TXT, JSON.")

        # Génération d'un identifiant unique pour le document
        document_id = str(uuid.uuid4())
        
        # Création du dossier unique pour le document
        document_dir = os.path.join(self.upload_folder, document_id)
        os.makedirs(document_dir, exist_ok=True)
        
        # Copie du fichier dans le dossier unique
        destination = os.path.join(document_dir, f"original{extension}")

        # Tentative de copie du fichier dans le dossier de dépôt
        try:
            shutil.copy(source_file, destination)
            logger.info("Document déposé avec succès : %s", destination)
        except Exception as e:
            logger.error("Erreur lors du dépôt du document %s: %s", source_file, e)
            raise e

        return document_id

    def get_document_path(self, document_id: str) -> str:
        """
        Recherche le chemin du document en fonction de son identifiant.

        :param document_id: Identifiant unique du document.
        :return: Chemin complet du fichier déposé.
        :raises FileNotFoundError: Si le document n'est pas trouvé.
        """
        document_dir = os.path.join(self.upload_folder, document_id)
        if not os.path.exists(document_dir):
            raise FileNotFoundError("Document non trouvé.")
            
        # Chercher le fichier original dans le dossier du document
        for filename in os.listdir(document_dir):
            if filename.startswith("original"):
                return os.path.join(document_dir, filename)
        raise FileNotFoundError("Document non trouvé.")

    def delete_document(self, document_id: str) -> bool:
        """
        Supprime le document correspondant à l'identifiant donné.

        :param document_id: Identifiant unique du document.
        :return: True si la suppression a réussi.
        :raises FileNotFoundError: Si le document n'est pas trouvé ou une erreur survient.
        """
        try:
            path = self.get_document_path(document_id)
            os.remove(path)
            print(f"Document {document_id} supprimé avec succès.")
            return True
        except Exception as e:
            raise FileNotFoundError(f"Erreur lors de la suppression: {e}")
