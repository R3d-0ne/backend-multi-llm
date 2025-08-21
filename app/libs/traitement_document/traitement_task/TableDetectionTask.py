import os
import logging

# Conditional import for camelot
try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False

from ..ClassTraitement import Traitement

logger = logging.getLogger(__name__)


class TableDetectionTask(Traitement):
    def __init__(self):
        # Nom et paramètres de base pour les tentatives
        super().__init__("Table Detection Task", max_retries=2, retry_delay=2)

    def prepare(self, data):
        """
        Attend que data soit un dictionnaire contenant au minimum :
          - 'deposited_path': chemin du fichier (PDF) déposé
          - 'document_id': identifiant unique du document
          - éventuellement 'extension', etc.

        Vérifie également que le fichier existe.
        """
        if not isinstance(data, dict):
            raise ValueError("TableDetectionTask attend un dictionnaire en entrée.")

        deposited_path = data.get("deposited_path")
        if not deposited_path or not os.path.exists(deposited_path):
            raise FileNotFoundError(f"Le fichier PDF '{deposited_path}' est introuvable.")

        logger.info(f"{self.name} - Fichier vérifié : {deposited_path}")
        return data

    def execute(self, prepared_data):
        """
        Exécute la détection des tableaux dans le PDF, en utilisant Camelot.

        Retourne un dictionnaire mis à jour contenant :
          - 'document_id', 'deposited_path' (issus de la préparation)
          - 'has_tables': bool indiquant la présence de tableaux
          - 'tables_count': nombre total de tableaux détectés
          - 'pages_with_tables': liste des pages où des tableaux ont été trouvés
          - 'details_tables': informations supplémentaires par table (index, shape, etc.)
        """
        result = prepared_data.copy()
        file_path = prepared_data["deposited_path"]

        logger.info(f"{self.name} - Début de la détection de tableaux dans le PDF : {file_path}")

        try:
            if not CAMELOT_AVAILABLE:
                logger.warning(f"{self.name} - camelot non disponible - détection de tables désactivée")
                prepared_data.update({
                    "has_tables": False,
                    "tables_count": 0,
                    "pages_with_tables": [],
                    "details_tables": []
                })
                return prepared_data
                
            # Lecture du PDF avec Camelot
            # - flavor="lattice" si le PDF contient des lignes de table nettes
            # - flavor="stream" si la structure est moins évidente
            tables = camelot.read_pdf(file_path, pages="1-end", flavor="lattice")

            tables_count = len(tables)
            logger.info(f"{self.name} - {tables_count} tableau(x) détecté(s) dans le PDF.")

            # Pages concernées (Camelot stocke la page dans tables[i].page)
            pages_with_tables = sorted({int(t.page) for t in tables})

            # Infos détaillées sur chaque table (dimension, page, etc.)
            details = []
            for i, t in enumerate(tables):
                details.append({
                    "index": i,
                    "page": t.page,
                    "shape": t.df.shape,  # (lignes, colonnes)
                    "data": t.df.values.tolist()  # contenu brut si besoin
                })

            # Si tables_count > 0, on considère qu'il y a des tableaux
            has_tables = (tables_count > 0)

            # On enrichit le dictionnaire de résultat
            result["has_tables"] = has_tables
            result["tables_count"] = tables_count
            result["pages_with_tables"] = pages_with_tables
            result["details_tables"] = details

            logger.info(f"{self.name} - Détection de tableaux terminée avec succès !")
            return result

        except Exception as e:
            logger.error(f"{self.name} - Erreur lors de la détection de tableaux : {e}")
            raise e
