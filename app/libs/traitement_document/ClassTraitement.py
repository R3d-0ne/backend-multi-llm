# ClassTraitement.py
from abc import ABC, abstractmethod
import logging
import time

# Configuration du logger
logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class Traitement(ABC):
    def __init__(self, name: str, max_retries=3, retry_delay=2):
        self.name = name
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    @abstractmethod
    def prepare(self, data):
        """
        Transforme les données brutes en données préparées.
        Cette méthode doit renvoyer les données prêtes à être traitées par execute().
        """
        pass

    @abstractmethod
    def execute(self, prepared_data):
        """
        Traite les données préparées et renvoie le résultat.
        """
        pass

    def run(self, data):
        """
        Appelle prepare() pour préparer les données, puis execute() pour réaliser le traitement.
        En cas d'erreur dans execute(), il réessaie jusqu'à max_retries.
        """
        start_time = time.time()
        try:
            prepared_data = self.prepare(data)
            logging.info(f"{self.name} - Préparation terminée")
        except Exception as e:
            logging.error(f"{self.name} - Erreur lors de la préparation : {e}")
            return data

        retries = 0
        while retries < self.max_retries:
            try:
                result = self.execute(prepared_data)
                total_time = time.time() - start_time
                logging.info(f"{self.name} - Exécution réussie en {total_time:.2f}s")
                return result
            except Exception as e:
                retries += 1
                logging.warning(f"{self.name} - Erreur ({retries}/{self.max_retries}) : {e}")
                logging.info(f"{self.name} - Échec tentative {retries}/{self.max_retries}, attente {self.retry_delay}s...")
                time.sleep(self.retry_delay)

        logging.error(f"{self.name} - Échec final après {self.max_retries} tentatives.")
        return data
