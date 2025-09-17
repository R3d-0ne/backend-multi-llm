import logging
import spacy
import flair
from sentence_transformers import SentenceTransformer

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    logger.info("🚀 Chargement des modèles NLP et d'embedding...")

    # Chargement des modèles une seule fois
    nlp = spacy.load("fr_dep_news_trf")
    bert_large = "dbmdz/bert-large-cased-finetuned-conll03-english"

    # Modèle d'embedding local (Sentence-Transformers)
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    logger.info("✅ Modèles chargés avec succès !")

except Exception as e:
    logger.error(f"❌ Erreur lors du chargement des modèles : {e}")
    raise RuntimeError("Impossible de charger les modèles NLP.")
