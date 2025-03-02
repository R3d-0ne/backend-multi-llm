import logging
import spacy
import flair
from sentence_transformers import SentenceTransformer
from spellchecker import SpellChecker

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    logger.info("🚀 Chargement des modèles NLP...")

    # Chargement des modèles une seule fois
    # nlp = spacy.load("fr_core_news_md")
    nlp = spacy.load("fr_dep_news_trf")
    spell = SpellChecker(language='fr')
    bert_large = "dbmdz/bert-large-cased-finetuned-conll03-english"

    nlp_flair = flair.models.SequenceTagger.load("fr-ner")
    paraphrase_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    minilm_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    logger.info("✅ Modèles chargés avec succès !")

except Exception as e:
    logger.error(f"❌ Erreur lors du chargement des modèles : {e}")
    raise RuntimeError("Impossible de charger les modèles NLP.")
