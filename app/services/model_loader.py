import logging
import spacy
import flair

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaModel:
    def __init__(self, model_name: str, port: int = 11434):
        self.base_url = f"http://host.docker.internal:{port}"
        self.model_name = model_name

try:
    logger.info("🚀 Chargement des modèles NLP...")

    # Chargement des modèles une seule fois - utiliser le modèle français disponible
    try:
        nlp = spacy.load("fr_core_news_sm")
    except OSError:
        logger.warning("Modèle fr_core_news_sm non trouvé, tentative avec fr_core_news_md")
        try:
            nlp = spacy.load("fr_core_news_md")
        except OSError:
            logger.warning("Modèle fr_core_news_md non trouvé, utilisation du modèle anglais par défaut")
            nlp = spacy.load("en_core_web_sm")
    
    bert_large = "dbmdz/bert-large-cased-finetuned-conll03-english"
    
    # Configuration du modèle d'embedding (port 11434 par défaut d'Ollama)
    minilm_model = OllamaModel("nomic-embed-text:latest", port=11434)

    logger.info("✅ Modèles chargés avec succès !")

except Exception as e:
    logger.error(f"❌ Erreur lors du chargement des modèles : {e}")
    raise RuntimeError("Impossible de charger les modèles NLP.")
