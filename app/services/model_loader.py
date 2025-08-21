import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Conditional imports for heavy dependencies
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spacy not available - NLP features will be disabled")

try:
    import flair
    FLAIR_AVAILABLE = True
except ImportError:
    FLAIR_AVAILABLE = False
    logger.warning("flair not available - some NLP features will be disabled")

class OllamaModel:
    def __init__(self, model_name: str, port: int = 11434):
        self.base_url = f"http://host.docker.internal:{port}"
        self.model_name = model_name

try:
    logger.info("🚀 Chargement des modèles NLP...")

    # Initialisation des variables par défaut
    nlp = None
    bert_large = None
    
    # Chargement des modèles spaCy seulement si disponible
    if SPACY_AVAILABLE:
        try:
            nlp = spacy.load("fr_core_news_sm")
        except OSError:
            logger.warning("Modèle fr_core_news_sm non trouvé, tentative avec fr_core_news_md")
            try:
                nlp = spacy.load("fr_core_news_md")
            except OSError:
                logger.warning("Modèle fr_core_news_md non trouvé, utilisation du modèle anglais par défaut")
                try:
                    nlp = spacy.load("en_core_web_sm")
                except OSError:
                    logger.warning("Aucun modèle spaCy disponible")
                    nlp = None
    
    if FLAIR_AVAILABLE:
        bert_large = "dbmdz/bert-large-cased-finetuned-conll03-english"
    
    # Configuration du modèle d'embedding (port 11434 par défaut d'Ollama)
    minilm_model = OllamaModel("nomic-embed-text:latest", port=11434)

    logger.info("✅ Modèles chargés avec succès !")

except Exception as e:
    logger.error(f"❌ Erreur lors du chargement des modèles : {e}")
    # Ne pas lever d'exception pour permettre à l'app de démarrer
    logger.warning("Application démarrée en mode dégradé sans modèles NLP")
    nlp = None
    bert_large = None
    minilm_model = OllamaModel("nomic-embed-text:latest", port=11434)
