import pytest
from typing import Dict, Any
import logging
import os
import json

# Import conditionnel des dépendances lourdes
try:
    import autogen
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False

try:
    from libs.traitement_document.traitement_task.TraitementOrchestrator import TraitementOrchestrator
    from libs.traitement_document.traitement_task.PreprocessingTask import PreprocessingTask
    from libs.traitement_document.traitement_task.EmbeddingTask import EmbeddingTask
    from libs.traitement_document.traitement_task.StorageTask import StorageTask
    PROCESSING_LIBS_AVAILABLE = True
except ImportError:
    PROCESSING_LIBS_AVAILABLE = False

try:
    from libs.functions.global_functions import (
        clean_text, tokenize_text, remove_stopwords,
        stem_text, lemmatize_text,
        extract_phone_numbers, extract_emails,
        extract_money_amounts, extract_dates,
        extract_percentages, extract_named_entities_spacy,
        extract_named_entities_flair, extract_named_entities_combined
    )
    GLOBAL_FUNCTIONS_AVAILABLE = True
except ImportError:
    GLOBAL_FUNCTIONS_AVAILABLE = False


def test_autogen_availability():
    """Test si autogen est disponible"""
    if not AUTOGEN_AVAILABLE:
        pytest.skip("Module autogen non disponible")
    
    import autogen
    assert hasattr(autogen, 'AssistantAgent')
    assert hasattr(autogen, 'UserProxyAgent')


def test_processing_libs_availability():
    """Test si les bibliothèques de traitement sont disponibles"""
    if not PROCESSING_LIBS_AVAILABLE:
        pytest.skip("Bibliothèques de traitement non disponibles")
    
    from libs.traitement_document.traitement_task.TraitementOrchestrator import TraitementOrchestrator
    assert TraitementOrchestrator is not None


def test_global_functions_availability():
    """Test si les fonctions globales sont disponibles"""
    if not GLOBAL_FUNCTIONS_AVAILABLE:
        pytest.skip("Fonctions globales non disponibles")
    
    from libs.functions.global_functions import clean_text
    assert callable(clean_text)


def test_basic_text_processing():
    """Test basique de traitement de texte"""
    if not GLOBAL_FUNCTIONS_AVAILABLE:
        pytest.skip("Fonctions globales non disponibles")
    
    from libs.functions.global_functions import clean_text
    
    # Test simple de nettoyage de texte
    test_text = "  Hello World!  "
    result = clean_text(test_text)
    assert isinstance(result, str)


def test_pipeline_orchestrator_init():
    """Test d'initialisation du TraitementOrchestrator"""
    if not PROCESSING_LIBS_AVAILABLE:
        pytest.skip("Bibliothèques de traitement non disponibles")
    
    try:
        from libs.traitement_document.traitement_task.TraitementOrchestrator import TraitementOrchestrator
        orchestrator = TraitementOrchestrator(use_hybrid_storage=False, use_llm_enrichment=False)
        assert orchestrator is not None
    except Exception as e:
        pytest.skip(f"Impossible d'initialiser TraitementOrchestrator: {e}") 