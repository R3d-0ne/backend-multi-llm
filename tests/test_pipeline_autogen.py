import autogen
from typing import Dict, Any
import logging
import os
import json

from libs.traitement_document.traitement_task.TraitementOrchestrator import TraitementOrchestrator
from libs.traitement_document.traitement_task.PreprocessingTask import PreprocessingTask
from libs.traitement_document.traitement_task.EmbeddingTask import EmbeddingTask
from libs.traitement_document.traitement_task.StorageTask import StorageTask
from libs.functions.global_functions import (
    clean_text, tokenize_text, remove_stopwords,
    stem_text, lemmatize_text,
    extract_phone_numbers, extract_emails,
    extract_money_amounts, extract_dates,
    extract_percentages, extract_named_entities_spacy,
    extract_named_entities_flair, extract_named_entities_combined
) 