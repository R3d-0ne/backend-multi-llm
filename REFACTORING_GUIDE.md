# Services Refactorisés - Guide de Migration

Ce document décrit le refactoring des services du backend multi-LLM et explique comment migrer vers les nouvelles versions.

## 🎯 Objectifs du Refactoring

Le refactoring vise à corriger les nombreuses fonctionnalités manquantes dans `app/services/search_service.py` et les autres services, tout en gardant l'esprit du code original. Les améliorations incluent :

- **Architecture modulaire** : Séparation des responsabilités
- **Gestion d'erreurs robuste** : Exceptions structurées et gestion uniforme
- **Performance** : Mise en cache et optimisations
- **Observabilité** : Logs, métriques et health checks
- **Maintenabilité** : Code plus propre et testable

## 📁 Structure des Nouveaux Services

```
app/services/
├── base_service.py                    # Classes de base communes
├── config_service.py                  # Gestion centralisée de la configuration
├── search_components.py               # Composants modulaires de recherche
├── search_service_refactored.py       # Service de recherche refactorisé
├── embedding_service_refactored.py    # Service d'embeddings refactorisé
├── document_service_refactored.py     # Service de documents refactorisé
├── generate_service_refactored.py     # Service de génération refactorisé
└── service_compatibility.py           # Couche de compatibilité pour migration progressive
```

## 🔧 Services Refactorisés

### 1. ConfigService - Gestion Centralisée de la Configuration

**Fichier** : `config_service.py`

**Fonctionnalités** :
- Centralisation de toutes les variables d'environnement
- Configuration par sections (LLM, Search, Qdrant)
- Mise à jour dynamique des paramètres
- Validation automatique

**Exemple d'utilisation** :
```python
from app.services.config_service import config_service

# Accès à la configuration
llm_model = config_service.llm.model
search_limit = config_service.search.default_limit

# Mise à jour dynamique
config_service.update_config("llm", "temperature", 0.7)
```

### 2. BaseService - Classes de Base Communes

**Fichier** : `base_service.py`

**Fonctionnalités** :
- Gestion d'erreurs standardisée
- Validation d'entrées
- Logging structuré
- Pattern de service avec cache (CacheableService)
- Health checks

**Classes disponibles** :
- `BaseService` : Service de base avec gestion d'erreurs
- `CacheableService` : Service avec cache automatique
- `ServiceResponse` : Format de réponse standardisé
- `ServiceError`, `ValidationError`, `ExternalServiceError` : Exceptions typées

### 3. SearchService Refactorisé

**Fichier** : `search_service_refactored.py`

**Améliorations** :
- Architecture modulaire avec composants spécialisés
- Cache automatique des résultats (30 minutes)
- Gestion d'erreurs robuste
- Health checks complets
- Configuration flexible

**Composants modulaires** (`search_components.py`) :
- `DocumentProcessor` : Extraction de texte et traitement de documents
- `KeywordBooster` : Boost des scores basé sur les mots-clés
- `LLMReranker` : Réordonnancement des résultats par LLM
- `SearchResultFilter` : Filtrage des résultats par métadonnées

**Exemple d'utilisation** :
```python
from app.services.search_service_refactored import search_service_refactored

# Recherche hybride
result = search_service_refactored.hybrid_search(
    query="recherche de test",
    limit=10,
    use_llm_reranking=True,
    boost_keywords=True
)

# Health check
health = search_service_refactored.health_check()
```

### 4. EmbeddingService Refactorisé

**Fichier** : `embedding_service_refactored.py`

**Améliorations** :
- Cache de 2 heures pour éviter les recalculs
- Traitement par lots pour grandes collections
- Gestion de timeout et retry
- Validation robuste des embeddings
- Métriques de performance

**Nouvelles fonctionnalités** :
```python
from app.services.embedding_service_refactored import embedding_service_refactored

# Embedding simple avec cache
embedding = embedding_service_refactored.get_embedding("texte")

# Traitement par lots
embeddings = embedding_service_refactored.get_embeddings_batch(
    texts=["texte1", "texte2", "texte3"],
    batch_size=10
)

# Dimension du modèle
dimension = embedding_service_refactored.get_embedding_dimension()
```

### 5. DocumentService Refactorisé

**Fichier** : `document_service_refactored.py`

**Améliorations** :
- Pagination efficace pour le listage
- Recherche et filtrage avancés
- Statistiques de collection
- Validation complète des entrées
- Gestion d'erreurs contextuelle

**Nouvelles fonctionnalités** :
```python
from app.services.document_service_refactored import document_service_refactored

# Listage avec pagination
result = document_service_refactored.list_documents(limit=100, offset=0)

# Recherche par critères
search_result = document_service_refactored.search_documents({
    "file_type": "pdf",
    "upload_date": "2024"
})

# Statistiques
stats = document_service_refactored.get_document_statistics()
```

### 6. GenerateService Refactorisé

**Fichier** : `generate_service_refactored.py`

**Améliorations** :
- Orchestration améliorée du workflow
- Validation complète des requêtes
- Monitoring des dépendances
- Gestion d'erreurs contextuelle
- Configuration flexible

**Nouvelles fonctionnalités** :
```python
from app.services.generate_service_refactored import generate_service_refactored

# Validation de requête
validation = generate_service_refactored.validate_generate_request({
    "current_message": "Bonjour",
    "additional_info": "Test"
})

# Génération avec gestion d'erreurs
result = generate_service_refactored.generate_response(
    discussion_id=None,
    current_message="Bonjour",
    settings_id="custom-settings"
)
```

## 🔄 Migration Progressive

### Couche de Compatibilité

**Fichier** : `service_compatibility.py`

La couche de compatibilité permet une migration progressive sans casser l'API existante :

```python
from app.services.service_compatibility import migration_manager

# Basculer vers les services refactorisés
migration_manager.migrate_service('search', True)
migration_manager.migrate_service('embedding', True)

# Vérifier le statut de migration
status = migration_manager.get_migration_status()

# Utiliser les services via les wrappers de compatibilité
from app.services.service_compatibility import (
    search_service_compat,
    embedding_service_compat
)

# L'API reste identique, mais utilise les services refactorisés
result = search_service_compat.hybrid_search("test")
embedding = embedding_service_compat.get_embedding("text")
```

### Endpoints API v2

**Fichier** : `app/api/refactored_endpoints.py`

Nouveaux endpoints qui utilisent directement les services refactorisés :

```
GET    /api/v2/health                    # État de santé global
POST   /api/v2/search/hybrid             # Recherche hybride
POST   /api/v2/embeddings/batch          # Embeddings par lot
GET    /api/v2/documents/stats           # Statistiques des documents
POST   /api/v2/migration/toggle-service  # Bascule de service
```

## 🧪 Tests et Validation

### Script de Test

**Fichier** : `test_refactored_services.py`

Script de démonstration pour tester tous les services refactorisés :

```bash
python test_refactored_services.py
```

Le script teste :
- Configuration centralisée
- Services d'embeddings avec cache
- Gestion des documents avec pagination
- Recherche hybride avec composants modulaires
- Service de génération avec validation
- Couche de compatibilité
- Comparaison de performance

### Health Checks

Chaque service fournit des health checks détaillés :

```python
# Health check global
from app.services.service_compatibility import get_service_health_summary
health = get_service_health_summary()

# Health check par service
search_health = search_service_refactored.health_check()
embedding_health = embedding_service_refactored.health_check()
```

## 📊 Métriques et Monitoring

### Logs Structurés

Tous les services utilisent un logging structuré avec niveaux appropriés :

```python
# Les logs incluent automatiquement :
# - Service name
# - Log level
# - Timestamp
# - Context information
```

### Statistiques de Performance

- **Cache hit/miss ratios** pour les embeddings et recherche
- **Temps de réponse** par opération
- **Taux d'erreur** par service
- **Utilisation mémoire** du cache

### Métriques de Santé

- **Disponibilité** des services externes (LLM, Qdrant)
- **Latence** des appels API
- **État** des dépendances
- **Capacité** de traitement

## 🚀 Plan de Déploiement

### Phase 1 : Déploiement des Services Refactorisés ✅
- [x] Déployer les nouveaux services en parallèle
- [x] Activer la couche de compatibilité
- [x] Tester avec les endpoints v2

### Phase 2 : Migration Progressive
- [ ] Basculer les services un par un
- [ ] Monitorer les performances
- [ ] Valider la compatibilité

### Phase 3 : Nettoyage
- [ ] Supprimer les anciens services
- [ ] Mettre à jour la documentation
- [ ] Optimiser les performances

## 🔧 Configuration

### Variables d'Environnement

Les nouveaux services utilisent des variables d'environnement pour la configuration :

```env
# LLM Configuration
LLM_BASE_URL=http://host.docker.internal:11434
LLM_MODEL=llama3.1:8b
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=1024
LLM_TIMEOUT=30

# Search Configuration
DEFAULT_COLLECTION=documents
DEFAULT_SEARCH_LIMIT=10
MAX_DOCUMENT_LENGTH=2000
KEYWORD_BOOST_MAX=0.4
LLM_SCORE_WEIGHT=0.6

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_TIMEOUT=30
```

## 🛠️ Développement et Extension

### Ajouter un Nouveau Service

1. Hériter de `BaseService` ou `CacheableService`
2. Implémenter la méthode `health_check()`
3. Utiliser les patterns de gestion d'erreurs
4. Ajouter des logs structurés

```python
from app.services.base_service import BaseService, LogLevel, ServiceResponse

class MonNouveauService(BaseService):
    def __init__(self):
        super().__init__("MonNouveauService")
    
    def ma_methode(self, param: str) -> ServiceResponse:
        def operation():
            # Logique métier
            return {"result": "success"}
        
        return self.safe_execute(operation, "Erreur dans ma_methode")
    
    def health_check(self) -> ServiceResponse:
        # Vérifications de santé
        return ServiceResponse.success_response({"status": "healthy"})
```

### Ajouter des Métriques

Les services supportent l'ajout de métriques personnalisées via le système de logging structuré.

## 📚 Documentation API

La documentation complète des endpoints est disponible via FastAPI :
- `/docs` - Documentation Swagger
- `/redoc` - Documentation ReDoc

## 🤝 Contribution

Pour contribuer aux services refactorisés :

1. Respecter l'architecture modulaire
2. Utiliser les classes de base communes
3. Ajouter des tests appropriés
4. Documenter les nouvelles fonctionnalités
5. Maintenir la compatibilité descendante

## 📞 Support

Pour toute question sur la migration ou les services refactorisés, consulter :
- Cette documentation
- Les commentaires dans le code
- Les tests de démonstration
- Les health checks pour diagnostiquer les problèmes