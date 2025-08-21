# Guide d'Utilisation des Services Refactorisés

## Vue d'ensemble

Ce guide explique comment utiliser les nouveaux services refactorisés qui ont été intégrés dans l'application backend-multi-llm. Les services refactorisés offrent des fonctionnalités améliorées avec une meilleure gestion d'erreurs, des performances optimisées et un système de migration progressive.

## Nouveaux Endpoints API v2

### Endpoints de Santé et Monitoring

#### `GET /api/v2/health`
Vérifie l'état de santé de tous les services refactorisés.

**Réponse :**
```json
{
  "status": "success",
  "data": {
    "migration_status": {
      "search": true,
      "embedding": true,
      "document": true,
      "generate": true
    },
    "health_checks": {
      "search": {"status": "healthy"},
      "embedding": {"status": "healthy"},
      "document": {"status": "healthy"},
      "generate": {"status": "healthy"}
    }
  },
  "message": "État de santé des services récupéré avec succès"
}
```

### Endpoints de Recherche

#### `POST /api/v2/search/hybrid`
Recherche hybride améliorée avec les services refactorisés.

**Corps de la requête :**
```json
{
  "query": "votre recherche",
  "limit": 10,
  "filters": {
    "has_tables": true,
    "has_dates": true
  },
  "use_llm_reranking": true,
  "boost_keywords": true,
  "generate_answer": false
}
```

#### `POST /api/v2/search/with-generation`
Recherche avec génération automatique de réponse.

**Paramètres :**
- `query` (string): Requête de recherche
- `limit` (int): Nombre de résultats (défaut: 5)
- `filters` (object): Filtres optionnels

### Endpoints d'Embeddings

#### `POST /api/v2/embeddings/single`
Génère l'embedding d'un seul texte.

**Paramètres :**
- `text` (string): Texte à convertir
- `use_cache` (bool): Utiliser le cache (défaut: true)

#### `POST /api/v2/embeddings/batch`
Génère des embeddings pour plusieurs textes.

**Corps de la requête :**
```json
{
  "texts": ["texte 1", "texte 2", "texte 3"],
  "use_cache": true,
  "batch_size": 10
}
```

#### `GET /api/v2/embeddings/stats`
Obtient les statistiques du cache d'embeddings.

### Endpoints de Documents

#### `GET /api/v2/documents/stats`
Obtient les statistiques des documents.

#### `GET /api/v2/documents`
Liste tous les documents avec pagination.

#### `GET /api/v2/documents/{document_id}`
Récupère un document spécifique.

#### `POST /api/v2/documents/search`
Recherche avancée dans les documents.

### Endpoints de Génération

#### `POST /api/v2/generate/validate`
Valide les paramètres de génération.

#### `GET /api/v2/generate/stats`
Obtient les statistiques du service de génération.

## Migration Progressive

### Endpoints de Migration

#### `GET /api/v2/migration/status`
Obtient l'état de migration de tous les services.

#### `POST /api/v2/migration/toggle-service`
Bascule un service entre sa version originale et refactorisée.

**Corps de la requête :**
```json
{
  "service_name": "search",
  "use_refactored": true
}
```

**Services disponibles :** `search`, `embedding`, `document`, `generate`

### Endpoints dans les Routes Existantes

#### `GET /services/migration-status`
Alternative dans les routes v1 pour obtenir le statut de migration.

#### `POST /services/toggle-migration`
Alternative dans les routes v1 pour basculer les services.

## Utilisation de la Couche de Compatibilité

Les routes existantes (v1) ont été mises à jour pour utiliser automatiquement la couche de compatibilité. Cela signifie que :

1. **Transparence** : Les API existantes continuent de fonctionner sans changement
2. **Migration progressive** : Vous pouvez activer les services refactorisés un par un
3. **Rollback facile** : Retour aux services originaux en cas de problème

### Exemple d'utilisation

```bash
# Vérifier l'état actuel
curl -X GET "http://localhost:8000/services/migration-status"

# Activer le service de recherche refactorisé
curl -X POST "http://localhost:8000/services/toggle-migration" \
  -H "Content-Type: application/json" \
  -d '{"service_name": "search", "use_refactored": true}'

# Tester la recherche (utilise maintenant le service refactorisé)
curl -X POST "http://localhost:8000/search/" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "test",
    "limit": 5,
    "use_llm_reranking": true
  }'

# Ou utiliser directement l'API v2
curl -X POST "http://localhost:8000/api/v2/search/hybrid" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "test",
    "limit": 5,
    "use_llm_reranking": true
  }'
```

## Avantages des Services Refactorisés

### 1. **Gestion d'erreurs améliorée**
- Exceptions typées et structurées
- Messages d'erreur plus informatifs
- Gestion des timeouts et reconnexions

### 2. **Performance optimisée**
- Cache intelligent pour les embeddings
- Optimisations des requêtes de recherche
- Traitement par lots pour les embeddings

### 3. **Observabilité**
- Logs structurés avec contexte
- Métriques de performance
- Health checks détaillés

### 4. **Maintenabilité**
- Code modulaire et testable
- Configuration centralisée
- Architecture extensible

## Migration Recommandée

1. **Phase 1** : Tester avec l'API v2
   ```bash
   # Tester les nouveaux endpoints
   curl -X GET "http://localhost:8000/api/v2/health"
   ```

2. **Phase 2** : Migration progressive des services
   ```bash
   # Activer un service à la fois
   curl -X POST "http://localhost:8000/services/toggle-migration" \
     -d '{"service_name": "search", "use_refactored": true}'
   ```

3. **Phase 3** : Validation et monitoring
   ```bash
   # Surveiller les performances
   curl -X GET "http://localhost:8000/api/v2/embeddings/stats"
   ```

4. **Phase 4** : Migration complète
   ```bash
   # Activer tous les services refactorisés
   for service in search embedding document generate; do
     curl -X POST "http://localhost:8000/services/toggle-migration" \
       -d "{\"service_name\": \"$service\", \"use_refactored\": true}"
   done
   ```

## Dépannage

### Services non disponibles
Si les services refactorisés ne sont pas disponibles, vérifiez :
1. Les dépendances sont installées
2. Qdrant est accessible (pour les fonctionnalités complètes)
3. Les logs d'application pour les erreurs d'initialisation

### Retour aux services originaux
En cas de problème, vous pouvez rapidement revenir aux services originaux :
```bash
curl -X POST "http://localhost:8000/services/toggle-migration" \
  -d '{"service_name": "search", "use_refactored": false}'
```

## Support

Pour toute question ou problème :
1. Consultez les logs de l'application
2. Vérifiez l'état de santé : `GET /api/v2/health`
3. Consultez la documentation du REFACTORING_GUIDE.md