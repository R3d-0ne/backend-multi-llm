"""
Script de démonstration pour tester les services refactorisés.
Ce script montre comment utiliser les nouveaux services et effectuer des tests de migration.
"""
import json
import asyncio
from typing import Dict, Any

# Import des services de compatibilité
from app.services.service_compatibility import (
    migration_manager,
    search_service_compat,
    embedding_service_compat,
    document_service_compat,
    generate_service_compat,
    get_service_health_summary,
    run_migration_tests
)

# Import direct des services refactorisés pour tests avancés
from app.services.search_service_refactored import search_service_refactored
from app.services.embedding_service_refactored import embedding_service_refactored
from app.services.document_service_refactored import document_service_refactored
from app.services.generate_service_refactored import generate_service_refactored


def print_section(title: str):
    """Affiche un titre de section"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def print_result(name: str, result: Any, success: bool = True):
    """Affiche un résultat de test"""
    status = "✅" if success else "❌"
    print(f"{status} {name}")
    if isinstance(result, dict):
        print(json.dumps(result, indent=2, default=str)[:500] + ("..." if len(str(result)) > 500 else ""))
    else:
        print(f"   {str(result)[:200]}{'...' if len(str(result)) > 200 else ''}")
    print()


def test_config_service():
    """Test du service de configuration"""
    print_section("Test du Service de Configuration")
    
    try:
        from app.services.config_service import config_service
        
        # Test de récupération de configuration
        config_dict = config_service.get_config_dict()
        print_result("Configuration chargée", config_dict)
        
        # Test de mise à jour de configuration
        original_temp = config_service.llm.temperature
        success = config_service.update_config("llm", "temperature", 0.5)
        print_result(f"Mise à jour température: {original_temp} → 0.5", {"success": success})
        
        # Restaurer la valeur originale
        config_service.update_config("llm", "temperature", original_temp)
        
    except Exception as e:
        print_result("Erreur configuration", {"error": str(e)}, False)


def test_embedding_service():
    """Test du service d'embeddings refactorisé"""
    print_section("Test du Service d'Embeddings Refactorisé")
    
    try:
        # Test simple
        test_text = "Ceci est un test pour le service d'embeddings"
        embedding = embedding_service_refactored.get_embedding(test_text)
        print_result("Embedding simple", {
            "text": test_text,
            "embedding_dimension": len(embedding) if embedding else 0,
            "first_values": embedding[:5] if embedding else None
        })
        
        # Test batch
        texts = ["Premier texte", "Deuxième texte", "Troisième texte"]
        batch_embeddings = embedding_service_refactored.get_embeddings_batch(texts, batch_size=2)
        print_result("Embedding batch", {
            "texts_count": len(texts),
            "embeddings_count": len(batch_embeddings) if batch_embeddings else 0
        })
        
        # Test de santé
        health = embedding_service_refactored.health_check()
        print_result("Santé du service", health.data if health.success else health.error, health.success)
        
        # Test des statistiques de cache
        cache_stats = embedding_service_refactored.get_cache_stats()
        print_result("Statistiques de cache", cache_stats)
        
    except Exception as e:
        print_result("Erreur embedding", {"error": str(e)}, False)


def test_document_service():
    """Test du service de documents refactorisé"""
    print_section("Test du Service de Documents Refactorisé")
    
    try:
        # Test de listage des documents
        list_result = document_service_refactored.list_documents(limit=5)
        if list_result.success:
            print_result("Listage des documents", {
                "total_returned": list_result.data.get("total_returned", 0),
                "documents": [doc.get("document_id", "unknown") for doc in list_result.data.get("documents", [])[:3]]
            })
        else:
            print_result("Erreur listage", list_result.error, False)
        
        # Test des statistiques
        stats_result = document_service_refactored.get_document_statistics()
        if stats_result.success:
            print_result("Statistiques des documents", {
                "total_documents": stats_result.data.get("total_documents", 0),
                "file_types": stats_result.data.get("file_types", {}),
                "collection_info": stats_result.data.get("collection_info", {})
            })
        else:
            print_result("Erreur statistiques", stats_result.error, False)
        
        # Test de santé
        health = document_service_refactored.health_check()
        print_result("Santé du service", health.data if health.success else health.error, health.success)
        
    except Exception as e:
        print_result("Erreur document", {"error": str(e)}, False)


def test_search_service():
    """Test du service de recherche refactorisé"""
    print_section("Test du Service de Recherche Refactorisé")
    
    try:
        # Test de recherche simple
        search_query = "test de recherche"
        search_result = search_service_refactored.hybrid_search(
            query=search_query,
            limit=3,
            use_llm_reranking=False  # Désactiver le LLM pour ce test
        )
        print_result("Recherche hybride", {
            "query": search_query,
            "total_found": search_result.get("total_found", 0),
            "results_count": len(search_result.get("results", []))
        })
        
        # Test de santé
        health = search_service_refactored.health_check()
        print_result("Santé du service", {
            "status": health.data.get("status") if health.success else "error",
            "components": list(health.data.get("components", {}).keys()) if health.success else []
        }, health.success)
        
        # Test des statistiques de cache
        cache_stats = search_service_refactored.get_cache_stats()
        print_result("Statistiques de cache", cache_stats)
        
    except Exception as e:
        print_result("Erreur recherche", {"error": str(e)}, False)


def test_generate_service():
    """Test du service de génération refactorisé"""
    print_section("Test du Service de Génération Refactorisé")
    
    try:
        # Test de validation de requête
        test_request = {
            "current_message": "Bonjour, comment allez-vous ?",
            "additional_info": "Test de génération"
        }
        
        validation = generate_service_refactored.validate_generate_request(test_request)
        print_result("Validation de requête", validation.data if validation.success else validation.error, validation.success)
        
        # Test des statistiques
        stats = generate_service_refactored.get_generation_statistics()
        print_result("Statistiques de génération", stats.data if stats.success else stats.error, stats.success)
        
        # Test de santé
        health = generate_service_refactored.health_check()
        print_result("Santé du service", {
            "overall_status": health.data.get("overall_status") if health.success else "error",
            "dependencies": list(health.data.get("dependencies", {}).keys()) if health.success else []
        }, health.success)
        
    except Exception as e:
        print_result("Erreur génération", {"error": str(e)}, False)


def test_compatibility_layer():
    """Test de la couche de compatibilité"""
    print_section("Test de la Couche de Compatibilité")
    
    try:
        # Statut de migration
        migration_status = migration_manager.get_migration_status()
        print_result("Statut de migration", migration_status)
        
        # Tests de migration
        migration_tests = run_migration_tests()
        print_result("Tests de migration", migration_tests)
        
        # Résumé de santé global
        health_summary = get_service_health_summary()
        print_result("Résumé de santé", {
            "migration_status": health_summary.get("migration_status", {}),
            "health_checks": {k: v.get("status") for k, v in health_summary.get("health_checks", {}).items()}
        })
        
    except Exception as e:
        print_result("Erreur compatibilité", {"error": str(e)}, False)


def test_service_comparison():
    """Compare les performances entre anciens et nouveaux services"""
    print_section("Comparaison des Services")
    
    try:
        import time
        
        # Test d'embedding avec timing
        test_text = "Texte de test pour comparaison de performance"
        
        # Ancien service (via compatibilité)
        migration_manager.migrate_service('embedding', False)
        start_time = time.time()
        old_embedding = embedding_service_compat.get_embedding(test_text)
        old_time = time.time() - start_time
        
        # Nouveau service
        migration_manager.migrate_service('embedding', True)
        start_time = time.time()
        new_embedding = embedding_service_compat.get_embedding(test_text)
        new_time = time.time() - start_time
        
        print_result("Comparaison embeddings", {
            "old_service_time": f"{old_time:.3f}s",
            "new_service_time": f"{new_time:.3f}s",
            "improvement": f"{((old_time - new_time) / old_time * 100):.1f}%" if old_time > 0 else "N/A",
            "results_match": (old_embedding == new_embedding) if old_embedding and new_embedding else False
        })
        
    except Exception as e:
        print_result("Erreur comparaison", {"error": str(e)}, False)


def main():
    """Fonction principale pour exécuter tous les tests"""
    print("🚀 Démarrage des tests des services refactorisés")
    print(f"Timestamp: {json.datetime.now() if hasattr(json, 'datetime') else 'N/A'}")
    
    # Exécuter tous les tests
    test_config_service()
    test_embedding_service()
    test_document_service()
    test_search_service()
    test_generate_service()
    test_compatibility_layer()
    test_service_comparison()
    
    print_section("Tests Terminés")
    print("✅ Tous les tests de démonstration ont été exécutés.")
    print("📊 Consultez les résultats ci-dessus pour évaluer les performances.")
    print("🔧 Utilisez la couche de compatibilité pour une migration progressive.")


if __name__ == "__main__":
    # Exécuter les tests seulement si ce script est appelé directement
    main()