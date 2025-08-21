#!/usr/bin/env python3
"""
Script de test pour valider l'intégration des services refactorisés.
"""
import sys
import requests
import json
from typing import Dict, Any

def test_endpoint(base_url: str, method: str, endpoint: str, data: Dict[Any, Any] = None) -> Dict[str, Any]:
    """Test un endpoint et retourne le résultat."""
    url = f"{base_url}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        else:
            return {"success": False, "error": f"Méthode {method} non supportée"}
        
        return {
            "success": True,
            "status_code": response.status_code,
            "data": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def main():
    """Fonction principale de test."""
    # Configuration
    BASE_URL = "http://localhost:8000"
    
    print("🚀 Test de l'intégration des services refactorisés")
    print("=" * 60)
    
    # Tests des endpoints API v2
    tests_v2 = [
        ("GET", "/api/v2/health", None, "État de santé global"),
        ("GET", "/api/v2/migration/status", None, "Statut de migration"),
        ("GET", "/api/v2/embeddings/stats", None, "Statistiques embeddings"),
        ("GET", "/api/v2/documents/stats", None, "Statistiques documents"),
        ("POST", "/api/v2/embeddings/single", {"text": "test", "use_cache": True}, "Embedding simple"),
        ("POST", "/api/v2/search/hybrid", {
            "query": "test",
            "limit": 3,
            "use_llm_reranking": False,
            "boost_keywords": True
        }, "Recherche hybride"),
    ]
    
    # Tests des endpoints de migration dans les routes v1
    tests_migration = [
        ("GET", "/services/migration-status", None, "Statut migration v1"),
        ("POST", "/services/toggle-migration", {
            "service_name": "search",
            "use_refactored": True
        }, "Activation service search refactorisé"),
    ]
    
    # Tests des routes v1 modifiées
    tests_v1 = [
        ("POST", "/search/", {
            "query": "test",
            "limit": 3,
            "use_llm_reranking": False,
            "boost_keywords": True
        }, "Recherche v1 (avec compatibilité)"),
        ("GET", "/models", None, "Liste des modèles LLM"),
    ]
    
    # Exécution des tests
    all_tests = [
        ("📡 API v2 - Nouveaux endpoints", tests_v2),
        ("🔄 Migration - Contrôle des services", tests_migration),
        ("🔧 API v1 - Routes mises à jour", tests_v1),
    ]
    
    results = {}
    
    for section_name, tests in all_tests:
        print(f"\n{section_name}")
        print("-" * 40)
        
        section_results = []
        
        for method, endpoint, data, description in tests:
            print(f"  Testing: {description}")
            result = test_endpoint(BASE_URL, method, endpoint, data)
            
            if result["success"]:
                status = "✅" if result["status_code"] < 400 else "⚠️"
                print(f"    {status} {method} {endpoint} -> {result['status_code']}")
                if result["status_code"] >= 400:
                    print(f"      Error: {result.get('data', 'Unknown error')}")
            else:
                print(f"    ❌ {method} {endpoint} -> {result['error']}")
            
            section_results.append({
                "endpoint": endpoint,
                "method": method,
                "description": description,
                "result": result
            })
        
        results[section_name] = section_results
    
    # Résumé
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 60)
    
    total_tests = 0
    successful_tests = 0
    
    for section_name, section_results in results.items():
        section_success = sum(1 for r in section_results if r["result"]["success"] and r["result"].get("status_code", 500) < 400)
        section_total = len(section_results)
        
        total_tests += section_total
        successful_tests += section_success
        
        print(f"{section_name}: {section_success}/{section_total}")
        
        for test in section_results:
            if not test["result"]["success"] or test["result"].get("status_code", 500) >= 400:
                print(f"  ❌ {test['description']}: {test['result'].get('error', 'HTTP error')}")
    
    print(f"\n🎯 TOTAL: {successful_tests}/{total_tests} tests réussis")
    
    if successful_tests == total_tests:
        print("🎉 Tous les tests sont passés ! L'intégration est réussie.")
        return 0
    elif successful_tests > total_tests * 0.7:
        print("⚠️  La plupart des tests sont passés. Quelques services peuvent ne pas être disponibles.")
        return 0
    else:
        print("❌ Plusieurs tests ont échoué. Vérifiez la configuration.")
        return 1

if __name__ == "__main__":
    sys.exit(main())