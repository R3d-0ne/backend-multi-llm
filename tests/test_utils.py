"""
Tests pour les fonctions utilitaires et les imports de base
Ces tests s'exécutent sans dépendances externes lourdes
"""

import pytest
import os
import sys

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_requirements_file_exists():
    """Test que le fichier requirements.txt existe"""
    req_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "requirements.txt")
    assert os.path.exists(req_path), "Le fichier requirements.txt doit exister"

def test_dockerfile_exists():
    """Test que le Dockerfile existe"""
    dockerfile_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Dockerfile")
    assert os.path.exists(dockerfile_path), "Le Dockerfile doit exister"

def test_app_directory_structure():
    """Test que la structure de l'application est présente"""
    app_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app")
    assert os.path.exists(app_dir), "Le répertoire app/ doit exister"
    
    main_py = os.path.join(app_dir, "main.py")
    assert os.path.exists(main_py), "Le fichier app/main.py doit exister"

def test_environment_variable_handling():
    """Test de la gestion des variables d'environnement"""
    # Test avec une variable d'environnement fictive
    test_var = os.getenv("TEST_VAR", "default_value")
    assert test_var == "default_value"
    
    # Test avec COMMIT_ID (utilisé dans le CI/CD)
    commit_id = os.getenv("COMMIT_ID", "dev")
    assert isinstance(commit_id, str)
    assert len(commit_id) > 0

def test_python_version_compatibility():
    """Test que la version Python est compatible"""
    import sys
    version_info = sys.version_info
    
    # Le CI utilise Python 3.8+
    assert version_info.major == 3
    assert version_info.minor >= 8, f"Python 3.8+ requis, trouvé {version_info.major}.{version_info.minor}"

class TestBasicFunctionality:
    """Classe de tests pour les fonctionnalités de base"""
    
    def test_json_handling(self):
        """Test de la manipulation JSON de base"""
        import json
        
        test_data = {"version": "1.0.0", "status": "healthy"}
        json_str = json.dumps(test_data)
        parsed_data = json.loads(json_str)
        
        assert parsed_data == test_data
        
    def test_logging_setup(self):
        """Test que le logging peut être configuré"""
        import logging
        
        logger = logging.getLogger("test_logger")
        assert logger is not None
        assert hasattr(logger, "info")
        
    def test_os_operations(self):
        """Test des opérations système de base"""
        import tempfile
        import os
        
        # Créer un fichier temporaire
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"test content")
            tmp_path = tmp.name
            
        # Vérifier qu'il existe
        assert os.path.exists(tmp_path)
        
        # Le nettoyer
        os.unlink(tmp_path)
        assert not os.path.exists(tmp_path)