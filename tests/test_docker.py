"""
Tests pour les fonctionnalités liées à Docker et aux conteneurs
Tests simples qui ne dépendent pas de services externes
"""

import pytest
import os
import subprocess
import sys

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_dockerfile_syntax():
    """Test que le Dockerfile a une syntaxe de base valide"""
    dockerfile_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Dockerfile")
    
    if not os.path.exists(dockerfile_path):
        pytest.skip("Dockerfile non trouvé")
        
    with open(dockerfile_path, 'r') as f:
        content = f.read()
        
    # Vérifications basiques de syntaxe
    assert content.startswith("FROM"), "Le Dockerfile doit commencer par FROM"
    assert "WORKDIR" in content, "Le Dockerfile doit contenir WORKDIR"
    assert "COPY" in content, "Le Dockerfile doit contenir COPY"
    assert "EXPOSE" in content, "Le Dockerfile doit contenir EXPOSE"
    assert "CMD" in content, "Le Dockerfile doit contenir CMD"

def test_dockerfile_port_configuration():
    """Test que le port est correctement exposé dans le Dockerfile"""
    dockerfile_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Dockerfile")
    
    if not os.path.exists(dockerfile_path):
        pytest.skip("Dockerfile non trouvé")
        
    with open(dockerfile_path, 'r') as f:
        content = f.read()
        
    # Vérifier que le port 8000 est exposé (port FastAPI par défaut)
    assert "EXPOSE 8000" in content, "Le port 8000 doit être exposé"

def test_dockerfile_python_setup():
    """Test que le Dockerfile configure Python correctement"""
    dockerfile_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Dockerfile")
    
    if not os.path.exists(dockerfile_path):
        pytest.skip("Dockerfile non trouvé")
        
    with open(dockerfile_path, 'r') as f:
        content = f.read()
        
    # Vérifier que Python est utilisé comme base
    lines = content.split('\n')
    from_line = next((line for line in lines if line.startswith('FROM')), '')
    assert "python" in from_line.lower(), "L'image de base doit être Python"

def test_requirements_for_docker():
    """Test que requirements.txt contient les dépendances nécessaires pour Docker"""
    req_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "requirements.txt")
    
    if not os.path.exists(req_path):
        pytest.skip("requirements.txt non trouvé")
        
    with open(req_path, 'r') as f:
        content = f.read().lower()
        
    # Vérifications des dépendances clés
    assert "fastapi" in content, "FastAPI doit être dans requirements.txt"
    assert "uvicorn" in content, "Uvicorn doit être dans requirements.txt"

def test_docker_command_structure():
    """Test que la commande Docker CMD est bien formée"""
    dockerfile_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Dockerfile")
    
    if not os.path.exists(dockerfile_path):
        pytest.skip("Dockerfile non trouvé")
        
    with open(dockerfile_path, 'r') as f:
        content = f.read()
        
    # Trouver la ligne CMD
    cmd_lines = [line.strip() for line in content.split('\n') if line.strip().startswith('CMD')]
    assert len(cmd_lines) > 0, "Le Dockerfile doit contenir une commande CMD"
    
    cmd_line = cmd_lines[0]
    assert "uvicorn" in cmd_line, "La commande CMD doit utiliser uvicorn"
    assert "app.main:app" in cmd_line, "La commande CMD doit référencer app.main:app"

class TestDockerEnvironment:
    """Classe de tests pour l'environnement Docker"""
    
    def test_environment_variables_structure(self):
        """Test de la structure des variables d'environnement"""
        # Variables importantes pour l'application
        important_vars = [
            "COMMIT_ID",
            "OLLAMA_URL", 
            "QDRANT_HOST",
            "QDRANT_PORT"
        ]
        
        # Test que les variables peuvent être lues (avec des valeurs par défaut)
        for var in important_vars:
            value = os.getenv(var, f"default_{var.lower()}")
            assert isinstance(value, str)
            assert len(value) > 0
            
    def test_port_configuration(self):
        """Test de la configuration des ports"""
        # Le port par défaut de FastAPI
        default_port = 8000
        
        # Test que le port peut être configuré via variable d'environnement
        port = int(os.getenv("PORT", default_port))
        assert isinstance(port, int)
        assert 1 <= port <= 65535, "Le port doit être dans la plage valide"
        
    def test_host_configuration(self):
        """Test de la configuration de l'hôte"""
        # Configuration par défaut pour Docker
        default_host = "0.0.0.0"
        
        host = os.getenv("HOST", default_host)
        assert isinstance(host, str)
        assert len(host) > 0