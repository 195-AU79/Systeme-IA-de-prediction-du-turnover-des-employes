#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de test pour vérifier la configuration LLM
"""

import os
import sys

def test_openai_available():
    """Test si OpenAI est disponible"""
    try:
        from openai import OpenAI
        print("✓ Bibliothèque OpenAI disponible")
        return True
    except ImportError:
        print("✗ Bibliothèque OpenAI non disponible")
        print("  Installation: pip install openai")
        return False

def test_api_key():
    """Test si la clé API est configurée"""
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print(f"✓ Clé API OpenAI trouvée ({api_key[:10]}...)")
        return True
    else:
        print("✗ Clé API OpenAI non trouvée")
        print("  Définir la variable d'environnement: OPENAI_API_KEY")
        print("  Windows PowerShell: $env:OPENAI_API_KEY='votre_cle'")
        print("  Linux/Mac: export OPENAI_API_KEY='votre_cle'")
        return False

def test_config():
    """Test si le fichier config.yaml est valide"""
    try:
        import yaml
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        llm_config = config.get('llm_data_generation', {})
        if llm_config.get('enabled', False):
            print("✓ Génération LLM activée dans config.yaml")
            print(f"  Modèle: {llm_config.get('model', 'N/A')}")
            print(f"  Batch size: {llm_config.get('batch_size', 'N/A')}")
        else:
            print("ℹ Génération LLM désactivée dans config.yaml")
            print("  Pour l'activer, mettre 'enabled: true' dans llm_data_generation")
        
        return True
    except Exception as e:
        print(f"✗ Erreur lecture config.yaml: {e}")
        return False

def test_llm_connection():
    """Test la connexion à l'API OpenAI"""
    try:
        from openai import OpenAI
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("⚠ Impossible de tester: clé API absente")
            return False
        
        client = OpenAI(api_key=api_key)
        # Test simple
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Réponds juste 'OK'"}],
            max_tokens=5
        )
        print("✓ Connexion à l'API OpenAI réussie")
        return True
    except Exception as e:
        print(f"✗ Erreur connexion API: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("TEST DE CONFIGURATION LLM")
    print("=" * 60)
    print()
    
    results = {
        "Bibliothèque": test_openai_available(),
        "Clé API": test_api_key(),
        "Configuration": test_config(),
    }
    
    print()
    if all([results["Bibliothèque"], results["Clé API"]]):
        print("Test de connexion...")
        results["Connexion"] = test_llm_connection()
    
    print()
    print("=" * 60)
    if all(results.values()):
        print("✓ TOUS LES TESTS RÉUSSIS")
        print("Vous pouvez utiliser la génération LLM !")
    else:
        print("⚠ CERTAINS TESTS ONT ÉCHOUÉ")
        print("Consultez les messages ci-dessus pour plus de détails.")
    print("=" * 60)


