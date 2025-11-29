#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de diagnostic pour tester le dashboard
"""

import sqlite3
import pandas as pd
import os
import sys

# Forcer l'encodage UTF-8 pour éviter les problèmes avec les emojis
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("=" * 60)
print("DIAGNOSTIC DU DASHBOARD")
print("=" * 60)

# 1. Vérifier que la base de données existe
print("\n[1] Verification de la base de donnees...")
if os.path.exists('turnover_data.db'):
    print("  [OK] turnover_data.db existe")
else:
    print("  [ERREUR] turnover_data.db n'existe pas")
    print("     Exécutez: python create_database.py")
    exit(1)

# 2. Vérifier les tables
print("\n[2] Vérification des tables...")
try:
    conn = sqlite3.connect('turnover_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"  Tables trouvées: {', '.join(tables)}")
    
    required_tables = ['employees', 'turnover', 'performance', 'training', 'overtime', 'absences']
    missing_tables = [t for t in required_tables if t not in tables]
    
    if missing_tables:
        print(f"  [ERREUR] Tables manquantes: {', '.join(missing_tables)}")
    else:
        print("  [OK] Toutes les tables requises existent")
    
    # 3. Vérifier le contenu des tables
    print("\n[3] Vérification du contenu des tables...")
    for table in required_tables:
        if table in tables:
            try:
                df = pd.read_sql_query(f"SELECT COUNT(*) as count FROM {table}", conn)
                count = df['count'].iloc[0]
                if count > 0:
                    print(f"  [OK] {table}: {count} enregistrements")
                else:
                    print(f"  [ATTENTION] {table}: table vide")
            except Exception as e:
                print(f"  [ERREUR] Erreur lors de la lecture de {table}: {e}")
        else:
            print(f"  [ERREUR] {table}: table non trouvee")
    
    # 4. Vérifier les colonnes de la table employees
    print("\n[4] Vérification de la structure de la table employees...")
    try:
        df = pd.read_sql_query("SELECT * FROM employees LIMIT 1", conn)
        if not df.empty:
            print(f"  [OK] Colonnes: {', '.join(df.columns.tolist())}")
        else:
            print("  [ATTENTION] Table employees vide")
    except Exception as e:
        print(f"  [ERREUR] Erreur: {e}")
    
    conn.close()
    
except Exception as e:
    print(f"  ❌ Erreur lors de la connexion à la base de données: {e}")
    exit(1)

# 5. Vérifier Streamlit
print("\n[5] Vérification de Streamlit...")
try:
    import streamlit as st
    print(f"  [OK] Streamlit installe (version {st.__version__})")
except ImportError:
    print("  [ERREUR] Streamlit non installe")
    print("     Exécutez: pip install streamlit")

# 6. Vérifier les autres dépendances
print("\n[6] Vérification des dépendances...")
dependencies = ['pandas', 'numpy', 'plotly', 'openpyxl']
for dep in dependencies:
    try:
        __import__(dep)
        print(f"  [OK] {dep}")
    except ImportError:
        print(f"  [ERREUR] {dep} manquant")
        print(f"     Exécutez: pip install {dep}")

print("\n" + "=" * 60)
print("DIAGNOSTIC TERMINÉ")
print("=" * 60)

