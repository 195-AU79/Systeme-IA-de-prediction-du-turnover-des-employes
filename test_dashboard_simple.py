#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test simple du dashboard - vérifie que les imports fonctionnent
"""

import sys

print("Test des imports du dashboard...")

try:
    import streamlit as st
    print("  [OK] streamlit")
except ImportError as e:
    print(f"  [ERREUR] streamlit: {e}")
    sys.exit(1)

try:
    import pandas as pd
    print("  [OK] pandas")
except ImportError as e:
    print(f"  [ERREUR] pandas: {e}")
    sys.exit(1)

try:
    import plotly.express as px
    print("  [OK] plotly")
except ImportError as e:
    print(f"  [ERREUR] plotly: {e}")
    sys.exit(1)

try:
    import sqlite3
    print("  [OK] sqlite3")
except ImportError as e:
    print(f"  [ERREUR] sqlite3: {e}")
    sys.exit(1)

# Tester l'import du dashboard
try:
    print("\nTest de l'import du module dashboard...")
    import dashboard
    print("  [OK] Le module dashboard s'importe correctement")
except Exception as e:
    print(f"  [ERREUR] Erreur lors de l'import du dashboard: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("TOUS LES TESTS SONT PASSE!")
print("="*60)
print("\nVous pouvez maintenant lancer le dashboard avec:")
print("  python -m streamlit run dashboard.py")
print("ou")
print("  python start.py (option 3)")



