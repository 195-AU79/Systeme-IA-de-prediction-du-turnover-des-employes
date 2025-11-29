#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de démarrage du système de prédiction du turnover
"""

import os
import sys
import subprocess

def main():
    print("\n" + "="*60)
    print("SYSTEME DE PREDICTION DU TURNOVER")
    print("="*60)
    
    print("\nChoisissez une option:")
    print("1. Entrainer le modele principal (analyse des criteres)")
    print("2. Analyser des employes specifiques")
    print("3. Demarrer le dashboard")
    print("4. Quitter")
    
    choice = input("\nVotre choix (1-4): ").strip()
    
    if choice == '1':
        print("\nEntrainement du modele principal...")
        subprocess.run([sys.executable, "main.py"])
    
    elif choice == '2':
        print("\nAnalyse d'employes specifiques...")
        subprocess.run([sys.executable, "employee_analyzer_simple.py"])
    
    elif choice == '3':
        print("\nDemarrage du dashboard...")
        # Vérifier que Streamlit est installé
        try:
            import streamlit
            print("Dashboard disponible sur: http://localhost:8501")
            subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"])
        except ImportError:
            print("\n❌ ERREUR: Streamlit n'est pas installe pour cette version de Python")
            print(f"Version Python utilisee: {sys.version}")
            print("\nPour installer Streamlit, executez:")
            print(f"  {sys.executable} -m pip install streamlit")
            print("\nOu installez toutes les dependances avec:")
            print(f"  {sys.executable} -m pip install -r requirements.txt")
    
    elif choice == '4':
        print("\nAu revoir!")
    
    else:
        print("\nChoix invalide")

if __name__ == "__main__":
    main()
