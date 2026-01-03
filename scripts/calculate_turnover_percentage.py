#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour calculer le pourcentage de salariés qui vont partir
Utilise le modèle entraîné pour prédire le risque de turnover
"""

import pandas as pd
import numpy as np
import sqlite3
import joblib
import json
from datetime import datetime
from paths import get_model_path, get_data_path, get_db_path

def calculate_turnover_percentage():
    """Calcule le pourcentage de salariés qui vont partir"""
    
    print("=" * 70)
    print("CALCUL DU POURCENTAGE DE SALARIÉS QUI VONT PARTIR")
    print("=" * 70)
    
    # Charger le modèle
    try:
        model = joblib.load(get_model_path('turnover_criteria_model.pkl'))
        scaler = joblib.load(get_model_path('criteria_scaler.pkl'))
        
        # Charger le seuil optimal
        best_threshold = 0.5
        try:
            with open(get_data_path('criteria_analysis_results.json'), 'r', encoding='utf-8') as f:
                results = json.load(f)
                if 'model_performance' in results and 'best_threshold' in results['model_performance']:
                    best_threshold = results['model_performance']['best_threshold']
        except:
            pass
        
        print(f"\n[OK] Modèle chargé (seuil optimal: {best_threshold:.3f})")
    except Exception as e:
        print(f"[ERREUR] Impossible de charger le modèle: {e}")
        print("Assurez-vous d'avoir exécuté main.py pour entraîner le modèle d'abord.")
        return
    
    # Charger les données depuis csv_original_data
    try:
        conn = sqlite3.connect(get_db_path())
        df = pd.read_sql_query("SELECT * FROM csv_original_data", conn)
        conn.close()
        
        if df.empty:
            print("[ERREUR] Aucune donnée trouvée dans csv_original_data")
            return
        
        print(f"[OK] {len(df)} employés chargés")
    except Exception as e:
        print(f"[ERREUR] Impossible de charger les données: {e}")
        return
    
    # Préparer les données comme dans main.py
    from main import TurnoverPredictionModel
    
    model_wrapper = TurnoverPredictionModel()
    model_wrapper.model = model
    model_wrapper.scaler = scaler
    model_wrapper.best_threshold = best_threshold
    
    # Charger les feature names depuis les résultats
    try:
        with open(get_data_path('criteria_analysis_results.json'), 'r', encoding='utf-8') as f:
            results = json.load(f)
            model_wrapper.feature_names = [item['feature'] for item in results['feature_importance']]
    except:
        print("[ERREUR] Impossible de charger les noms de features")
        return
    
    # Renommer les colonnes comme dans main.py
    column_mapping = {
        'Age': 'age',
        'Department': 'department',
        'JobLevel': 'job_level',
        'MonthlyIncome': 'salary',
        'YearsAtCompany': 'tenure_years',
        'PerformanceRating': 'performance_rating',
        'TrainingTimesLastYear': 'training_times',
        'WorkLifeBalance': 'work_life_balance',
        'JobSatisfaction': 'job_satisfaction',
        'EnvironmentSatisfaction': 'environment_satisfaction',
        'RelationshipSatisfaction': 'relationship_satisfaction',
        'JobInvolvement': 'job_involvement',
        'DistanceFromHome': 'distance_from_home',
        'YearsInCurrentRole': 'years_in_current_role',
        'YearsSinceLastPromotion': 'years_since_last_promotion',
        'YearsWithCurrManager': 'years_with_curr_manager',
        'NumCompaniesWorked': 'num_companies_worked',
        'TotalWorkingYears': 'total_working_years',
        'PercentSalaryHike': 'percent_salary_hike',
        'OverTime': 'overtime',
        'MaritalStatus': 'marital_status',
        'Education': 'education',
        'EducationField': 'education_field',
        'JobRole': 'job_role',
        'Gender': 'gender',
        'BusinessTravel': 'business_travel',
        'StockOptionLevel': 'stock_option_level',
        'DailyRate': 'DailyRate',
        'HourlyRate': 'HourlyRate',
        'MonthlyRate': 'MonthlyRate',
        'StandardHours': 'StandardHours'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Convertir MonthlyIncome en salaire annuel
    if 'salary' in df.columns:
        df['salary'] = df['salary'] * 12
    
    # Créer location basée sur DistanceFromHome
    if 'distance_from_home' in df.columns:
        df['location'] = pd.cut(df['distance_from_home'], 
                               bins=[0, 5, 10, 15, float('inf')], 
                               labels=['Proche', 'Moyenne', 'Loin', 'Tres_Loin'])
        df['location'] = df['location'].astype(str)
    
    # Convertir les colonnes catégorielles
    categorical_cols = ['department', 'job_level', 'location', 'overtime', 'marital_status', 
                       'education_field', 'job_role', 'gender', 'business_travel']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    # Préparer les features comme dans select_features
    numeric_features = [
        'age', 'salary', 'tenure_years', 'distance_from_home',
        'performance_rating', 'training_times', 'work_life_balance',
        'job_satisfaction', 'environment_satisfaction', 'relationship_satisfaction',
        'job_involvement', 'years_in_current_role', 'years_since_last_promotion',
        'years_with_curr_manager', 'num_companies_worked', 'total_working_years',
        'percent_salary_hike', 'stock_option_level', 'education',
        'DailyRate', 'HourlyRate', 'MonthlyRate', 'StandardHours'
    ]
    
    available_features = [col for col in numeric_features if col in df.columns]
    X_numeric = df[available_features].fillna(0)
    
    categorical_features = ['department', 'job_level', 'location', 'overtime', 
                          'marital_status', 'education_field', 'job_role', 
                          'gender', 'business_travel']
    categorical_features = [col for col in categorical_features if col in df.columns]
    
    X_categorical = pd.DataFrame()
    for col in categorical_features:
        if col in df.columns:
            dummies = pd.get_dummies(df[col].astype(str), prefix=col, drop_first=False)
            X_categorical = pd.concat([X_categorical, dummies], axis=1)
    
    X = pd.concat([X_numeric, X_categorical], axis=1)
    
    # S'assurer que toutes les features du modèle sont présentes
    for feature in model_wrapper.feature_names:
        if feature not in X.columns:
            X[feature] = 0
    
    # Réorganiser les colonnes dans l'ordre du modèle
    X = X[model_wrapper.feature_names]
    
    # Normaliser
    X_scaled = scaler.transform(X)
    
    # Prédictions
    predictions_proba = model.predict_proba(X_scaled)[:, 1]
    predictions = (predictions_proba >= best_threshold).astype(int)
    
    # Calculer les statistiques
    total_employees = len(df)
    employees_leaving = int(predictions.sum())
    percentage_leaving = (employees_leaving / total_employees) * 100
    
    # Par niveau de risque
    high_risk = (predictions_proba >= 0.7).sum()
    medium_risk = ((predictions_proba >= 0.4) & (predictions_proba < 0.7)).sum()
    low_risk = (predictions_proba < 0.4).sum()
    
    # Par département
    df['prediction'] = predictions
    df['risk_score'] = predictions_proba
    turnover_by_dept = df.groupby('department').agg({
        'prediction': ['sum', 'count'],
        'risk_score': 'mean'
    }).round(2)
    turnover_by_dept.columns = ['Départs_prévus', 'Total', 'Risque_moyen']
    turnover_by_dept['Pourcentage'] = (turnover_by_dept['Départs_prévus'] / turnover_by_dept['Total'] * 100).round(1)
    
    # Afficher les résultats
    print("\n" + "=" * 70)
    print("RÉSULTATS GLOBAUX")
    print("=" * 70)
    print(f"Total d'employés analysés: {total_employees}")
    print(f"Employés qui vont partir: {employees_leaving} ({percentage_leaving:.1f}%)")
    print(f"Employés qui vont rester: {total_employees - employees_leaving} ({100 - percentage_leaving:.1f}%)")
    
    print("\n" + "=" * 70)
    print("RÉPARTITION PAR NIVEAU DE RISQUE")
    print("=" * 70)
    print(f"Risque eleve (>=70%): {high_risk} employes ({high_risk/total_employees*100:.1f}%)")
    print(f"Risque moyen (40-70%): {medium_risk} employes ({medium_risk/total_employees*100:.1f}%)")
    print(f"Risque faible (<40%): {low_risk} employes ({low_risk/total_employees*100:.1f}%)")
    
    print("\n" + "=" * 70)
    print("RÉPARTITION PAR DÉPARTEMENT")
    print("=" * 70)
    print(turnover_by_dept.to_string())
    
    # Sauvegarder les résultats
    results = {
        'timestamp': datetime.now().isoformat(),
        'total_employees': int(total_employees),
        'employees_leaving': int(employees_leaving),
        'percentage_leaving': float(percentage_leaving),
        'risk_distribution': {
            'high_risk': int(high_risk),
            'medium_risk': int(medium_risk),
            'low_risk': int(low_risk)
        },
        'by_department': turnover_by_dept.to_dict('index')
    }
    
    with open(get_data_path('turnover_predictions.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Résultats sauvegardés dans 'turnover_predictions.json'")
    
    return results

if __name__ == "__main__":
    calculate_turnover_percentage()

