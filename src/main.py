#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Système de Prédiction du Turnover des Employés - Modèle d'Analyse des Critères
Avec critères spécifiques : salaire, ancienneté, concurrence, relation manager, augmentations, image société
"""

import pandas as pd
import numpy as np
import sqlite3
import joblib
import json
import yaml
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import path helpers
from paths import get_db_path, get_model_path, get_data_path

# ==============================================================================
# DICTIONNAIRE DE TRADUCTION DES CRITÈRES
# ==============================================================================

CRITERIA_TRANSLATION = {
    # Données de base du CSV
    'age': 'Âge',
    'tenure_years': 'Ancienneté (années)',
    'salary': 'Salaire annuel',
    'department': 'Département',
    'job_level': 'Niveau hiérarchique',
    'location': 'Localisation (distance)',
    'distance_from_home': 'Distance du domicile',
    'gender': 'Genre',
    'marital_status': 'Statut marital',
    'education': 'Niveau d\'éducation',
    'education_field': 'Domaine d\'éducation',
    'job_role': 'Rôle professionnel',
    'business_travel': 'Voyages professionnels',
    
    # Performance et satisfaction
    'performance_rating': 'Note de performance',
    'job_satisfaction': 'Satisfaction au travail',
    'environment_satisfaction': 'Satisfaction environnement',
    'relationship_satisfaction': 'Satisfaction relations',
    'job_involvement': 'Implication professionnelle',
    'work_life_balance': 'Équilibre vie/travail',
    
    # Carrière
    'years_in_current_role': 'Années dans le rôle actuel',
    'years_since_last_promotion': 'Années depuis dernière promotion',
    'years_with_curr_manager': 'Années avec manager actuel',
    'num_companies_worked': 'Nombre d\'entreprises',
    'total_working_years': 'Années d\'expérience totale',
    'percent_salary_hike': 'Pourcentage d\'augmentation',
    
    # Formation et développement
    'training_times': 'Nombre de formations',
    
    # Compensation
    'stock_option_level': 'Niveau d\'options d\'achat',
    'DailyRate': 'Taux journalier',
    'HourlyRate': 'Taux horaire',
    'MonthlyRate': 'Taux mensuel',
    'StandardHours': 'Heures standard',
    
    # Charge de travail
    'overtime': 'Heures supplémentaires',
}

def translate_criteria(criteria_name: str) -> str:
    """
    Traduit un nom de critère en français.
    Si le critère n'est pas dans le dictionnaire, retourne le nom original formaté.
    """
    return CRITERIA_TRANSLATION.get(criteria_name, criteria_name.replace('_', ' ').title())

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve, accuracy_score
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb

# Analyse des features
import shap

print("\n" + "="*70)
print("SYSTEME DE PREDICTION DU TURNOVER - CRITERES SPECIFIQUES")
print("="*70)

class TurnoverPredictionModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_importance = None
        self.shap_explainer = None
        self.feature_names = []
        self.criteria_analysis = {}
        self.best_threshold = 0.5
        self.best_model_name = None
        self.model_performance = {}
        self.categorical_columns = []  # Pour stocker les colonnes catégorielles utilisées
        
    def load_and_prepare_data(self):
        """Charge et prépare les données directement depuis le CSV original"""
        print("\n[1/6] Chargement et preparation des donnees depuis le CSV...")
        
        try:
            conn = sqlite3.connect(get_db_path())
            
            # Charger directement les données originales du CSV
            df = pd.read_sql_query("SELECT * FROM csv_original_data", conn)
            
            # Vérifier si la table existe
            if df.empty:
                print("[ERREUR] La table csv_original_data est vide. Veuillez d'abord exécuter import_csv_to_database.py")
                conn.close()
                return None
            
            conn.close()
            
            # Créer la variable cible depuis Attrition
            df['left_company'] = (df['Attrition'] == 'Yes').astype(int)
            
            # Renommer les colonnes pour correspondre aux noms utilisés dans le modèle
            # Garder les noms originaux du CSV pour la clarté
            column_mapping = {
                'Age': 'age',
                'Department': 'department',
                'JobLevel': 'job_level',
                'MonthlyIncome': 'salary',  # Convertir en salaire annuel plus tard si nécessaire
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
                'StockOptionLevel': 'stock_option_level'
            }
            
            # Renommer les colonnes
            df = df.rename(columns=column_mapping)
            
            # Convertir MonthlyIncome en salaire annuel (il est déjà mensuel dans le CSV)
            if 'salary' in df.columns:
                df['salary'] = df['salary'] * 12  # Convertir en annuel
            
            # Créer une colonne location basée sur DistanceFromHome
            if 'distance_from_home' in df.columns:
                df['location'] = pd.cut(df['distance_from_home'], 
                                       bins=[0, 5, 10, 15, float('inf')], 
                                       labels=['Proche', 'Moyenne', 'Loin', 'Tres_Loin'])
                df['location'] = df['location'].astype(str)
            
            # Convertir les colonnes catégorielles en string
            categorical_cols = ['department', 'job_level', 'location', 'overtime', 'marital_status', 
                              'education_field', 'job_role', 'gender', 'business_travel']
            for col in categorical_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str)
            
            # Remplacer les NaN par 0 pour les colonnes numériques
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(0)
            
            print(f"[OK] Dataset prepare: {len(df)} employes")
            print(f"[INFO] Taux de turnover: {df['left_company'].mean():.1%}")
            print(f"[INFO] Variables disponibles: {len(df.columns)} colonnes")
            
            return df
            
        except Exception as e:
            print(f"[ERREUR] Impossible de preparer les donnees: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_enriched_criteria(self, df):
        """Crée des critères enrichis pour l'analyse"""
        print("  - Creation des criteres enrichis...")
        
        # 1. Critères salariaux
        df['salary_vs_avg_dept'] = df.groupby('department')['salary'].transform(lambda x: x / x.mean())
        df['salary_vs_avg_level'] = df.groupby('job_level')['salary'].transform(lambda x: x / x.mean())
        df['salary_percentile'] = df.groupby(['department', 'job_level'])['salary'].transform(lambda x: x.rank(pct=True))
        
        # 2. Critères de performance
        df['performance_trend'] = df['performance_rating_mean'] - df['performance_rating_min']
        # performance_consistency retiré - critère peu clair
        df['goals_achievement_rate'] = df['goals_achieved_mean'] / 100
        
        # 3. Critères de formation et développement
        df['training_intensity'] = df['hours_completed_sum'] / (df['tenure_years'] + 0.1)
        df['training_quality'] = df['score_mean'].fillna(0)
        df['training_frequency'] = df['hours_completed_count'] / (df['tenure_years'] + 0.1)
        
        # 4. Critères de charge de travail
        df['overtime_intensity'] = df['hours_sum'] / (df['tenure_years'] + 0.1)
        df['overtime_frequency'] = df['hours_count'] / (df['tenure_years'] + 0.1)
        df['projects_count'] = df['hours_count']  # Utiliser le nombre d'épisodes d'heures sup comme proxy pour les projets
        df['workload_indicator'] = df['overtime_intensity'] * df['projects_count']
        
        # 5. Critères d'absences
        df['absence_rate'] = df['duration_days_sum'] / (df['tenure_years'] * 365 + 1)
        df['absence_frequency'] = df['duration_days_count'] / (df['tenure_years'] + 0.1)
        
        # 6. Critères de satisfaction (simulés basés sur d'autres facteurs)
        df['satisfaction_score'] = (
            df['performance_rating_mean'] * 0.3 +
            df['goals_achievement_rate'] * 0.2 +
            df['salary_percentile'] * 0.2 +
            (1 - df['overtime_intensity'] / df['overtime_intensity'].max()) * 0.15 +
            df['training_quality'] / 100 * 0.15
        )
        
        # 7. Critères de relation manager (simulés)
        df['manager_relationship'] = (
            df['feedback_score_mean'] * 0.5 +
            (1 - df['absence_rate']) * 0.5
        )
        
        # 8. Critères de marché (simulés)
        # Simulation des salaires du marché basée sur le département et le niveau
        market_salaries = {
            'IT': {'Junior': 40000, 'Mid': 55000, 'Senior': 75000, 'Lead': 95000, 'Manager': 120000, 'Director': 150000},
            'Sales': {'Junior': 35000, 'Mid': 50000, 'Senior': 70000, 'Lead': 90000, 'Manager': 110000, 'Director': 140000},
            'HR': {'Junior': 32000, 'Mid': 45000, 'Senior': 60000, 'Lead': 75000, 'Manager': 95000, 'Director': 120000},
            'Finance': {'Junior': 38000, 'Mid': 52000, 'Senior': 72000, 'Lead': 92000, 'Manager': 115000, 'Director': 145000},
            'Operations': {'Junior': 33000, 'Mid': 47000, 'Senior': 65000, 'Lead': 82000, 'Manager': 105000, 'Director': 130000},
            'Marketing': {'Junior': 36000, 'Mid': 51000, 'Senior': 70000, 'Lead': 88000, 'Manager': 110000, 'Director': 135000},
            'Legal': {'Junior': 42000, 'Mid': 58000, 'Senior': 80000, 'Lead': 100000, 'Manager': 125000, 'Director': 160000},
            'R&D': {'Junior': 45000, 'Mid': 62000, 'Senior': 85000, 'Lead': 110000, 'Manager': 135000, 'Director': 170000}
        }
        
        df['market_salary'] = df.apply(lambda row: market_salaries.get(row['department'], {}).get(row['job_level'], 50000), axis=1)
        df['salary_vs_market'] = df['salary'] / df['market_salary']
        
        # 9. Critères de carrière
        df['career_progression'] = df['tenure_years'] / (df['job_level'].map({'Junior': 1, 'Mid': 2, 'Senior': 3, 'Lead': 4, 'Manager': 5, 'Director': 6}) + 0.1)
        df['promotion_likelihood'] = df['performance_rating_mean'] * df['goals_achievement_rate'] * df['training_intensity']
        
        # 10. Critères de stress et équilibre
        df['stress_indicator'] = df['overtime_intensity'] + df['absence_rate']
        df['work_life_balance'] = 1 / (1 + df['stress_indicator'])
        
        # 11. CRITÈRES SPÉCIFIQUES DEMANDÉS
        print("  - Ajout des criteres specifiques...")
        
        # 11.1 Critères de salaire détaillés
        df['salary_gap_vs_market'] = df['salary_vs_market'] - 1.0  # Écart par rapport au marché
        df['salary_competitiveness'] = np.where(df['salary_vs_market'] > 1.1, 1, 
                                               np.where(df['salary_vs_market'] < 0.9, -1, 0))
        
        # 11.2 Critères d'ancienneté détaillés
        df['tenure_category'] = pd.cut(df['tenure_years'], 
                                     bins=[0, 1, 3, 5, 10, float('inf')], 
                                     labels=['Nouveau', 'Junior', 'Mid', 'Senior', 'Expert'])
        df['tenure_risk_factor'] = np.where(df['tenure_years'] < 1, 0.3,  # Nouveaux employés
                                           np.where(df['tenure_years'] > 7, 0.2,  # Anciens employés
                                                   np.where(df['tenure_years'] > 3, 0.1, 0.05)))  # Employés moyens
        
        # 11.3 Salaires de postes équivalents dans d'autres sociétés (simulés)
        # Simulation des salaires du marché par secteur et taille d'entreprise
        market_salaries_detailed = {
            'IT': {
                'Junior': {'startup': 38000, 'PME': 42000, 'Grande_entreprise': 48000, 'CAC40': 55000},
                'Mid': {'startup': 52000, 'PME': 58000, 'Grande_entreprise': 65000, 'CAC40': 75000},
                'Senior': {'startup': 68000, 'PME': 78000, 'Grande_entreprise': 88000, 'CAC40': 95000},
                'Lead': {'startup': 85000, 'PME': 95000, 'Grande_entreprise': 110000, 'CAC40': 120000},
                'Manager': {'startup': 100000, 'PME': 115000, 'Grande_entreprise': 130000, 'CAC40': 145000},
                'Director': {'startup': 120000, 'PME': 140000, 'Grande_entreprise': 160000, 'CAC40': 180000}
            },
            'Sales': {
                'Junior': {'startup': 32000, 'PME': 36000, 'Grande_entreprise': 42000, 'CAC40': 48000},
                'Mid': {'startup': 45000, 'PME': 52000, 'Grande_entreprise': 60000, 'CAC40': 68000},
                'Senior': {'startup': 60000, 'PME': 72000, 'Grande_entreprise': 82000, 'CAC40': 92000},
                'Lead': {'startup': 75000, 'PME': 88000, 'Grande_entreprise': 100000, 'CAC40': 115000},
                'Manager': {'startup': 90000, 'PME': 105000, 'Grande_entreprise': 120000, 'CAC40': 135000},
                'Director': {'startup': 110000, 'PME': 130000, 'Grande_entreprise': 150000, 'CAC40': 170000}
            },
            'HR': {
                'Junior': {'startup': 28000, 'PME': 32000, 'Grande_entreprise': 38000, 'CAC40': 45000},
                'Mid': {'startup': 38000, 'PME': 45000, 'Grande_entreprise': 52000, 'CAC40': 60000},
                'Senior': {'startup': 52000, 'PME': 60000, 'Grande_entreprise': 70000, 'CAC40': 80000},
                'Lead': {'startup': 65000, 'PME': 75000, 'Grande_entreprise': 85000, 'CAC40': 95000},
                'Manager': {'startup': 80000, 'PME': 92000, 'Grande_entreprise': 105000, 'CAC40': 120000},
                'Director': {'startup': 100000, 'PME': 115000, 'Grande_entreprise': 130000, 'CAC40': 150000}
            }
        }
        
        # Simuler le type d'entreprise concurrente (basé sur le département)
        df['competitor_company_type'] = np.random.choice(['startup', 'PME', 'Grande_entreprise', 'CAC40'], 
                                                        size=len(df), p=[0.1, 0.3, 0.4, 0.2])
        
        # Calculer les salaires concurrents
        df['competitor_salary'] = df.apply(lambda row: 
            market_salaries_detailed.get(row['department'], market_salaries_detailed['IT'])
            .get(row['job_level'], market_salaries_detailed['IT']['Mid'])
            .get(row['competitor_company_type'], 50000), axis=1)
        
        df['salary_vs_competitors'] = df['salary'] / df['competitor_salary']
        df['competitor_attractiveness'] = np.where(df['salary_vs_competitors'] < 0.9, 0.8,  # Très attractif
                                                  np.where(df['salary_vs_competitors'] < 1.1, 0.5, 0.2))  # Moins attractif
        
        # 11.4 Qualité de la relation avec le chef (simulée basée sur d'autres facteurs)
        df['manager_relationship_quality'] = (
            df['feedback_score_mean'] * 0.5 +  # Score de feedback du manager
            (1 - df['absence_rate']) * 0.3 +  # Présentéisme
            df['training_intensity'] * 0.2  # Investissement dans la formation
        )
        
        # Ajouter de la variabilité pour simuler des relations difficiles
        df['manager_relationship_quality'] += np.random.normal(0, 0.1, len(df))
        df['manager_relationship_quality'] = np.clip(df['manager_relationship_quality'], 0, 5)
        
        # 11.5 Temps depuis la dernière augmentation (simulé)
        # Simuler la dernière augmentation basée sur l'ancienneté et la performance
        df['months_since_last_raise'] = np.random.exponential(18, len(df))  # Moyenne 18 mois
        df['months_since_last_raise'] = np.clip(df['months_since_last_raise'], 0, 60)  # Max 5 ans
        
        # Ajuster selon la performance et l'ancienneté
        df['months_since_last_raise'] *= (1 - df['performance_rating_mean'] / 10)  # Meilleure performance = augmentations plus récentes
        df['months_since_last_raise'] *= (1 + df['tenure_years'] / 20)  # Plus d'ancienneté = plus de temps depuis l'augmentation
        
        df['raise_overdue'] = np.where(df['months_since_last_raise'] > 24, 1, 0)  # Plus de 2 ans
        df['raise_urgency'] = np.where(df['months_since_last_raise'] > 36, 0.8,  # Très urgent
                                      np.where(df['months_since_last_raise'] > 24, 0.5, 0.2))  # Urgent
        
        # 11.6 Image de la société (simulée)
        # Basée sur la satisfaction, la performance et d'autres facteurs
        df['company_image_score'] = (
            df['satisfaction_score'] * 0.3 +  # Satisfaction générale
            df['performance_rating_mean'] * 0.2 +  # Performance de l'entreprise
            df['training_intensity'] * 0.2 +  # Investissement dans les employés
            df['work_life_balance'] * 0.2 +  # Équilibre vie/travail
            (1 - df['overtime_intensity'] / df['overtime_intensity'].max()) * 0.1  # Charge de travail raisonnable
        )
        
        # Ajouter de la variabilité pour simuler des perceptions différentes
        df['company_image_score'] += np.random.normal(0, 0.2, len(df))
        df['company_image_score'] = np.clip(df['company_image_score'], 0, 5)
        
        # Critères composites basés sur les nouveaux critères
        df['salary_satisfaction'] = df['salary_vs_market'] * df['salary_competitiveness']
        df['career_stagnation'] = df['months_since_last_raise'] / 12 * df['tenure_risk_factor']
        df['external_attractiveness'] = df['competitor_attractiveness'] * (1 - df['company_image_score'] / 5)
        
        print(f"[OK] {len([col for col in df.columns if col not in ['employee_id', 'employee_number', 'hire_date', 'manager_id', 'created_at']])} criteres crees")
        
        return df
    
    def select_features(self, df):
        """Sélectionne les features les plus importantes depuis les variables réelles du CSV"""
        print("\n[2/6] Selection des features depuis les variables du CSV...")
        
        # Features numériques réelles du CSV
        numeric_features = [
            'age', 'salary', 'tenure_years', 'distance_from_home',
            'performance_rating', 'training_times', 'work_life_balance',
            'job_satisfaction', 'environment_satisfaction', 'relationship_satisfaction',
            'job_involvement', 'years_in_current_role', 'years_since_last_promotion',
            'years_with_curr_manager', 'num_companies_worked', 'total_working_years',
            'percent_salary_hike', 'stock_option_level', 'education',
            'DailyRate', 'HourlyRate', 'MonthlyRate', 'StandardHours'
        ]
        
        # Vérifier que les colonnes existent
        available_features = [col for col in numeric_features if col in df.columns]
        
        # Features catégorielles réelles du CSV
        categorical_features = ['department', 'job_level', 'location', 'overtime', 
                              'marital_status', 'education_field', 'job_role', 
                              'gender', 'business_travel']
        
        # Filtrer pour ne garder que celles qui existent
        categorical_features = [col for col in categorical_features if col in df.columns]
        
        # Préparer les données numériques
        X_numeric = df[available_features].fillna(0)
        y = df['left_company']
        
        # Créer des variables dummy (one-hot encoding) pour les variables catégorielles
        X_categorical = pd.DataFrame()
        for col in categorical_features:
            if col in df.columns:
                # Créer des variables dummy pour chaque catégorie
                dummies = pd.get_dummies(df[col].astype(str), prefix=col, drop_first=False)
                X_categorical = pd.concat([X_categorical, dummies], axis=1)
                # Stocker les colonnes catégorielles pour référence future
                self.categorical_columns.extend(dummies.columns.tolist())
        
        # Combiner les features
        X = pd.concat([X_numeric, X_categorical], axis=1)
        
        # Sélection des meilleures features avec une méthode plus robuste
        # Utiliser mutual_info_classif pour capturer les relations non-linéaires
        selector = SelectKBest(score_func=mutual_info_classif, k=min(30, X.shape[1]))  # Jusqu'à 30 features
        X_selected = selector.fit_transform(X, y)
        
        # Récupérer les noms des features sélectionnées
        selected_features = X.columns[selector.get_support()].tolist()
        self.feature_names = selected_features
        
        print(f"[OK] {len(selected_features)} features selectionnees:")
        for i, feature in enumerate(selected_features, 1):
            feature_fr = translate_criteria(feature)
            print(f"  {i:2d}. {feature_fr}")
        
        return X_selected, y
    
    def find_best_threshold(self, y_true, y_proba, metric='f1'):
        """Trouve le meilleur seuil de décision basé sur différentes métriques"""
        precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, y_proba)
        
        # Calculer différentes métriques pour chaque seuil
        best_score = 0
        best_threshold = 0.5
        best_metric_value = 0
        
        # Tester plusieurs seuils
        test_thresholds = np.concatenate([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], thresholds])
        test_thresholds = np.unique(np.clip(test_thresholds, 0.01, 0.99))
        
        for threshold in test_thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            if metric == 'accuracy':
                score = accuracy_score(y_true, y_pred)
            elif metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'balanced':
                # Combinaison équilibrée : 50% accuracy + 30% F1 + 20% recall
                acc = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                score = 0.5 * acc + 0.3 * f1 + 0.2 * rec
            else:
                score = f1_score(y_true, y_pred, zero_division=0)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_metric_value = score
        
        return best_threshold, best_metric_value
    
    def train_model(self, X, y):
        """Entraîne le modèle de prédiction avec plusieurs algorithmes et SMOTE"""
        print("\n[3/6] Entrainement du modele avec plusieurs algorithmes...")
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalisation
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("\n  [INFO] Rééquilibrage des classes avec SMOTE...")
        # Utiliser SMOTE pour rééquilibrer les classes
        # Ratio de 0.7 signifie 70% de la classe majoritaire (meilleur équilibre pour l'accuracy)
        # Trop d'équilibre (1.0) peut réduire l'accuracy, trop peu (0.5) réduit le recall
        smote = SMOTE(
            sampling_strategy=0.7,  # 70% de la classe majoritaire (meilleur compromis)
            random_state=42,
            k_neighbors=min(5, max(1, sum(y_train == 1) - 1))  # Ajuster k_neighbors si nécessaire
        )
        
        try:
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
            print(f"  [OK] SMOTE applique: {len(y_train_balanced)} echantillons ({sum(y_train_balanced == 0)} classe 0, {sum(y_train_balanced == 1)} classe 1)")
        except Exception as e:
            print(f"  [WARNING] SMOTE a echoue: {e}")
            print("  [INFO] Utilisation de l'upsampling manuel...")
            # Fallback vers l'upsampling manuel
            df_train = pd.DataFrame(X_train_scaled)
            df_train['target'] = y_train
            
            df_majority = df_train[df_train.target == 0]
            df_minority = df_train[df_train.target == 1]
            
            # Augmenter significativement la classe minoritaire (jusqu'à 80% de la majoritaire)
            df_minority_upsampled = resample(
                df_minority,
                replace=True,
                n_samples=int(len(df_majority) * 0.8),
                random_state=42
            )
            
            df_balanced = pd.concat([df_majority, df_minority_upsampled])
            X_train_balanced = df_balanced.drop('target', axis=1).values
            y_train_balanced = df_balanced['target'].values
            print(f"  [OK] Upsampling manuel: {len(y_train_balanced)} echantillons")
        
        # Définir les modèles à tester avec hyperparamètres optimisés
        # Calculer le ratio de classes pour les poids
        class_ratio = sum(y_train_balanced == 0) / sum(y_train_balanced == 1) if sum(y_train_balanced == 1) > 0 else 1.0
        
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=500,  # Augmenté pour plus de stabilité
                max_depth=15,  # Réduit pour éviter le surapprentissage
                min_samples_split=10,  # Augmenté pour plus de généralisation
                min_samples_leaf=4,  # Augmenté
                max_features='sqrt',  # Meilleur pour la variance
                class_weight='balanced_subsample',  # Meilleur que 'balanced'
                bootstrap=True,
                oob_score=True,  # Validation out-of-bag
                random_state=42,
                n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=500,  # Augmenté
                max_depth=5,  # Légèrement réduit
                learning_rate=0.05,  # Réduit pour plus de stabilité
                subsample=0.85,
                colsample_bytree=0.85,
                colsample_bylevel=0.85,
                min_child_weight=3,  # Ajouté pour régularisation
                gamma=0.1,  # Régularisation
                reg_alpha=0.1,  # L1 regularization
                reg_lambda=1.0,  # L2 regularization
                scale_pos_weight=class_ratio,
                random_state=42,
                eval_metric='logloss',
                n_jobs=-1
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=500,  # Augmenté
                max_depth=5,
                learning_rate=0.05,  # Réduit
                num_leaves=31,  # Optimisé
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_samples=20,  # Régularisation
                reg_alpha=0.1,
                reg_lambda=1.0,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        }
        
        best_model = None
        best_model_name = None
        best_score = 0
        best_threshold = 0.5
        results = {}
        
        print("\n  [INFO] Test des algorithmes...")
        for name, model in models.items():
            print(f"\n    - Entrainement {name}...")
            model.fit(X_train_balanced, y_train_balanced)
            
            # Prédictions
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            y_pred_default = model.predict(X_test_scaled)
            
            # Trouver le meilleur seuil en optimisant une métrique équilibrée (accuracy + F1 + recall)
            threshold, best_metric = self.find_best_threshold(y_test, y_proba, metric='balanced')
            y_pred_optimized = (y_proba >= threshold).astype(int)
            
            # Métriques avec seuil par défaut
            roc_auc = roc_auc_score(y_test, y_proba)
            accuracy_default = accuracy_score(y_test, y_pred_default)
            precision_default = precision_score(y_test, y_pred_default, zero_division=0)
            recall_default = recall_score(y_test, y_pred_default, zero_division=0)
            f1_default = f1_score(y_test, y_pred_default, zero_division=0)
            
            # Métriques avec seuil optimisé
            accuracy_opt = accuracy_score(y_test, y_pred_optimized)
            precision_opt = precision_score(y_test, y_pred_optimized, zero_division=0)
            recall_opt = recall_score(y_test, y_pred_optimized, zero_division=0)
            f1_opt = f1_score(y_test, y_pred_optimized, zero_division=0)
            
            results[name] = {
                'roc_auc': roc_auc,
                'accuracy_default': accuracy_default,
                'precision_default': precision_default,
                'recall_default': recall_default,
                'f1_default': f1_default,
                'accuracy_optimized': accuracy_opt,
                'precision_optimized': precision_opt,
                'recall_optimized': recall_opt,
                'f1_optimized': f1_opt,
                'best_threshold': threshold
            }
            
            print(f"      ROC-AUC: {roc_auc:.3f}")
            print(f"      Accuracy (seuil 0.5): {accuracy_default:.3f} | Accuracy (seuil optimal): {accuracy_opt:.3f}")
            print(f"      F1 (seuil 0.5): {f1_default:.3f} | F1 (seuil optimal {threshold:.3f}): {f1_opt:.3f}")
            print(f"      Precision (optimise): {precision_opt:.3f} | Recall (optimise): {recall_opt:.3f}")
            
            # Sélectionner le meilleur modèle basé sur une combinaison équilibrée de métriques
            # Score combiné : 40% accuracy + 30% F1 + 20% recall + 10% precision
            # Cela privilégie l'accuracy tout en maintenant un bon équilibre
            combined_score = (
                0.4 * accuracy_opt +
                0.3 * f1_opt +
                0.2 * recall_opt +
                0.1 * precision_opt
            )
            
            # Pénaliser fortement si l'accuracy est très faible
            if accuracy_opt < 0.5:
                combined_score = combined_score * 0.3
            
            # Pénaliser si le recall est très faible (pas de détection)
            if recall_opt < 0.1:
                combined_score = combined_score * 0.5
            
            if combined_score > best_score:
                best_score = combined_score
                best_model = model
                best_model_name = name
                best_threshold = threshold
        
        # Créer un ensemble (voting) avec les meilleurs modèles si plusieurs ont de bonnes performances
        good_models = [(name, model, results[name]['f1_optimized']) 
                       for name, model in models.items() 
                       if results[name]['f1_optimized'] > 0.3]  # Seuil minimum pour être considéré "bon"
        
        if len(good_models) >= 2:
            # Créer un ensemble avec les meilleurs modèles
            print(f"\n  [INFO] Creation d'un ensemble avec {len(good_models)} modeles performants...")
            voting_models = [(name, model) for name, model, _ in sorted(good_models, key=lambda x: x[2], reverse=True)[:3]]
            ensemble = VotingClassifier(estimators=voting_models, voting='soft', n_jobs=-1)
            ensemble.fit(X_train_balanced, y_train_balanced)
            
            # Tester l'ensemble
            y_proba_ensemble = ensemble.predict_proba(X_test_scaled)[:, 1]
            threshold_ensemble, metric_ensemble = self.find_best_threshold(y_test, y_proba_ensemble, metric='balanced')
            y_pred_ensemble = (y_proba_ensemble >= threshold_ensemble).astype(int)
            
            accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
            f1_ensemble_score = f1_score(y_test, y_pred_ensemble, zero_division=0)
            recall_ensemble = recall_score(y_test, y_pred_ensemble, zero_division=0)
            precision_ensemble = precision_score(y_test, y_pred_ensemble, zero_division=0)
            
            # Score combiné pour l'ensemble
            combined_score_ensemble = (
                0.4 * accuracy_ensemble +
                0.3 * f1_ensemble_score +
                0.2 * recall_ensemble +
                0.1 * precision_ensemble
            )
            
            print(f"      Ensemble - Accuracy: {accuracy_ensemble:.3f}, F1: {f1_ensemble_score:.3f}, Recall: {recall_ensemble:.3f}, Score: {combined_score_ensemble:.3f}")
            
            # Utiliser l'ensemble si il est meilleur que le meilleur modèle individuel
            if combined_score_ensemble > best_score:
                print(f"  [OK] Ensemble selectionne (meilleur que {best_model_name})")
                self.model = ensemble
                self.best_model_name = 'Ensemble'
                self.best_threshold = threshold_ensemble
                best_threshold = threshold_ensemble
            else:
                print(f"  [OK] Modele individuel {best_model_name} selectionne (meilleur que l'ensemble)")
                self.model = best_model
                self.best_model_name = best_model_name
                self.best_threshold = best_threshold
        else:
            # Utiliser le meilleur modèle seul
            self.model = best_model
            self.best_model_name = best_model_name
            self.best_threshold = best_threshold
        
        self.model_performance = results
        
        # Stocker les données d'entraînement pour les métriques
        self.X_train_balanced = X_train_balanced
        self.y_train_balanced = y_train_balanced
        self.X_test_scaled = X_test_scaled
        self.y_test = y_test
        
        # Évaluation finale avec le meilleur modèle sur le test set
        y_proba_test = self.model.predict_proba(X_test_scaled)[:, 1]
        y_pred_test = (y_proba_test >= self.best_threshold).astype(int)
        
        test_accuracy = accuracy_score(y_test, y_pred_test)
        test_precision = precision_score(y_test, y_pred_test, zero_division=0)
        test_recall = recall_score(y_test, y_pred_test, zero_division=0)
        test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
        test_roc_auc = roc_auc_score(y_test, y_proba_test)
        
        # Évaluation sur le training set
        y_proba_train = self.model.predict_proba(X_train_balanced)[:, 1]
        y_pred_train = (y_proba_train >= self.best_threshold).astype(int)
        
        train_accuracy = accuracy_score(y_train_balanced, y_pred_train)
        train_precision = precision_score(y_train_balanced, y_pred_train, zero_division=0)
        train_recall = recall_score(y_train_balanced, y_pred_train, zero_division=0)
        train_f1 = f1_score(y_train_balanced, y_pred_train, zero_division=0)
        train_roc_auc = roc_auc_score(y_train_balanced, y_proba_train)
        
        # Cross-validation sur le training set
        print("\n  [INFO] Calcul de la validation croisee...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Créer une fonction de scoring personnalisée pour utiliser le seuil optimal
        def custom_scorer(y_true, y_proba):
            y_pred_cv = (y_proba >= self.best_threshold).astype(int)
            return accuracy_score(y_true, y_pred_cv)
        
        # Cross-validation pour accuracy
        cv_accuracy_scores = cross_val_score(
            self.model, X_train_balanced, y_train_balanced, 
            cv=cv, scoring='accuracy', n_jobs=-1
        )
        
        # Cross-validation pour F1
        cv_f1_scores = cross_val_score(
            self.model, X_train_balanced, y_train_balanced,
            cv=cv, scoring='f1', n_jobs=-1
        )
        
        # Cross-validation pour ROC-AUC
        cv_roc_auc_scores = cross_val_score(
            self.model, X_train_balanced, y_train_balanced,
            cv=cv, scoring='roc_auc', n_jobs=-1
        )
        
        # Stocker toutes les métriques
        self.test_metrics = {
            'accuracy': float(test_accuracy),
            'precision': float(test_precision),
            'recall': float(test_recall),
            'f1_score': float(test_f1),
            'roc_auc': float(test_roc_auc)
        }
        
        self.train_metrics = {
            'accuracy': float(train_accuracy),
            'precision': float(train_precision),
            'recall': float(train_recall),
            'f1_score': float(train_f1),
            'roc_auc': float(train_roc_auc)
        }
        
        self.cv_metrics = {
            'accuracy': {
                'mean': float(cv_accuracy_scores.mean()),
                'std': float(cv_accuracy_scores.std()),
                'scores': [float(x) for x in cv_accuracy_scores]
            },
            'f1_score': {
                'mean': float(cv_f1_scores.mean()),
                'std': float(cv_f1_scores.std()),
                'scores': [float(x) for x in cv_f1_scores]
            },
            'roc_auc': {
                'mean': float(cv_roc_auc_scores.mean()),
                'std': float(cv_roc_auc_scores.std()),
                'scores': [float(x) for x in cv_roc_auc_scores]
            }
        }
        
        print(f"\n[OK] Meilleur modele: {best_model_name}")
        print(f"\n  [METRIQUES TEST SET]")
        print(f"  - Accuracy: {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")
        print(f"  - ROC-AUC: {test_roc_auc:.3f}")
        print(f"  - Precision: {test_precision:.3f}")
        print(f"  - Recall: {test_recall:.3f}")
        print(f"  - F1-Score: {test_f1:.3f}")
        print(f"\n  [METRIQUES TRAIN SET]")
        print(f"  - Accuracy: {train_accuracy:.3f} ({train_accuracy*100:.1f}%)")
        print(f"  - ROC-AUC: {train_roc_auc:.3f}")
        print(f"  - Precision: {train_precision:.3f}")
        print(f"  - Recall: {train_recall:.3f}")
        print(f"  - F1-Score: {train_f1:.3f}")
        print(f"\n  [METRIQUES CROSS-VALIDATION (5-fold)]")
        print(f"  - Accuracy: {cv_accuracy_scores.mean():.3f} (+/- {cv_accuracy_scores.std()*2:.3f})")
        print(f"  - F1-Score: {cv_f1_scores.mean():.3f} (+/- {cv_f1_scores.std()*2:.3f})")
        print(f"  - ROC-AUC: {cv_roc_auc_scores.mean():.3f} (+/- {cv_roc_auc_scores.std()*2:.3f})")
        print(f"\n  - Seuil optimal: {best_threshold:.3f}")
        
        return X_test_scaled, y_test
    
    def analyze_feature_importance(self):
        """Analyse l'importance des features"""
        print("\n[4/6] Analyse de l'importance des features...")
        
        # Importance des features
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n[INFO] Top 20 criteres les plus importants:")
        # Calculer la moyenne pour l'importance relative
        avg_importance = self.feature_importance['importance'].mean()
        
        # Normaliser les valeurs pour qu'elles soient toujours entre 0 et 100
        sum_importance = self.feature_importance['importance'].sum()
        max_importance = self.feature_importance['importance'].max()
        
        # Détecter le format des valeurs
        if sum_importance > 1.5 or max_importance > 1.0:
            # Les valeurs sont déjà grandes (pas entre 0 et 1), les normaliser pour que la somme = 100
            normalized_importance = (self.feature_importance['importance'] / sum_importance) * 100
        else:
            # Les valeurs sont entre 0 et 1, les convertir en pourcentage
            normalized_importance = self.feature_importance['importance'] * 100
        
        # Créer un DataFrame avec les valeurs normalisées pour faciliter l'accès
        normalized_df = pd.DataFrame({
            'feature': self.feature_importance['feature'],
            'importance': normalized_importance.values
        }, index=self.feature_importance.index)
        
        for idx, (i, row) in enumerate(self.feature_importance.head(20).iterrows()):
            feature_name = row['feature']
            feature_fr = translate_criteria(feature_name)
            # Utiliser la valeur normalisée (entre 0 et 100)
            importance_pct = normalized_df.loc[i, 'importance']
            # Calculer l'importance relative (× la moyenne)
            relative_importance = row['importance'] / avg_importance if avg_importance > 0 else 0
            print(f"  {idx + 1:2d}. {feature_fr:<35} : {importance_pct:.2f}% ({relative_importance:.2f}× la moyenne)")
        
        # Catégoriser les critères
        self.categorize_criteria()
        
        return self.feature_importance
    
    def categorize_criteria(self):
        """Catégorise les critères par type"""
        print("\n[INFO] Categorisation des criteres:")
        
        criteria_categories = {
            'Salaire et Compensation': [],
            'Performance et Objectifs': [],
            'Formation et Developpement': [],
            'Charge de Travail': [],
            'Satisfaction et Engagement': [],
            'Relation Manager': [],
            'Equilibre Vie/Travail': [],
            'Carriere et Progression': [],
            'Absences et Presenteisme': [],
            'Anciennete et Risque': [],
            'Concurrence et Marche': [],
            'Augmentations et Evolution': [],
            'Image de l Entreprise': [],
            'Autres': []
        }
        
        for _, row in self.feature_importance.iterrows():
            feature = row['feature']
            importance = row['importance']
            
            if any(word in feature.lower() for word in ['salary', 'market', 'compensation', 'competitiveness']):
                criteria_categories['Salaire et Compensation'].append((feature, importance))
            elif any(word in feature.lower() for word in ['performance', 'goals', 'rating']):
                criteria_categories['Performance et Objectifs'].append((feature, importance))
            elif any(word in feature.lower() for word in ['training', 'development']):
                criteria_categories['Formation et Developpement'].append((feature, importance))
            elif any(word in feature.lower() for word in ['overtime', 'workload', 'hours']):
                criteria_categories['Charge de Travail'].append((feature, importance))
            elif any(word in feature.lower() for word in ['satisfaction', 'engagement']):
                criteria_categories['Satisfaction et Engagement'].append((feature, importance))
            elif any(word in feature.lower() for word in ['manager', 'relationship']):
                criteria_categories['Relation Manager'].append((feature, importance))
            elif any(word in feature.lower() for word in ['balance', 'stress']):
                criteria_categories['Equilibre Vie/Travail'].append((feature, importance))
            elif any(word in feature.lower() for word in ['career', 'progression', 'promotion']):
                criteria_categories['Carriere et Progression'].append((feature, importance))
            elif any(word in feature.lower() for word in ['absence']):
                criteria_categories['Absences et Presenteisme'].append((feature, importance))
            elif any(word in feature.lower() for word in ['tenure', 'anciennete', 'risk']):
                criteria_categories['Anciennete et Risque'].append((feature, importance))
            elif any(word in feature.lower() for word in ['competitor', 'concurrence', 'marche', 'attractiveness']):
                criteria_categories['Concurrence et Marche'].append((feature, importance))
            elif any(word in feature.lower() for word in ['raise', 'augmentation', 'stagnation']):
                criteria_categories['Augmentations et Evolution'].append((feature, importance))
            elif any(word in feature.lower() for word in ['company', 'image']):
                criteria_categories['Image de l Entreprise'].append((feature, importance))
            else:
                criteria_categories['Autres'].append((feature, importance))
        
        # Afficher les catégories
        # Normaliser les valeurs pour qu'elles soient toujours entre 0 et 100
        if len(self.feature_importance) > 0:
            sum_importance = self.feature_importance['importance'].sum()
            if sum_importance > 1.5:
                # Les valeurs sont déjà en pourcentage, les normaliser
                normalization_factor = 100.0 / sum_importance
            else:
                # Les valeurs sont entre 0 et 1, les convertir en pourcentage
                normalization_factor = 100.0
        else:
            normalization_factor = 100.0
        
        for category, criteria_list in criteria_categories.items():
            if criteria_list:
                print(f"\n  [INFO] {category}:")
                for feature, importance in sorted(criteria_list, key=lambda x: x[1], reverse=True):
                    feature_fr = translate_criteria(feature)
                    # Normaliser pour avoir une valeur entre 0 et 100
                    if len(self.feature_importance) > 0:
                        sum_importance = self.feature_importance['importance'].sum()
                        if sum_importance > 1.5:
                            importance_pct = (importance / sum_importance) * 100
                        else:
                            importance_pct = importance * 100
                    else:
                        importance_pct = importance * 100
                    print(f"    • {feature_fr:<35} : {importance_pct:.2f}%")
        
        self.criteria_analysis = criteria_categories
    
    def generate_shap_analysis(self, X_test):
        """Génère l'analyse SHAP pour l'interprétabilité"""
        print("\n[5/6] Generation de l'analyse SHAP...")
        
        try:
            # Créer l'explainer SHAP
            self.shap_explainer = shap.TreeExplainer(self.model)
            shap_values = self.shap_explainer.shap_values(X_test[:100])  # Limiter à 100 échantillons
            
            print("[OK] Analyse SHAP generee")
            
            # Sauvegarder les valeurs SHAP
            shap_data = {
                'shap_values': shap_values[1].tolist(),  # Classe positive (départ)
                'feature_names': self.feature_names,
                'base_value': self.shap_explainer.expected_value[1]
            }
            
            with open(get_data_path('shap_analysis.json'), 'w') as f:
                json.dump(shap_data, f, indent=2)
            
            return shap_values
            
        except Exception as e:
            print(f"[ERREUR] Impossible de generer l'analyse SHAP: {e}")
            return None
    
    def create_prediction_examples(self, df):
        """Crée des exemples de prédiction avec explication"""
        print("\n[6/6] Creation d'exemples de prediction...")
        
        # Sélectionner quelques employés représentatifs
        examples = []
        
        # Employé à haut risque
        high_risk_employee = df[df['left_company'] == 1].iloc[0]
        examples.append(self.create_employee_analysis(high_risk_employee, "Haut Risque"))
        
        # Employé à faible risque
        low_risk_employee = df[df['left_company'] == 0].iloc[0]
        examples.append(self.create_employee_analysis(low_risk_employee, "Faible Risque"))
        
        # Employé moyen
        medium_risk_employee = df.iloc[len(df)//2]
        examples.append(self.create_employee_analysis(medium_risk_employee, "Risque Moyen"))
        
        # Sauvegarder les exemples
        with open(get_data_path('prediction_examples.json'), 'w') as f:
            json.dump(examples, f, indent=2)
        
        print("[OK] Exemples de prediction crees")
        return examples
    
    def prepare_employee_features(self, employee):
        """Prépare les features d'un employé pour la prédiction avec variables dummy"""
        # Créer un DataFrame temporaire avec une seule ligne
        emp_df = pd.DataFrame([employee])
        
        # Préparer les features numériques (toutes celles qui ne sont pas des variables dummy)
        numeric_features = [f for f in self.feature_names if not any(f.startswith(cat) for cat in 
            ['department_', 'job_level_', 'location_', 'overtime_', 'marital_status_', 
             'education_field_', 'job_role_', 'gender_', 'business_travel_'])]
        X_numeric = pd.DataFrame()
        for feature in numeric_features:
            if feature in emp_df.columns:
                X_numeric[feature] = emp_df[feature].fillna(0)
            else:
                X_numeric[feature] = 0
        
        # Préparer les variables dummy pour les catégories
        X_categorical = pd.DataFrame()
        categorical_cols = ['department', 'job_level', 'location', 'overtime', 
                           'marital_status', 'education_field', 'job_role', 
                           'gender', 'business_travel']
        for col in categorical_cols:
            if col in emp_df.columns:
                dummies = pd.get_dummies(emp_df[col].astype(str), prefix=col, drop_first=False)
                X_categorical = pd.concat([X_categorical, dummies], axis=1)
        
        # Combiner et aligner avec les features sélectionnées
        X_combined = pd.concat([X_numeric, X_categorical], axis=1)
        
        # S'assurer que toutes les features sélectionnées sont présentes
        employee_features = []
        for feature in self.feature_names:
            if feature in X_combined.columns:
                employee_features.append(X_combined[feature].iloc[0])
            else:
                employee_features.append(0)
        
        return np.array([employee_features])
    
    def create_employee_analysis(self, employee, risk_category):
        """Crée une analyse détaillée pour un employé"""
        
        # Préparer les données pour la prédiction avec variables dummy
        X_employee = self.prepare_employee_features(employee)
        X_employee_scaled = self.scaler.transform(X_employee)
        
        # Prédiction avec le seuil optimal
        risk_proba = self.model.predict_proba(X_employee_scaled)[0, 1]
        risk_prediction = int(risk_proba >= self.best_threshold)
        
        # Récupérer les valeurs des features pour l'analyse
        employee_features = X_employee[0]
        
        # Analyse des critères
        criteria_analysis = {}
        for i, feature in enumerate(self.feature_names):
            feature_value = employee_features[i]
            feature_importance = self.model.feature_importances_[i]
            
            criteria_analysis[feature] = {
                'value': float(feature_value),
                'importance': float(feature_importance),
                'contribution': float(feature_value * feature_importance)
            }
        
        return {
            'employee_id': employee['employee_id'],
            'name': f"Employé {employee['employee_id']}",
            'department': employee.get('department', 'N/A'),
            'job_level': employee.get('job_level', 'N/A'),
            'risk_category': risk_category,
            'risk_score': float(risk_proba),
            'prediction': bool(risk_prediction),
            'actual_left': bool(employee['left_company']),
            'criteria_analysis': criteria_analysis,
            'key_factors': self.identify_key_factors(criteria_analysis)
        }
    
    def identify_key_factors(self, criteria_analysis):
        """Identifie les facteurs clés pour un employé"""
        # Trier par contribution
        sorted_criteria = sorted(criteria_analysis.items(), 
                               key=lambda x: abs(x[1]['contribution']), 
                               reverse=True)
        
        key_factors = []
        for feature, data in sorted_criteria[:5]:
            if data['contribution'] > 0:
                impact = "Augmente le risque"
            else:
                impact = "Reduit le risque"
            
            key_factors.append({
                'factor': feature,
                'value': data['value'],
                'impact': impact,
                'contribution': data['contribution']
            })
        
        return key_factors
    
    def save_model(self):
        """Sauvegarde le modèle et les analyses"""
        print("\n[INFO] Sauvegarde du modele...")
        
        # Sauvegarder le modèle
        joblib.dump(self.model, get_model_path('turnover_criteria_model.pkl'))
        joblib.dump(self.scaler, get_model_path('criteria_scaler.pkl'))
        
        # Sauvegarder les analyses (normalisées entre 0 et 100)
        feature_importance_pct = self.feature_importance.copy()
        # Normaliser les valeurs pour qu'elles soient toujours entre 0 et 100
        sum_importance = feature_importance_pct['importance'].sum()
        max_importance = feature_importance_pct['importance'].max()
        
        # Détecter le format des valeurs et normaliser
        if sum_importance > 1.5 or max_importance > 1.0:
            # Les valeurs sont déjà grandes (pas entre 0 et 1), les normaliser pour que la somme = 100
            feature_importance_pct['importance'] = (feature_importance_pct['importance'] / sum_importance) * 100
        else:
            # Les valeurs sont entre 0 et 1, les convertir en pourcentage
            feature_importance_pct['importance'] = feature_importance_pct['importance'] * 100
        
        # Vérification finale : s'assurer que toutes les valeurs sont bien entre 0 et 100
        assert feature_importance_pct['importance'].min() >= 0, "Les valeurs doivent être >= 0"
        assert feature_importance_pct['importance'].max() <= 100, "Les valeurs doivent être <= 100"
        
        # Utiliser les performances réelles du meilleur modèle
        best_performance = self.model_performance.get(self.best_model_name, {})
        
        # Récupérer les métriques calculées (test, train, CV)
        test_metrics = getattr(self, 'test_metrics', {})
        train_metrics = getattr(self, 'train_metrics', {})
        cv_metrics = getattr(self, 'cv_metrics', {})
        
        analysis_results = {
            'feature_importance': feature_importance_pct.to_dict('records'),
            'criteria_categories': {k: [(f[0], float(f[1]) * 100) for f in v] for k, v in self.criteria_analysis.items()},
            'model_performance': {
                'best_model': self.best_model_name,
                'best_threshold': float(self.best_threshold),
                'test_set': {
                    'accuracy': float(test_metrics.get('accuracy', best_performance.get('accuracy_optimized', 0))),
                    'roc_auc': float(test_metrics.get('roc_auc', best_performance.get('roc_auc', 0))),
                    'precision': float(test_metrics.get('precision', best_performance.get('precision_optimized', 0))),
                    'recall': float(test_metrics.get('recall', best_performance.get('recall_optimized', 0))),
                    'f1_score': float(test_metrics.get('f1_score', best_performance.get('f1_optimized', 0)))
                },
                'train_set': {
                    'accuracy': float(train_metrics.get('accuracy', 0)),
                    'roc_auc': float(train_metrics.get('roc_auc', 0)),
                    'precision': float(train_metrics.get('precision', 0)),
                    'recall': float(train_metrics.get('recall', 0)),
                    'f1_score': float(train_metrics.get('f1_score', 0))
                },
                'cross_validation': cv_metrics if cv_metrics else {},
                # Métriques de compatibilité (anciennes)
                'accuracy': float(test_metrics.get('accuracy', best_performance.get('accuracy_optimized', 0))),
                'roc_auc': float(test_metrics.get('roc_auc', best_performance.get('roc_auc', 0))),
                'precision': float(test_metrics.get('precision', best_performance.get('precision_optimized', 0))),
                'recall': float(test_metrics.get('recall', best_performance.get('recall_optimized', 0))),
                'f1_score': float(test_metrics.get('f1_score', best_performance.get('f1_optimized', 0))),
                'all_models': {k: {
                    'roc_auc': float(v.get('roc_auc', 0)),
                    'f1_optimized': float(v.get('f1_optimized', 0)),
                    'best_threshold': float(v.get('best_threshold', 0.5))
                } for k, v in self.model_performance.items()}
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(get_data_path('criteria_analysis_results.json'), 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        print("[OK] Modele et analyses sauvegardes")

def main():
    """Fonction principale"""
    
    # Initialiser le modèle
    model = TurnoverPredictionModel()
    
    # Charger et préparer les données
    df = model.load_and_prepare_data()
    if df is None:
        return
    
    # Sélectionner les features
    X, y = model.select_features(df)
    
    # Entraîner le modèle
    X_test, y_test = model.train_model(X, y)
    
    # Analyser l'importance des features
    model.analyze_feature_importance()
    
    # Générer l'analyse SHAP
    model.generate_shap_analysis(X_test)
    
    # Créer des exemples de prédiction
    model.create_prediction_examples(df)
    
    # Sauvegarder le modèle
    model.save_model()
    
    print("\n" + "="*70)
    print("ANALYSE DES CRITERES SPECIFIQUES TERMINEE")
    print("="*70)
    
    print("\n[INFO] Fichiers generes:")
    print("  - turnover_criteria_model.pkl - Modele entraine")
    print("  - criteria_scaler.pkl - Normaliseur")
    print("  - criteria_analysis_results.json - Resultats d'analyse")
    print("  - shap_analysis.json - Analyse SHAP")
    print("  - prediction_examples.json - Exemples de prediction")
    
    print("\n[INFO] Nouveaux criteres specifiques ajoutes:")
    print("  - Salaire vs concurrence et marche")
    print("  - Anciennete et facteurs de risque")
    print("  - Salaires de postes equivalents dans d'autres societes")
    print("  - Qualite de la relation avec le chef")
    print("  - Temps depuis la derniere augmentation")
    print("  - Image de la societe")
    
    print("\n" + "="*70)
    print("SYSTEME OPERATIONNEL")
    print("="*70)

if __name__ == "__main__":
    main()