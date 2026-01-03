#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Système de Continuous Learning et Retraining Automatique
Mise à jour automatique des modèles avec de nouvelles données
"""

import pandas as pd
import numpy as np
import sqlite3
import joblib
import json
import yaml
from datetime import datetime, timedelta
import schedule
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from paths import get_config_path, get_model_path, get_data_path, get_db_path
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.utils import resample

# Ensemble Learning
import xgboost as xgb
import lightgbm as lgb

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('continuous_learning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ContinuousLearningSystem:
    def __init__(self):
        self.config = self.load_config()
        self.model_performance_history = []
        self.retraining_active = False
        self.last_retraining_date = None
        
    def load_config(self) -> Dict[str, Any]:
        """Charge la configuration depuis config.yaml"""
        try:
            with open(get_config_path(), 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info("Configuration chargée avec succès")
            return config
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration: {e}")
            return {}
    
    def load_current_models(self) -> Dict[str, Any]:
        """Charge les modèles actuels"""
        try:
            models = joblib.load(get_model_path('all_models.pkl'))
            scaler = joblib.load(get_model_path('scaler.pkl'))
            le_dept = joblib.load(get_model_path('label_encoder_department.pkl'))
            le_level = joblib.load(get_model_path('label_encoder_job_level.pkl'))
            
            logger.info("Modèles actuels chargés avec succès")
            return {
                'models': models,
                'scaler': scaler,
                'label_encoders': {'department': le_dept, 'job_level': le_level}
            }
        except Exception as e:
            logger.error(f"Erreur lors du chargement des modèles actuels: {e}")
            return None
    
    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Récupère les données d'entraînement depuis la base SQLite"""
        try:
            conn = sqlite3.connect(get_db_path())
            
            # Charger les données
            employees_df = pd.read_sql_query("SELECT * FROM employees", conn)
            turnover_df = pd.read_sql_query("SELECT * FROM turnover", conn)
            performance_df = pd.read_sql_query("SELECT * FROM performance", conn)
            training_df = pd.read_sql_query("SELECT * FROM training", conn)
            overtime_df = pd.read_sql_query("SELECT * FROM overtime", conn)
            absences_df = pd.read_sql_query("SELECT * FROM absences", conn)
            
            conn.close()
            
            # Préparer les données (même logique que main.py)
            employees_df['left_company'] = employees_df['employee_id'].isin(turnover_df['employee_id']).astype(int)
            employees_df['hire_date'] = pd.to_datetime(employees_df['hire_date'])
            employees_df['tenure_years'] = (datetime.now() - employees_df['hire_date']).dt.days / 365.25
            
            # Agréger les données
            perf_agg = performance_df.groupby('employee_id').agg({
                'performance_rating': 'mean',
                'goals_achieved': 'mean',
                'feedback_score': 'mean'
            }).reset_index()
            
            training_agg = training_df.groupby('employee_id').agg({
                'hours_completed': 'sum',
                'score': 'mean'
            }).reset_index()
            
            overtime_agg = overtime_df.groupby('employee_id').agg({
                'hours': 'sum'
            }).reset_index()
            
            absences_agg = absences_df.groupby('employee_id').agg({
                'duration_days': 'count'
            }).reset_index()
            
            # Fusionner
            df = employees_df.copy()
            df = df.merge(perf_agg, on='employee_id', how='left')
            df = df.merge(training_agg, on='employee_id', how='left')
            df = df.merge(overtime_agg, on='employee_id', how='left')
            df = df.merge(absences_agg, on='employee_id', how='left')
            
            df = df.fillna(0)
            
            # Renommer et créer les colonnes
            df['training_hours'] = df['hours_completed']
            df['overtime_hours'] = df['hours']
            df['projects_count'] = df['duration_days']
            df['satisfaction_score'] = df['feedback_score'].fillna(3.5)
            df['work_life_balance'] = 3.5
            df['last_promotion_months'] = df['tenure_years'] * 12 * 0.5
            df['manager_change_count'] = 0
            
            # Encoder les variables catégorielles
            le_dept = LabelEncoder()
            le_level = LabelEncoder()
            
            df['department_encoded'] = le_dept.fit_transform(df['department'])
            df['job_level_encoded'] = le_level.fit_transform(df['job_level'])
            
            # Sélectionner les features
            feature_cols = [
                'age', 'tenure_years', 'salary', 'last_promotion_months',
                'performance_rating', 'training_hours', 'overtime_hours', 
                'projects_count', 'satisfaction_score', 'work_life_balance',
                'manager_change_count', 'goals_achieved', 'feedback_score',
                'department_encoded', 'job_level_encoded'
            ]
            
            X = df[feature_cols].values
            y = df['left_company'].values
            
            logger.info(f"Données d'entraînement préparées: {X.shape[0]} employés, {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            logger.error(f"Erreur lors de la préparation des données d'entraînement: {e}")
            return None, None
    
    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Entraîne tous les modèles avec les nouvelles données"""
        try:
            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Normalisation
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Équilibrage des classes
            df_train = pd.DataFrame(X_train_scaled)
            df_train['target'] = y_train
            
            df_majority = df_train[df_train.target == 0]
            df_minority = df_train[df_train.target == 1]
            
            df_minority_upsampled = resample(
                df_minority,
                replace=True,
                n_samples=int(len(df_majority) * 0.5),
                random_state=42
            )
            
            df_balanced = pd.concat([df_majority, df_minority_upsampled])
            X_train_balanced = df_balanced.drop('target', axis=1).values
            y_train_balanced = df_balanced['target'].values
            
            # Configuration des modèles
            model_config = self.config.get('model_config', {}).get('ensemble', {}).get('models', [])
            
            # Random Forest
            rf_params = model_config[0].get('random_forest', {})
            rf_model = RandomForestClassifier(
                n_estimators=rf_params.get('n_estimators', 200),
                max_depth=rf_params.get('max_depth', 15),
                min_samples_split=rf_params.get('min_samples_split', 10),
                class_weight='balanced',
                random_state=42
            )
            
            # XGBoost
            xgb_params = model_config[1].get('xgboost', {})
            xgb_model = xgb.XGBClassifier(
                n_estimators=xgb_params.get('n_estimators', 200),
                learning_rate=xgb_params.get('learning_rate', 0.1),
                max_depth=xgb_params.get('max_depth', 6),
                random_state=42,
                eval_metric='logloss'
            )
            
            # LightGBM
            lgb_params = model_config[2].get('lightgbm', {})
            lgb_model = lgb.LGBMClassifier(
                n_estimators=lgb_params.get('n_estimators', 200),
                num_leaves=lgb_params.get('num_leaves', 31),
                random_state=42,
                verbose=-1
            )
            
            # Gradient Boosting
            gb_params = model_config[3].get('gradient_boosting', {})
            from sklearn.ensemble import GradientBoostingClassifier
            gb_model = GradientBoostingClassifier(
                n_estimators=gb_params.get('n_estimators', 150),
                learning_rate=gb_params.get('learning_rate', 0.1),
                random_state=42
            )
            
            # Entraînement des modèles
            logger.info("Entraînement des modèles individuels...")
            rf_model.fit(X_train_balanced, y_train_balanced)
            xgb_model.fit(X_train_balanced, y_train_balanced)
            lgb_model.fit(X_train_balanced, y_train_balanced)
            gb_model.fit(X_train_balanced, y_train_balanced)
            
            # Créer l'ensemble
            voting_type = self.config.get('model_config', {}).get('ensemble', {}).get('voting_type', 'soft')
            ensemble_model = VotingClassifier(
                estimators=[
                    ('rf', rf_model),
                    ('xgb', xgb_model),
                    ('lgb', lgb_model),
                    ('gb', gb_model)
                ],
                voting=voting_type
            )
            
            logger.info("Entraînement de l'ensemble...")
            ensemble_model.fit(X_train_balanced, y_train_balanced)
            
            # Évaluation des modèles
            models = {
                'Random Forest': rf_model,
                'XGBoost': xgb_model,
                'LightGBM': lgb_model,
                'Gradient Boosting': gb_model,
                'Ensemble': ensemble_model
            }
            
            results = {}
            for name, model in models.items():
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                results[name] = {
                    'roc_auc': roc_auc_score(y_test, y_proba),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1_score': f1_score(y_test, y_pred)
                }
            
            # Trouver le meilleur modèle
            best_model_name = max(results.keys(), key=lambda x: results[x]['roc_auc'])
            best_model = models[best_model_name]
            
            logger.info(f"Meilleur modèle: {best_model_name} (ROC-AUC: {results[best_model_name]['roc_auc']:.3f})")
            
            return {
                'models': models,
                'best_model': best_model,
                'best_model_name': best_model_name,
                'scaler': scaler,
                'results': results,
                'feature_cols': [
                    'age', 'tenure_years', 'salary', 'last_promotion_months',
                    'performance_rating', 'training_hours', 'overtime_hours', 
                    'projects_count', 'satisfaction_score', 'work_life_balance',
                    'manager_change_count', 'goals_achieved', 'feedback_score',
                    'department_encoded', 'job_level_encoded'
                ]
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement des modèles: {e}")
            return None
    
    def evaluate_model_performance(self, new_results: Dict[str, Any], old_results: Optional[Dict[str, Any]] = None) -> bool:
        """Évalue si les nouveaux modèles sont meilleurs que les anciens"""
        try:
            if old_results is None:
                logger.info("Premier entraînement - acceptation automatique")
                return True
            
            # Comparer les performances
            improvement_threshold = self.config.get('continuous_learning', {}).get('improvement_threshold', 0.01)
            
            new_best_auc = max(new_results.values(), key=lambda x: x['roc_auc'])['roc_auc']
            old_best_auc = max(old_results.values(), key=lambda x: x['roc_auc'])['roc_auc']
            
            improvement = new_best_auc - old_best_auc
            
            logger.info(f"Amélioration du ROC-AUC: {improvement:.4f}")
            
            if improvement >= improvement_threshold:
                logger.info("Amélioration significative détectée - nouveaux modèles acceptés")
                return True
            else:
                logger.info("Amélioration insuffisante - anciens modèles conservés")
                return False
                
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation des performances: {e}")
            return False
    
    def save_models(self, training_results: Dict[str, Any]):
        """Sauvegarde les nouveaux modèles"""
        try:
            # Sauvegarder tous les modèles
            joblib.dump(training_results['models'], get_model_path('all_models.pkl'))
            
            # Sauvegarder le meilleur modèle
            joblib.dump(training_results['best_model'], get_model_path('best_ensemble_model.pkl'))
            
            # Sauvegarder le scaler
            joblib.dump(training_results['scaler'], get_model_path('scaler.pkl'))
            
            # Sauvegarder les résultats
            with open(get_data_path('continuous_learning_results.json'), 'w') as f:
                json.dump({
                    'model_performance': training_results['results'],
                    'best_model': training_results['best_model_name'],
                    'retraining_date': datetime.now().isoformat(),
                    'feature_columns': training_results['feature_cols']
                }, f, indent=2)
            
            logger.info("Modèles sauvegardés avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des modèles: {e}")
    
    def backup_current_models(self):
        """Crée une sauvegarde des modèles actuels"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Sauvegarder les modèles actuels
            try:
                current_models = joblib.load(get_model_path('all_models.pkl'))
                joblib.dump(current_models, get_model_path(f'backup_models_{timestamp}.pkl'))
            except:
                pass
            
            try:
                current_scaler = joblib.load(get_model_path('scaler.pkl'))
                joblib.dump(current_scaler, get_model_path(f'backup_scaler_{timestamp}.pkl'))
            except:
                pass
            
            logger.info(f"Sauvegarde créée avec timestamp: {timestamp}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la création de la sauvegarde: {e}")
    
    def check_data_drift(self) -> bool:
        """Vérifie s'il y a un drift dans les données"""
        try:
            # Charger les données actuelles
            X, y = self.get_training_data()
            if X is None:
                return False
            
            # Charger les données de référence (dernière sauvegarde)
            try:
                with open(get_data_path('data_reference.json'), 'r') as f:
                    reference_data = json.load(f)
                
                # Comparer les statistiques
                current_stats = {
                    'mean_age': np.mean(X[:, 0]),
                    'mean_salary': np.mean(X[:, 2]),
                    'mean_tenure': np.mean(X[:, 1]),
                    'turnover_rate': np.mean(y)
                }
                
                drift_threshold = self.config.get('continuous_learning', {}).get('drift_threshold', 0.1)
                
                for key, current_value in current_stats.items():
                    reference_value = reference_data.get(key, current_value)
                    drift = abs(current_value - reference_value) / reference_value if reference_value != 0 else 0
                    
                    if drift > drift_threshold:
                        logger.info(f"Drift détecté pour {key}: {drift:.3f}")
                        return True
                
                logger.info("Aucun drift significatif détecté")
                return False
                
            except FileNotFoundError:
                # Première exécution - sauvegarder les données de référence
                reference_stats = {
                    'mean_age': np.mean(X[:, 0]),
                    'mean_salary': np.mean(X[:, 2]),
                    'mean_tenure': np.mean(X[:, 1]),
                    'turnover_rate': np.mean(y)
                }
                
                with open(get_data_path('data_reference.json'), 'w') as f:
                    json.dump(reference_stats, f, indent=2)
                
                logger.info("Données de référence sauvegardées")
                return False
                
        except Exception as e:
            logger.error(f"Erreur lors de la vérification du drift: {e}")
            return False
    
    def retrain_models(self):
        """Effectue le retraining des modèles"""
        try:
            logger.info("Début du retraining des modèles")
            
            # Vérifier le drift des données
            if not self.check_data_drift():
                logger.info("Aucun drift détecté - retraining non nécessaire")
                return
            
            # Charger les modèles actuels pour comparaison
            current_models = self.load_current_models()
            old_results = None
            
            if current_models:
                # Charger les anciens résultats
                try:
                    with open(get_data_path('continuous_learning_results.json'), 'r') as f:
                        old_data = json.load(f)
                        old_results = old_data.get('model_performance', {})
                except:
                    pass
            
            # Récupérer les données d'entraînement
            X, y = self.get_training_data()
            if X is None or y is None:
                logger.error("Impossible de récupérer les données d'entraînement")
                return
            
            # Entraîner les nouveaux modèles
            training_results = self.train_models(X, y)
            if training_results is None:
                logger.error("Échec de l'entraînement des modèles")
                return
            
            # Évaluer les performances
            if self.evaluate_model_performance(training_results['results'], old_results):
                # Créer une sauvegarde
                self.backup_current_models()
                
                # Sauvegarder les nouveaux modèles
                self.save_models(training_results)
                
                # Mettre à jour la date de retraining
                self.last_retraining_date = datetime.now()
                
                logger.info("Retraining terminé avec succès")
            else:
                logger.info("Retraining terminé - anciens modèles conservés")
                
        except Exception as e:
            logger.error(f"Erreur lors du retraining: {e}")
    
    def start_continuous_learning(self):
        """Démarre le système de continuous learning"""
        logger.info("Démarrage du système de continuous learning")
        
        # Configuration de la fréquence depuis config.yaml
        retrain_frequency = self.config.get('environments', {}).get('production', {}).get('retrain_frequency', 'monthly')
        
        if retrain_frequency == 'daily':
            schedule.every().day.at("02:00").do(self.retrain_models)
        elif retrain_frequency == 'weekly':
            schedule.every().monday.at("02:00").do(self.retrain_models)
        elif retrain_frequency == 'monthly':
            schedule.every().month.do(self.retrain_models)
        else:
            # Par défaut, toutes les semaines
            schedule.every().monday.at("02:00").do(self.retrain_models)
        
        self.retraining_active = True
        
        logger.info(f"Retraining programmé: {retrain_frequency}")
        
        while self.retraining_active:
            schedule.run_pending()
            time.sleep(3600)  # Vérifier toutes les heures
    
    def stop_continuous_learning(self):
        """Arrête le système de continuous learning"""
        logger.info("Arrêt du système de continuous learning")
        self.retraining_active = False

def main():
    """Fonction principale"""
    print("\n" + "="*60)
    print("SYSTÈME DE CONTINUOUS LEARNING ET RETRAINING")
    print("="*60)
    
    # Initialiser le système
    cl_system = ContinuousLearningSystem()
    
    print("\n[INFO] Système de continuous learning initialisé")
    print("[INFO] Fonctionnalités:")
    print("  • Détection automatique du drift des données")
    print("  • Retraining automatique des modèles")
    print("  • Évaluation des performances")
    print("  • Sauvegarde automatique des modèles")
    
    try:
        # Démarrer le continuous learning
        cl_system.start_continuous_learning()
    except KeyboardInterrupt:
        print("\n[INFO] Arrêt du système de continuous learning")
        cl_system.stop_continuous_learning()

if __name__ == "__main__":
    main()
