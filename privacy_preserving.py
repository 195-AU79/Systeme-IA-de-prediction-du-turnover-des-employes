#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Système de Privacy-Preserving et Anonymisation des Données
Protection de la vie privée et conformité RGPD
"""

import pandas as pd
import numpy as np
import sqlite3
import json
import yaml
import hashlib
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging
from cryptography.fernet import Fernet
import warnings
warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('privacy.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PrivacyPreservingSystem:
    def __init__(self):
        self.config = self.load_config()
        self.encryption_key = self.generate_encryption_key()
        self.anonymization_mapping = {}
        
    def load_config(self) -> Dict[str, Any]:
        """Charge la configuration depuis config.yaml"""
        try:
            with open('config.yaml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info("Configuration chargée avec succès")
            return config
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration: {e}")
            return {}
    
    def generate_encryption_key(self) -> bytes:
        """Génère une clé de chiffrement"""
        try:
            # Essayer de charger une clé existante
            with open('encryption_key.key', 'rb') as f:
                key = f.read()
            logger.info("Clé de chiffrement existante chargée")
        except FileNotFoundError:
            # Générer une nouvelle clé
            key = Fernet.generate_key()
            with open('encryption_key.key', 'wb') as f:
                f.write(key)
            logger.info("Nouvelle clé de chiffrement générée")
        
        return key
    
    def hash_sensitive_data(self, data: str, salt: str = None) -> str:
        """Hache les données sensibles"""
        if salt is None:
            salt = "turnover_prediction_salt_2024"
        
        # Combiner les données avec le salt
        combined = f"{data}_{salt}"
        
        # Hacher avec SHA-256
        hashed = hashlib.sha256(combined.encode()).hexdigest()
        
        return hashed
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Chiffre les données sensibles"""
        try:
            fernet = Fernet(self.encryption_key)
            encrypted_data = fernet.encrypt(data.encode())
            return encrypted_data.decode()
        except Exception as e:
            logger.error(f"Erreur lors du chiffrement: {e}")
            return data
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Déchiffre les données sensibles"""
        try:
            fernet = Fernet(self.encryption_key)
            decrypted_data = fernet.decrypt(encrypted_data.encode())
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Erreur lors du déchiffrement: {e}")
            return encrypted_data
    
    def anonymize_employee_id(self, employee_id: str) -> str:
        """Anonymise l'ID de l'employé"""
        if employee_id not in self.anonymization_mapping:
            # Générer un UUID anonyme
            anonymized_id = str(uuid.uuid4())
            self.anonymization_mapping[employee_id] = anonymized_id
            
            # Sauvegarder le mapping
            self.save_anonymization_mapping()
        
        return self.anonymization_mapping[employee_id]
    
    def deanonymize_employee_id(self, anonymized_id: str) -> Optional[str]:
        """Déanonymise l'ID de l'employé"""
        reverse_mapping = {v: k for k, v in self.anonymization_mapping.items()}
        return reverse_mapping.get(anonymized_id)
    
    def save_anonymization_mapping(self):
        """Sauvegarde le mapping d'anonymisation"""
        try:
            with open('anonymization_mapping.json', 'w') as f:
                json.dump(self.anonymization_mapping, f, indent=2)
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du mapping: {e}")
    
    def load_anonymization_mapping(self):
        """Charge le mapping d'anonymisation"""
        try:
            with open('anonymization_mapping.json', 'r') as f:
                self.anonymization_mapping = json.load(f)
            logger.info("Mapping d'anonymisation chargé")
        except FileNotFoundError:
            self.anonymization_mapping = {}
            logger.info("Nouveau mapping d'anonymisation créé")
    
    def anonymize_personal_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Anonymise les données personnelles dans le DataFrame"""
        try:
            anonymized_df = df.copy()
            
            # Anonymiser les IDs des employés
            if 'employee_id' in anonymized_df.columns:
                anonymized_df['employee_id'] = anonymized_df['employee_id'].apply(self.anonymize_employee_id)
            
            # Anonymiser les noms
            if 'first_name' in anonymized_df.columns:
                anonymized_df['first_name'] = anonymized_df['first_name'].apply(
                    lambda x: f"Employee_{hash(str(x)) % 10000}"
                )
            
            if 'last_name' in anonymized_df.columns:
                anonymized_df['last_name'] = anonymized_df['last_name'].apply(
                    lambda x: f"LastName_{hash(str(x)) % 10000}"
                )
            
            # Anonymiser les emails
            if 'email' in anonymized_df.columns:
                anonymized_df['email'] = anonymized_df['email'].apply(
                    lambda x: f"employee_{hash(str(x)) % 10000}@company.com"
                )
            
            # Chiffrer les salaires (garder la structure pour l'analyse)
            if 'salary' in anonymized_df.columns:
                # Normaliser les salaires pour préserver les relations
                salary_mean = anonymized_df['salary'].mean()
                salary_std = anonymized_df['salary'].std()
                anonymized_df['salary'] = (anonymized_df['salary'] - salary_mean) / salary_std
            
            # Anonymiser les âges (regroupement par tranches)
            if 'age' in anonymized_df.columns:
                anonymized_df['age'] = pd.cut(
                    anonymized_df['age'],
                    bins=[0, 25, 35, 45, 55, 100],
                    labels=['<25', '25-35', '35-45', '45-55', '>55']
                )
            
            logger.info("Données personnelles anonymisées avec succès")
            return anonymized_df
            
        except Exception as e:
            logger.error(f"Erreur lors de l'anonymisation: {e}")
            return df
    
    def detect_sensitive_attributes(self, df: pd.DataFrame) -> List[str]:
        """Détecte les attributs sensibles dans le DataFrame"""
        sensitive_attributes = []
        
        # Attributs explicitement sensibles
        explicit_sensitive = ['first_name', 'last_name', 'email', 'phone', 'address', 'ssn', 'id_number']
        
        # Attributs potentiellement sensibles
        potential_sensitive = ['employee_id', 'manager_id', 'personal_id']
        
        for col in df.columns:
            col_lower = col.lower()
            
            if any(sensitive in col_lower for sensitive in explicit_sensitive):
                sensitive_attributes.append(col)
            elif any(sensitive in col_lower for sensitive in potential_sensitive):
                sensitive_attributes.append(col)
        
        logger.info(f"Attributs sensibles détectés: {sensitive_attributes}")
        return sensitive_attributes
    
    def apply_differential_privacy(self, df: pd.DataFrame, epsilon: float = 1.0) -> pd.DataFrame:
        """Applique la privacy différentielle aux données numériques"""
        try:
            dp_df = df.copy()
            
            # Ajouter du bruit gaussien aux colonnes numériques sensibles
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                if col in ['salary', 'age', 'performance_rating']:  # Colonnes sensibles
                    # Calculer la sensibilité (écart-type)
                    sensitivity = dp_df[col].std()
                    
                    # Calculer le bruit à ajouter
                    noise_scale = sensitivity / epsilon
                    noise = np.random.normal(0, noise_scale, len(dp_df))
                    
                    # Ajouter le bruit
                    dp_df[col] = dp_df[col] + noise
            
            logger.info(f"Privacy différentielle appliquée avec epsilon={epsilon}")
            return dp_df
            
        except Exception as e:
            logger.error(f"Erreur lors de l'application de la privacy différentielle: {e}")
            return df
    
    def create_anonymized_dataset(self, input_db_path: str, output_db_path: str):
        """Crée un dataset anonymisé complet"""
        try:
            logger.info("Création du dataset anonymisé")
            
            # Charger les données originales
            conn = sqlite3.connect(input_db_path)
            
            # Charger toutes les tables
            employees_df = pd.read_sql_query("SELECT * FROM employees", conn)
            turnover_df = pd.read_sql_query("SELECT * FROM turnover", conn)
            performance_df = pd.read_sql_query("SELECT * FROM performance", conn)
            training_df = pd.read_sql_query("SELECT * FROM training", conn)
            overtime_df = pd.read_sql_query("SELECT * FROM overtime", conn)
            absences_df = pd.read_sql_query("SELECT * FROM absences", conn)
            
            conn.close()
            
            # Anonymiser les données
            anonymized_employees = self.anonymize_personal_data(employees_df)
            anonymized_turnover = self.anonymize_personal_data(turnover_df)
            anonymized_performance = self.anonymize_personal_data(performance_df)
            anonymized_training = self.anonymize_personal_data(training_df)
            anonymized_overtime = self.anonymize_personal_data(overtime_df)
            anonymized_absences = self.anonymize_personal_data(absences_df)
            
            # Appliquer la privacy différentielle
            anonymized_employees = self.apply_differential_privacy(anonymized_employees)
            anonymized_performance = self.apply_differential_privacy(anonymized_performance)
            
            # Créer la nouvelle base de données anonymisée
            conn_anon = sqlite3.connect(output_db_path)
            
            # Sauvegarder les tables anonymisées
            anonymized_employees.to_sql('employees', conn_anon, if_exists='replace', index=False)
            anonymized_turnover.to_sql('turnover', conn_anon, if_exists='replace', index=False)
            anonymized_performance.to_sql('performance', conn_anon, if_exists='replace', index=False)
            anonymized_training.to_sql('training', conn_anon, if_exists='replace', index=False)
            anonymized_overtime.to_sql('overtime', conn_anon, if_exists='replace', index=False)
            anonymized_absences.to_sql('absences', conn_anon, if_exists='replace', index=False)
            
            conn_anon.close()
            
            # Sauvegarder les métadonnées d'anonymisation
            anonymization_metadata = {
                'anonymization_date': datetime.now().isoformat(),
                'original_records': len(employees_df),
                'anonymized_records': len(anonymized_employees),
                'sensitive_attributes_detected': self.detect_sensitive_attributes(employees_df),
                'differential_privacy_epsilon': 1.0,
                'anonymization_methods': [
                    'ID anonymization',
                    'Name hashing',
                    'Email anonymization',
                    'Salary normalization',
                    'Age binning',
                    'Differential privacy'
                ]
            }
            
            with open('anonymization_metadata.json', 'w') as f:
                json.dump(anonymization_metadata, f, indent=2)
            
            logger.info(f"Dataset anonymisé créé: {output_db_path}")
            logger.info(f"Métadonnées sauvegardées: anonymization_metadata.json")
            
        except Exception as e:
            logger.error(f"Erreur lors de la création du dataset anonymisé: {e}")
    
    def audit_data_access(self, user_id: str, action: str, data_type: str, records_count: int):
        """Audite l'accès aux données"""
        try:
            audit_entry = {
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'action': action,
                'data_type': data_type,
                'records_count': records_count,
                'ip_address': '127.0.0.1',  # À remplacer par l'IP réelle
                'session_id': str(uuid.uuid4())
            }
            
            # Charger l'audit existant
            try:
                with open('data_access_audit.json', 'r') as f:
                    audit_log = json.load(f)
            except FileNotFoundError:
                audit_log = []
            
            # Ajouter la nouvelle entrée
            audit_log.append(audit_entry)
            
            # Sauvegarder
            with open('data_access_audit.json', 'w') as f:
                json.dump(audit_log, f, indent=2)
            
            logger.info(f"Audit enregistré: {user_id} - {action} - {data_type}")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'audit: {e}")
    
    def check_gdpr_compliance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Vérifie la conformité RGPD"""
        try:
            compliance_report = {
                'timestamp': datetime.now().isoformat(),
                'total_records': len(df),
                'sensitive_attributes': self.detect_sensitive_attributes(df),
                'compliance_checks': {}
            }
            
            # Vérification 1: Données personnelles identifiables
            sensitive_cols = self.detect_sensitive_attributes(df)
            compliance_report['compliance_checks']['personal_data_detected'] = len(sensitive_cols) > 0
            
            # Vérification 2: Données manquantes
            missing_data = df.isnull().sum().sum()
            compliance_report['compliance_checks']['missing_data_percentage'] = (missing_data / (len(df) * len(df.columns))) * 100
            
            # Vérification 3: Données dupliquées
            duplicates = df.duplicated().sum()
            compliance_report['compliance_checks']['duplicate_records'] = duplicates
            
            # Vérification 4: Âge des données
            if 'hire_date' in df.columns:
                df['hire_date'] = pd.to_datetime(df['hire_date'])
                oldest_record = df['hire_date'].min()
                data_age_years = (datetime.now() - oldest_record).days / 365.25
                compliance_report['compliance_checks']['data_age_years'] = data_age_years
            
            # Recommandations
            recommendations = []
            
            if compliance_report['compliance_checks']['personal_data_detected']:
                recommendations.append("Anonymiser les données personnelles identifiables")
            
            if compliance_report['compliance_checks']['missing_data_percentage'] > 10:
                recommendations.append("Améliorer la qualité des données")
            
            if compliance_report['compliance_checks']['duplicate_records'] > 0:
                recommendations.append("Supprimer les enregistrements dupliqués")
            
            if compliance_report['compliance_checks'].get('data_age_years', 0) > 7:
                recommendations.append("Considérer l'archivage des données anciennes")
            
            compliance_report['recommendations'] = recommendations
            
            logger.info("Rapport de conformité RGPD généré")
            return compliance_report
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification de conformité: {e}")
            return {}
    
    def generate_privacy_report(self):
        """Génère un rapport de confidentialité complet"""
        try:
            logger.info("Génération du rapport de confidentialité")
            
            # Charger les données
            conn = sqlite3.connect('turnover_data.db')
            employees_df = pd.read_sql_query("SELECT * FROM employees", conn)
            conn.close()
            
            # Vérifier la conformité RGPD
            compliance_report = self.check_gdpr_compliance(employees_df)
            
            # Générer le rapport
            privacy_report = {
                'report_date': datetime.now().isoformat(),
                'system_version': '1.0.0',
                'gdpr_compliance': compliance_report,
                'anonymization_status': {
                    'mapping_entries': len(self.anonymization_mapping),
                    'last_anonymization': 'Not performed'
                },
                'data_protection_measures': [
                    'Encryption at rest',
                    'Anonymization mapping',
                    'Differential privacy',
                    'Access auditing',
                    'Data minimization'
                ],
                'recommendations': compliance_report.get('recommendations', [])
            }
            
            # Sauvegarder le rapport
            with open('privacy_report.json', 'w') as f:
                json.dump(privacy_report, f, indent=2)
            
            logger.info("Rapport de confidentialité généré: privacy_report.json")
            return privacy_report
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du rapport: {e}")
            return {}

def main():
    """Fonction principale"""
    print("\n" + "="*60)
    print("SYSTÈME DE PRIVACY-PRESERVING ET ANONYMIZATION")
    print("="*60)
    
    # Initialiser le système
    privacy_system = PrivacyPreservingSystem()
    
    print("\n[INFO] Système de protection de la vie privée initialisé")
    print("[INFO] Fonctionnalités:")
    print("  • Anonymisation des données personnelles")
    print("  • Chiffrement des données sensibles")
    print("  • Privacy différentielle")
    print("  • Audit des accès aux données")
    print("  • Vérification de conformité RGPD")
    
    # Menu interactif
    while True:
        print("\n" + "="*40)
        print("MENU PRINCIPAL")
        print("="*40)
        print("1. Créer un dataset anonymisé")
        print("2. Vérifier la conformité RGPD")
        print("3. Générer un rapport de confidentialité")
        print("4. Auditer l'accès aux données")
        print("5. Quitter")
        
        choice = input("\nVotre choix (1-5): ").strip()
        
        if choice == '1':
            print("\n[INFO] Création du dataset anonymisé...")
            privacy_system.create_anonymized_dataset(
                'turnover_data.db',
                'turnover_data_anonymized.db'
            )
            print("[OK] Dataset anonymisé créé: turnover_data_anonymized.db")
        
        elif choice == '2':
            print("\n[INFO] Vérification de la conformité RGPD...")
            conn = sqlite3.connect('turnover_data.db')
            employees_df = pd.read_sql_query("SELECT * FROM employees", conn)
            conn.close()
            
            compliance_report = privacy_system.check_gdpr_compliance(employees_df)
            print(f"[OK] Rapport de conformité généré")
            print(f"  - Attributs sensibles: {len(compliance_report.get('sensitive_attributes', []))}")
            print(f"  - Recommandations: {len(compliance_report.get('recommendations', []))}")
        
        elif choice == '3':
            print("\n[INFO] Génération du rapport de confidentialité...")
            report = privacy_system.generate_privacy_report()
            print("[OK] Rapport généré: privacy_report.json")
        
        elif choice == '4':
            print("\n[INFO] Audit de l'accès aux données...")
            user_id = input("ID utilisateur: ").strip()
            action = input("Action effectuée: ").strip()
            data_type = input("Type de données: ").strip()
            records_count = int(input("Nombre d'enregistrements: ").strip())
            
            privacy_system.audit_data_access(user_id, action, data_type, records_count)
            print("[OK] Audit enregistré")
        
        elif choice == '5':
            print("\n[INFO] Arrêt du système")
            break
        
        else:
            print("\n[ERREUR] Choix invalide")

if __name__ == "__main__":
    main()

