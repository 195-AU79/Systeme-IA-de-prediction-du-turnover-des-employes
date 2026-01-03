#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Système de Monitoring et Alertes Automatiques pour le Turnover
Surveillance en temps réel et notifications automatiques
"""

import pandas as pd
import numpy as np
import sqlite3
import joblib
import json
import yaml
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import schedule
import time
import logging
from typing import List, Dict, Any, Optional
import requests
from dataclasses import dataclass
import threading
from paths import get_config_path, get_model_path, get_data_path, get_db_path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Alert:
    employee_id: str
    employee_name: str
    department: str
    job_level: str
    risk_score: float
    risk_level: str
    alert_type: str
    timestamp: datetime
    recommendations: List[str]

class MonitoringSystem:
    def __init__(self):
        self.config = self.load_config()
        self.model_manager = self.load_models()
        self.alert_history = []
        self.monitoring_active = False
        
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
    
    def load_models(self):
        """Charge les modèles de prédiction"""
        try:
            models = joblib.load(get_model_path('all_models.pkl'))
            scaler = joblib.load(get_model_path('scaler.pkl'))
            le_dept = joblib.load(get_model_path('label_encoder_department.pkl'))
            le_level = joblib.load(get_model_path('label_encoder_job_level.pkl'))
            
            logger.info("Modèles chargés avec succès")
            return {
                'models': models,
                'scaler': scaler,
                'label_encoders': {'department': le_dept, 'job_level': le_level}
            }
        except Exception as e:
            logger.error(f"Erreur lors du chargement des modèles: {e}")
            return None
    
    def get_employee_data(self) -> pd.DataFrame:
        """Récupère les données des employés depuis la base SQLite"""
        try:
            conn = sqlite3.connect(get_db_path())
            
            # Charger les données de base
            employees_df = pd.read_sql_query("SELECT * FROM employees", conn)
            turnover_df = pd.read_sql_query("SELECT * FROM turnover", conn)
            performance_df = pd.read_sql_query("SELECT * FROM performance", conn)
            training_df = pd.read_sql_query("SELECT * FROM training", conn)
            overtime_df = pd.read_sql_query("SELECT * FROM overtime", conn)
            absences_df = pd.read_sql_query("SELECT * FROM absences", conn)
            
            conn.close()
            
            # Préparer les données comme dans main.py
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
            df['department_encoded'] = self.model_manager['label_encoders']['department'].transform(df['department'])
            df['job_level_encoded'] = self.model_manager['label_encoders']['job_level'].transform(df['job_level'])
            
            logger.info(f"Données de {len(df)} employés chargées")
            return df
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {e}")
            return pd.DataFrame()
    
    def predict_employee_risk(self, employee_row: pd.Series) -> Dict[str, Any]:
        """Prédit le risque de turnover pour un employé"""
        try:
            feature_cols = [
                'age', 'tenure_years', 'salary', 'last_promotion_months',
                'performance_rating', 'training_hours', 'overtime_hours', 
                'projects_count', 'satisfaction_score', 'work_life_balance',
                'manager_change_count', 'goals_achieved', 'feedback_score',
                'department_encoded', 'job_level_encoded'
            ]
            
            # Préparer les données
            X = employee_row[feature_cols].values.reshape(1, -1)
            X_scaled = self.model_manager['scaler'].transform(X)
            
            # Prédiction avec le meilleur modèle
            best_model = self.model_manager['models']['best']
            risk_proba = best_model.predict_proba(X_scaled)[0, 1]
            risk_prediction = best_model.predict(X_scaled)[0]
            
            # Déterminer le niveau de risque
            thresholds = self.config.get('model_config', {}).get('thresholds', {})
            if risk_proba < thresholds.get('low_risk', 0.2):
                risk_level = "Très Faible"
            elif risk_proba < thresholds.get('medium_risk', 0.4):
                risk_level = "Faible"
            elif risk_proba < thresholds.get('high_risk', 0.6):
                risk_level = "Moyen"
            elif risk_proba < thresholds.get('critical_risk', 0.8):
                risk_level = "Élevé"
            else:
                risk_level = "Critique"
            
            return {
                'risk_score': float(risk_proba),
                'risk_level': risk_level,
                'will_leave': bool(risk_prediction),
                'confidence': float(max(risk_proba, 1-risk_proba))
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction pour {employee_row['employee_id']}: {e}")
            return {'risk_score': 0.0, 'risk_level': 'Inconnu', 'will_leave': False, 'confidence': 0.0}
    
    def generate_recommendations(self, risk_score: float, employee_row: pd.Series) -> List[str]:
        """Génère des recommandations basées sur le score de risque"""
        recommendations = []
        
        if risk_score > 0.8:
            recommendations.extend([
                "🚨 ALERTE CRITIQUE - Entretien urgent avec le manager",
                "📊 Enquête de satisfaction immédiate",
                "💰 Évaluation de l'augmentation de salaire",
                "🎯 Plan de rétention personnalisé"
            ])
        elif risk_score > 0.6:
            recommendations.extend([
                "⚠️ Entretien avec le manager dans les 48h",
                "📈 Discussion sur l'évolution de carrière",
                "🏆 Programme de reconnaissance",
                "⚖️ Révision de la charge de travail"
            ])
        elif risk_score > 0.4:
            recommendations.extend([
                "💬 Entretien de suivi mensuel",
                "📚 Plan de développement des compétences",
                "🤝 Amélioration de l'environnement de travail",
                "📅 Évaluation des objectifs"
            ])
        else:
            recommendations.extend([
                "✅ Maintenir la satisfaction actuelle",
                "🎯 Continuer le développement professionnel",
                "👥 Renforcer l'engagement d'équipe",
                "📊 Monitoring régulier"
            ])
        
        return recommendations
    
    def check_alerts(self) -> List[Alert]:
        """Vérifie les alertes pour tous les employés"""
        alerts = []
        
        try:
            # Récupérer les données
            df = self.get_employee_data()
            if df.empty:
                return alerts
            
            # Seuil d'alerte depuis la configuration
            alert_threshold = self.config.get('monitoring', {}).get('risk_score_threshold', 0.6)
            
            # Analyser chaque employé
            for _, employee in df.iterrows():
                # Prédiction du risque
                prediction = self.predict_employee_risk(employee)
                
                # Vérifier si une alerte est nécessaire
                if prediction['risk_score'] >= alert_threshold:
                    recommendations = self.generate_recommendations(prediction['risk_score'], employee)
                    
                    alert = Alert(
                        employee_id=employee['employee_id'],
                        employee_name=f"{employee['first_name']} {employee['last_name']}",
                        department=employee['department'],
                        job_level=employee['job_level'],
                        risk_score=prediction['risk_score'],
                        risk_level=prediction['risk_level'],
                        alert_type="High Risk",
                        timestamp=datetime.now(),
                        recommendations=recommendations
                    )
                    
                    alerts.append(alert)
            
            logger.info(f"{len(alerts)} alertes générées")
            return alerts
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification des alertes: {e}")
            return []
    
    def send_email_alert(self, alert: Alert):
        """Envoie une alerte par email"""
        try:
            # Configuration email depuis la config
            email_config = self.config.get('monitoring', {}).get('email', {})
            
            if not email_config.get('enabled', False):
                return
            
            # Préparer le message
            msg = MIMEMultipart()
            msg['From'] = email_config.get('from_email', 'noreply@company.com')
            msg['To'] = ', '.join(email_config.get('alert_recipients', []))
            msg['Subject'] = f"🚨 ALERTE TURNOVER - {alert.employee_name}"
            
            # Corps du message
            body = f"""
            <h2>🚨 Alerte de Risque de Turnover</h2>
            
            <h3>Employé concerné :</h3>
            <ul>
                <li><strong>Nom :</strong> {alert.employee_name}</li>
                <li><strong>ID :</strong> {alert.employee_id}</li>
                <li><strong>Département :</strong> {alert.department}</li>
                <li><strong>Niveau :</strong> {alert.job_level}</li>
            </ul>
            
            <h3>Détails du risque :</h3>
            <ul>
                <li><strong>Score de risque :</strong> {alert.risk_score:.1%}</li>
                <li><strong>Niveau :</strong> {alert.risk_level}</li>
                <li><strong>Type d'alerte :</strong> {alert.alert_type}</li>
                <li><strong>Timestamp :</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</li>
            </ul>
            
            <h3>Recommandations d'actions :</h3>
            <ul>
                {''.join([f'<li>{rec}</li>' for rec in alert.recommendations])}
            </ul>
            
            <p><em>Cette alerte a été générée automatiquement par le système de monitoring du turnover.</em></p>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            # Envoyer l'email
            server = smtplib.SMTP(email_config.get('smtp_server', 'localhost'), 
                                 email_config.get('smtp_port', 587))
            server.starttls()
            server.login(email_config.get('username', ''), email_config.get('password', ''))
            
            text = msg.as_string()
            server.sendmail(msg['From'], email_config.get('alert_recipients', []), text)
            server.quit()
            
            logger.info(f"Email d'alerte envoyé pour {alert.employee_name}")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'envoi de l'email pour {alert.employee_name}: {e}")
    
    def send_slack_alert(self, alert: Alert):
        """Envoie une alerte sur Slack"""
        try:
            slack_config = self.config.get('monitoring', {}).get('slack', {})
            
            if not slack_config.get('enabled', False):
                return
            
            webhook_url = slack_config.get('webhook_url', '')
            if not webhook_url:
                return
            
            # Message Slack
            message = {
                "text": f"🚨 *ALERTE TURNOVER* - {alert.employee_name}",
                "attachments": [
                    {
                        "color": "danger" if alert.risk_score > 0.8 else "warning",
                        "fields": [
                            {"title": "Employé", "value": f"{alert.employee_name} ({alert.employee_id})", "short": True},
                            {"title": "Département", "value": alert.department, "short": True},
                            {"title": "Niveau", "value": alert.job_level, "short": True},
                            {"title": "Score de risque", "value": f"{alert.risk_score:.1%}", "short": True},
                            {"title": "Niveau", "value": alert.risk_level, "short": True},
                            {"title": "Timestamp", "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), "short": True}
                        ],
                        "footer": "Système de Monitoring Turnover",
                        "ts": int(alert.timestamp.timestamp())
                    }
                ]
            }
            
            # Envoyer sur Slack
            response = requests.post(webhook_url, json=message)
            response.raise_for_status()
            
            logger.info(f"Alerte Slack envoyée pour {alert.employee_name}")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'envoi de l'alerte Slack pour {alert.employee_name}: {e}")
    
    def process_alerts(self):
        """Traite toutes les alertes"""
        try:
            alerts = self.check_alerts()
            
            for alert in alerts:
                # Vérifier si l'alerte n'a pas déjà été envoyée récemment
                recent_alerts = [a for a in self.alert_history 
                               if a.employee_id == alert.employee_id and 
                               (datetime.now() - a.timestamp).seconds < 3600]  # 1 heure
                
                if not recent_alerts:
                    # Envoyer les alertes
                    self.send_email_alert(alert)
                    self.send_slack_alert(alert)
                    
                    # Ajouter à l'historique
                    self.alert_history.append(alert)
                    
                    # Sauvegarder l'alerte
                    self.save_alert(alert)
            
            logger.info(f"Traitement de {len(alerts)} alertes terminé")
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement des alertes: {e}")
    
    def save_alert(self, alert: Alert):
        """Sauvegarde une alerte dans un fichier JSON"""
        try:
            alert_data = {
                'employee_id': alert.employee_id,
                'employee_name': alert.employee_name,
                'department': alert.department,
                'job_level': alert.job_level,
                'risk_score': alert.risk_score,
                'risk_level': alert.risk_level,
                'alert_type': alert.alert_type,
                'timestamp': alert.timestamp.isoformat(),
                'recommendations': alert.recommendations
            }
            
            # Charger les alertes existantes
            try:
                with open(get_data_path('alerts_history.json'), 'r') as f:
                    alerts_history = json.load(f)
            except:
                alerts_history = []
            
            # Ajouter la nouvelle alerte
            alerts_history.append(alert_data)
            
            # Sauvegarder
            with open(get_data_path('alerts_history.json'), 'w') as f:
                json.dump(alerts_history, f, indent=2)
                
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de l'alerte: {e}")
    
    def generate_daily_report(self):
        """Génère un rapport quotidien"""
        try:
            df = self.get_employee_data()
            if df.empty:
                return
            
            # Calculer les statistiques
            total_employees = len(df)
            high_risk_count = 0
            medium_risk_count = 0
            low_risk_count = 0
            
            risk_scores = []
            
            for _, employee in df.iterrows():
                prediction = self.predict_employee_risk(employee)
                risk_scores.append(prediction['risk_score'])
                
                if prediction['risk_score'] > 0.6:
                    high_risk_count += 1
                elif prediction['risk_score'] > 0.4:
                    medium_risk_count += 1
                else:
                    low_risk_count += 1
            
            avg_risk = np.mean(risk_scores)
            
            # Générer le rapport
            report = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'total_employees': total_employees,
                'high_risk_count': high_risk_count,
                'medium_risk_count': medium_risk_count,
                'low_risk_count': low_risk_count,
                'average_risk_score': avg_risk,
                'alerts_today': len([a for a in self.alert_history 
                                   if a.timestamp.date() == datetime.now().date()])
            }
            
            # Sauvegarder le rapport
            with open(get_data_path(f'daily_report_{datetime.now().strftime("%Y%m%d")}.json'), 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Rapport quotidien généré: {report}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du rapport quotidien: {e}")
    
    def start_monitoring(self):
        """Démarre le système de monitoring"""
        logger.info("Démarrage du système de monitoring")
        
        # Planifier les tâches
        schedule.every(30).minutes.do(self.process_alerts)
        schedule.every().day.at("09:00").do(self.generate_daily_report)
        
        self.monitoring_active = True
        
        while self.monitoring_active:
            schedule.run_pending()
            time.sleep(60)  # Vérifier toutes les minutes
    
    def stop_monitoring(self):
        """Arrête le système de monitoring"""
        logger.info("Arrêt du système de monitoring")
        self.monitoring_active = False

def main():
    """Fonction principale"""
    print("\n" + "="*60)
    print("SYSTÈME DE MONITORING ET ALERTES AUTOMATIQUES")
    print("="*60)
    
    # Initialiser le système
    monitoring_system = MonitoringSystem()
    
    if not monitoring_system.model_manager:
        print("[ERREUR] Impossible de charger les modèles")
        return
    
    print("\n[INFO] Système de monitoring initialisé")
    print("[INFO] Tâches programmées:")
    print("  • Vérification des alertes: toutes les 30 minutes")
    print("  • Rapport quotidien: 09:00")
    
    try:
        # Démarrer le monitoring
        monitoring_system.start_monitoring()
    except KeyboardInterrupt:
        print("\n[INFO] Arrêt du système de monitoring")
        monitoring_system.stop_monitoring()

if __name__ == "__main__":
    main()
