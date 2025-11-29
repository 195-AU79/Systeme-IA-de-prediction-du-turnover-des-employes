#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dashboard Analytique pour le Système de Prédiction du Turnover
Interface web interactive avec Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import joblib
import json
import yaml
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# DICTIONNAIRE DE TRADUCTION DES CRITÈRES
# ==============================================================================

CRITERIA_TRANSLATION = {
    # Données de base
    'age': 'Âge',
    'tenure_years': 'Ancienneté (années)',
    'salary': 'Salaire',
    'department': 'Département',
    'job_level': 'Niveau hiérarchique',
    'location': 'Localisation',
    
    # Performance
    'performance_rating': 'Note de performance',
    'performance_rating_mean': 'Note de performance moyenne',
    'performance_trend': 'Tendance de performance',
    'goals_achieved': 'Objectifs atteints',
    'goals_achievement_rate': 'Taux d\'atteinte des objectifs',
    
    # Formation
    'training_hours': 'Heures de formation',
    'training_intensity': 'Intensité de formation',
    'training_quality': 'Qualité de la formation',
    'training_frequency': 'Fréquence de formation',
    
    # Charge de travail
    'overtime_hours': 'Heures supplémentaires',
    'overtime_intensity': 'Intensité d\'heures supplémentaires',
    'overtime_frequency': 'Fréquence d\'heures supplémentaires',
    'workload_indicator': 'Indicateur de charge de travail',
    'projects_count': 'Nombre de projets',
    
    # Absences
    'absence_rate': 'Taux d\'absences',
    'absence_frequency': 'Fréquence d\'absences',
    
    # Satisfaction
    'satisfaction_score': 'Score de satisfaction',
    'work_life_balance': 'Équilibre vie/travail',
    'stress_indicator': 'Indicateur de stress',
    
    # Salaire et compensation
    'salary_vs_market': 'Salaire vs marché',
    'salary_vs_avg_dept': 'Salaire vs moyenne département',
    'salary_vs_avg_level': 'Salaire vs moyenne niveau',
    'salary_gap_vs_market': 'Écart salarial vs marché',
    'salary_competitiveness': 'Compétitivité salariale',
    'salary_satisfaction': 'Satisfaction salariale',
    'salary_vs_competitors': 'Salaire vs concurrents',
    
    # Carrière
    'career_progression': 'Progression de carrière',
    'career_stagnation': 'Stagnation de carrière',
    'promotion_likelihood': 'Probabilité de promotion',
    'months_since_last_raise': 'Mois depuis la dernière augmentation',
    'raise_overdue': 'Augmentation en retard',
    'raise_urgency': 'Urgence d\'augmentation',
    'last_promotion_months': 'Mois depuis dernière promotion',
    
    # Manager
    'manager_relationship': 'Relation manager',
    'manager_relationship_quality': 'Qualité relation manager',
    'manager_change_count': 'Nombre de changements de manager',
    'feedback_score': 'Score de feedback',
    
    # Marché et concurrence
    'competitor_attractiveness': 'Attractivité concurrente',
    'external_attractiveness': 'Attractivité externe',
    'company_image_score': 'Score d\'image société',
    
    # Ancienneté
    'tenure_risk_factor': 'Facteur de risque d\'ancienneté',
    
    # Autres
    'left_company': 'A quitté l\'entreprise',
    'hire_date': 'Date d\'embauche',
    'gender': 'Genre',
    'first_name': 'Prénom',
    'last_name': 'Nom',
    'email': 'Email',
}

def translate_criteria(criteria_name: str) -> str:
    """
    Traduit un nom de critère en français.
    Si le critère n'est pas dans le dictionnaire, retourne le nom original en titre.
    """
    return CRITERIA_TRANSLATION.get(criteria_name, criteria_name.replace('_', ' ').title())

# Configuration de la page
st.set_page_config(
    page_title="Dashboard Turnover Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    .alert-low {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

class DashboardData:
    def __init__(self):
        self.data = None
        self.predictions = None
        self.alerts = None
        self.load_data()
    
    def load_data(self):
        """Charge toutes les données nécessaires"""
        try:
            # Vérifier que la base de données existe
            import os
            if not os.path.exists('turnover_data.db'):
                self.data = None
                return
            
            # Charger les données de base
            conn = sqlite3.connect('turnover_data.db')
            
            # Vérifier que les tables existent
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            existing_tables = [row[0] for row in cursor.fetchall()]
            required_tables = ['employees', 'turnover', 'performance', 'training', 'overtime', 'absences']
            missing_tables = [t for t in required_tables if t not in existing_tables]
            
            if missing_tables:
                conn.close()
                self.data = None
                return
            
            self.employees_df = pd.read_sql_query("SELECT * FROM employees", conn)
            self.turnover_df = pd.read_sql_query("SELECT * FROM turnover", conn)
            self.performance_df = pd.read_sql_query("SELECT * FROM performance", conn)
            self.training_df = pd.read_sql_query("SELECT * FROM training", conn)
            self.overtime_df = pd.read_sql_query("SELECT * FROM overtime", conn)
            self.absences_df = pd.read_sql_query("SELECT * FROM absences", conn)
            conn.close()
            
            # Vérifier que les DataFrames ne sont pas vides
            if self.employees_df.empty:
                self.data = None
                return
            
            # Préparer les données
            self.prepare_data()
            
            # Charger les prédictions si disponibles
            self.load_predictions()
            
            # Charger les alertes si disponibles
            self.load_alerts()
            
        except sqlite3.Error as e:
            # Erreur spécifique à SQLite
            self.data = None
            if 'st' in globals():
                st.error(f"❌ Erreur de base de données SQLite: {e}")
        except Exception as e:
            # Autre erreur
            self.data = None
            if 'st' in globals():
                st.error(f"❌ Erreur lors du chargement des données: {e}")
    
    def prepare_data(self):
        """Prépare les données pour l'analyse"""
        # Créer la variable cible
        self.employees_df['left_company'] = self.employees_df['employee_id'].isin(self.turnover_df['employee_id']).astype(int)
        
        # Calculer l'ancienneté
        self.employees_df['hire_date'] = pd.to_datetime(self.employees_df['hire_date'])
        self.employees_df['tenure_years'] = (datetime.now() - self.employees_df['hire_date']).dt.days / 365.25
        
        # Agréger les données de performance
        perf_agg = self.performance_df.groupby('employee_id').agg({
            'performance_rating': 'mean',
            'goals_achieved': 'mean',
            'feedback_score': 'mean'
        }).reset_index()
        
        # Agréger les données de formation
        training_agg = self.training_df.groupby('employee_id').agg({
            'hours_completed': 'sum',
            'score': 'mean'
        }).reset_index()
        
        # Agréger les heures supplémentaires
        overtime_agg = self.overtime_df.groupby('employee_id').agg({
            'hours': 'sum'
        }).reset_index()
        
        # Agréger les absences
        absences_agg = self.absences_df.groupby('employee_id').agg({
            'duration_days': 'count'
        }).reset_index()
        
        # Fusionner toutes les données
        self.data = self.employees_df.copy()
        self.data = self.data.merge(perf_agg, on='employee_id', how='left')
        self.data = self.data.merge(training_agg, on='employee_id', how='left')
        self.data = self.data.merge(overtime_agg, on='employee_id', how='left')
        self.data = self.data.merge(absences_agg, on='employee_id', how='left')
        
        # Remplacer les NaN et renommer
        self.data = self.data.fillna(0)
        
        # Créer les colonnes avec des valeurs par défaut si elles n'existent pas
        if 'hours_completed' in self.data.columns:
            self.data['training_hours'] = self.data['hours_completed']
        else:
            self.data['training_hours'] = 0
            
        if 'hours' in self.data.columns:
            self.data['overtime_hours'] = self.data['hours']
        else:
            self.data['overtime_hours'] = 0
            
        if 'duration_days' in self.data.columns:
            self.data['projects_count'] = self.data['duration_days']
        else:
            self.data['projects_count'] = 0
            
        if 'feedback_score' in self.data.columns:
            self.data['satisfaction_score'] = self.data['feedback_score'].fillna(3.5)
        else:
            self.data['satisfaction_score'] = 3.5
            
        self.data['work_life_balance'] = 3.5
        self.data['last_promotion_months'] = self.data['tenure_years'] * 12 * 0.5
        self.data['manager_change_count'] = 0
        
        # S'assurer que performance_rating existe
        if 'performance_rating' not in self.data.columns:
            self.data['performance_rating'] = 3.5
    
    def load_predictions(self):
        """Charge les prédictions depuis les fichiers JSON"""
        try:
            with open('ensemble_prediction_results.json', 'r') as f:
                self.predictions = json.load(f)
        except FileNotFoundError:
            # Fichier non trouvé, créer une structure vide
            self.predictions = None
        except json.JSONDecodeError:
            # Fichier corrompu, ignorer
            self.predictions = None
        except Exception as e:
            # Autre erreur, ignorer silencieusement
            self.predictions = None
    
    def load_alerts(self):
        """Charge les alertes depuis le fichier JSON"""
        try:
            with open('alerts_history.json', 'r') as f:
                self.alerts = json.load(f)
        except FileNotFoundError:
            # Fichier non trouvé, liste vide par défaut
            self.alerts = []
        except json.JSONDecodeError:
            # Fichier corrompu, liste vide
            self.alerts = []
        except Exception as e:
            # Autre erreur, liste vide
            self.alerts = []

def main():
    """Fonction principale du dashboard"""
    
    # En-tête
    st.markdown('<h1 class="main-header">📊 Dashboard de Prédiction du Turnover</h1>', unsafe_allow_html=True)
    
    # Charger les données
    try:
        data_manager = DashboardData()
        
        if data_manager.data is None:
            st.error("❌ Impossible de charger les données depuis la base de données")
            st.info("💡 Assurez-vous que :")
            st.write("1. La base de données `turnover_data.db` existe")
            st.write("2. Les tables `employees`, `turnover`, `performance`, `training`, `overtime`, `absences` existent")
            st.write("3. Les tables contiennent des données")
            st.write("\n💡 Pour créer la base de données, exécutez : `python create_database.py`")
            return
    except Exception as e:
        st.error(f"❌ Erreur lors de l'initialisation du dashboard : {str(e)}")
        st.exception(e)
        return
    
    # Sidebar pour les filtres
    st.sidebar.header("🔍 Filtres")
    
    # Filtres
    departments = ['Tous'] + list(data_manager.data['department'].unique())
    selected_dept = st.sidebar.selectbox("Département", departments)
    
    job_levels = ['Tous'] + list(data_manager.data['job_level'].unique())
    selected_level = st.sidebar.selectbox("Niveau", job_levels)
    
    # Appliquer les filtres
    filtered_data = data_manager.data.copy()
    if selected_dept != 'Tous':
        filtered_data = filtered_data[filtered_data['department'] == selected_dept]
    if selected_level != 'Tous':
        filtered_data = filtered_data[filtered_data['job_level'] == selected_level]
    
    # Métriques principales
    st.header("📈 Métriques Principales")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_employees = len(filtered_data)
        st.metric("Total Employés", total_employees)
    
    with col2:
        departed_employees = len(filtered_data[filtered_data['left_company'] == 1])
        turnover_rate = departed_employees / total_employees if total_employees > 0 else 0
        st.metric("Taux de Turnover", f"{turnover_rate:.1%}")
    
    with col3:
        avg_tenure = filtered_data['tenure_years'].mean()
        st.metric("Ancienneté Moyenne", f"{avg_tenure:.1f} ans")
    
    with col4:
        avg_salary = filtered_data['salary'].mean()
        st.metric("Salaire Moyen", f"{avg_salary:,.0f}€")
    
    # Graphiques principaux
    st.header("📊 Analyses Visuelles")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution par département
        dept_counts = filtered_data['department'].value_counts()
        fig_dept = px.pie(
            values=dept_counts.values,
            names=dept_counts.index,
            title="Distribution par Département"
        )
        st.plotly_chart(fig_dept, width='stretch')
    
    with col2:
        # Distribution par niveau
        level_counts = filtered_data['job_level'].value_counts()
        fig_level = px.bar(
            x=level_counts.index,
            y=level_counts.values,
            title="Distribution par Niveau",
            labels={'x': translate_criteria('job_level'), 'y': 'Nombre d\'employés'}
        )
        st.plotly_chart(fig_level, width='stretch')
    
    # Analyse du turnover
    st.header("🔄 Analyse du Turnover")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Turnover par département
        turnover_by_dept = filtered_data.groupby('department').agg({
            'left_company': ['count', 'sum', 'mean']
        }).round(3)
        turnover_by_dept.columns = ['Total', 'Départs', 'Taux']
        
        fig_turnover_dept = px.bar(
            x=turnover_by_dept.index,
            y=turnover_by_dept['Taux'],
            title="Taux de Turnover par Département",
            labels={'x': translate_criteria('department'), 'y': 'Taux de Turnover'}
        )
        st.plotly_chart(fig_turnover_dept, width='stretch')
    
    with col2:
        # Turnover par ancienneté
        filtered_data['tenure_group'] = pd.cut(
            filtered_data['tenure_years'],
            bins=[0, 1, 3, 5, 10, 100],
            labels=['<1an', '1-3ans', '3-5ans', '5-10ans', '>10ans']
        )
        
        turnover_by_tenure = filtered_data.groupby('tenure_group')['left_company'].mean()
        
        fig_turnover_tenure = px.bar(
            x=turnover_by_tenure.index,
            y=turnover_by_tenure.values,
            title="Taux de Turnover par Ancienneté",
            labels={'x': translate_criteria('tenure_years'), 'y': 'Taux de Turnover'}
        )
        st.plotly_chart(fig_turnover_tenure, width='stretch')
    
    # Analyse des performances
    st.header("🎯 Analyse des Performances")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance vs Turnover
        fig_perf = px.scatter(
            filtered_data,
            x='performance_rating',
            y='left_company',
            color='department',
            title="Performance vs Risque de Départ",
            labels={'performance_rating': translate_criteria('performance_rating'), 'left_company': translate_criteria('left_company')}
        )
        st.plotly_chart(fig_perf, width='stretch')
    
    with col2:
        # Salaire vs Turnover
        fig_salary = px.scatter(
            filtered_data,
            x='salary',
            y='left_company',
            color='job_level',
            title="Salaire vs Risque de Départ",
            labels={'salary': translate_criteria('salary'), 'left_company': translate_criteria('left_company')}
        )
        st.plotly_chart(fig_salary, width='stretch')
    
    # Prédictions et Alertes
    if data_manager.predictions:
        st.header("🔮 Prédictions et Alertes")
        
        # Afficher les prédictions récentes
        if 'sample_prediction' in data_manager.predictions:
            pred = data_manager.predictions['sample_prediction']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                risk_score = pred['risk_score']
                risk_level = "Critique" if risk_score > 0.8 else "Élevé" if risk_score > 0.6 else "Moyen" if risk_score > 0.4 else "Faible"
                
                if risk_score > 0.6:
                    st.markdown(f'<div class="metric-card alert-high">', unsafe_allow_html=True)
                elif risk_score > 0.4:
                    st.markdown(f'<div class="metric-card alert-medium">', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="metric-card alert-low">', unsafe_allow_html=True)
                
                st.metric("Score de Risque", f"{risk_score:.1%}")
                st.write(f"Niveau: {risk_level}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.metric("Confiance", f"{pred['confidence']:.1%}")
            
            with col3:
                prediction_text = "Partira" if pred['prediction'] else "Restera"
                st.metric("Prédiction", prediction_text)
    
    # Alertes récentes
    if data_manager.alerts:
        st.header("🚨 Alertes Récentes")
        
        # Convertir en DataFrame pour l'affichage
        alerts_df = pd.DataFrame(data_manager.alerts)
        
        if not alerts_df.empty:
            # Filtrer les alertes des 7 derniers jours
            alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
            recent_alerts = alerts_df[alerts_df['timestamp'] >= datetime.now() - timedelta(days=7)]
            
            if not recent_alerts.empty:
                st.dataframe(
                    recent_alerts[['employee_name', 'department', 'risk_score', 'risk_level', 'timestamp']],
                    width='stretch'
                )
            else:
                st.info("Aucune alerte récente")
        else:
            st.info("Aucune alerte enregistrée")
    
    # Tableau détaillé des employés
    st.header("👥 Détail des Employés")
    
    # Options d'affichage
    available_cols = ['employee_id', 'first_name', 'last_name', 'department', 'job_level', 
                      'age', 'tenure_years', 'salary', 'performance_rating', 'left_company']
    # Filtrer pour ne garder que les colonnes qui existent
    available_cols = [col for col in available_cols if col in filtered_data.columns]
    
    # Créer des labels traduits pour le multiselect
    col_options = {col: translate_criteria(col) for col in available_cols}
    
    selected_labels = st.multiselect(
        "Colonnes à afficher",
        options=list(col_options.values()),
        default=[col_options.get('first_name', 'first_name'), 
                col_options.get('last_name', 'last_name'),
                col_options.get('department', 'department'),
                col_options.get('job_level', 'job_level'),
                col_options.get('salary', 'salary')]
    )
    
    # Convertir les labels sélectionnés en noms de colonnes
    reverse_translation = {v: k for k, v in col_options.items()}
    display_cols = [reverse_translation[label] for label in selected_labels if label in reverse_translation]
    
    if display_cols:
        # Renommer les colonnes pour l'affichage
        display_df = filtered_data[display_cols].copy()
        display_df.columns = [translate_criteria(col) for col in display_df.columns]
        
        st.dataframe(
            display_df,
            width='stretch',
            height=400
        )
    
    # Export des données
    st.header("📤 Export des Données")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Exporter CSV"):
            csv = filtered_data.to_csv(index=False)
            st.download_button(
                label="Télécharger CSV",
                data=csv,
                file_name=f"turnover_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("Exporter Excel"):
            # Créer un fichier Excel avec plusieurs feuilles
            from io import BytesIO
            output = BytesIO()
            
            try:
                import openpyxl
            except ImportError:
                st.error("⚠️ openpyxl n'est pas installé. Installez-le avec: pip install openpyxl")
                return
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                filtered_data.to_excel(writer, sheet_name='Employés', index=False)
                if not filtered_data.empty:
                    turnover_summary = filtered_data.groupby('department').agg({
                        'left_company': ['count', 'sum', 'mean']
                    }).round(3)
                    turnover_summary.to_excel(writer, sheet_name='Résumé Turnover')
            
            st.download_button(
                label="Télécharger Excel",
                data=output.getvalue(),
                file_name=f"turnover_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    # Pied de page
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: #666;'>"
        f"Dashboard généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')} | "
        f"Système de Prédiction du Turnover v1.0"
        f"</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
