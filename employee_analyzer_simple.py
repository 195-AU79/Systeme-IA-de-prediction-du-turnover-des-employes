#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilisation du Modèle de Prédiction du Turnover
Analyse des critères pour des employés spécifiques
"""

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
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
    'salary_percentile': 'Percentile salarial',
    
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
    
    # Encodés
    'department_encoded': 'Département (encodé)',
    'job_level_encoded': 'Niveau (encodé)',
    'location_encoded': 'Localisation (encodé)',
}

def translate_criteria(criteria_name: str) -> str:
    """
    Traduit un nom de critère en français.
    Si le critère n'est pas dans le dictionnaire, retourne le nom original formaté.
    """
    return CRITERIA_TRANSLATION.get(criteria_name, criteria_name.replace('_', ' ').title())

class TurnoverAnalyzer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = []
        self.load_model()
    
    def load_model(self):
        """Charge le modèle entraîné"""
        try:
            self.model = joblib.load('turnover_criteria_model.pkl')
            self.scaler = joblib.load('criteria_scaler.pkl')
            
            # Charger les encodeurs
            import os
            for file in os.listdir('.'):
                if file.startswith('criteria_encoder_') and file.endswith('.pkl'):
                    col_name = file.replace('criteria_encoder_', '').replace('.pkl', '')
                    self.label_encoders[col_name] = joblib.load(file)
            
            # Charger les noms des features
            with open('criteria_analysis_results.json', 'r') as f:
                results = json.load(f)
                self.feature_names = [item['feature'] for item in results['feature_importance']]
            
            print("[OK] Modèle chargé avec succès")
            
        except Exception as e:
            print(f"[ERREUR] Impossible de charger le modèle: {e}")
    
    def analyze_employee(self, employee_data):
        """Analyse un employé spécifique"""
        
        # Préparer les données
        prepared_data = self.prepare_employee_data(employee_data)
        
        # Prédiction
        X = np.array([prepared_data])
        X_scaled = self.scaler.transform(X)
        
        risk_proba = self.model.predict_proba(X_scaled)[0, 1]
        risk_prediction = self.model.predict(X_scaled)[0]
        
        # Analyse des critères
        criteria_analysis = {}
        for i, feature in enumerate(self.feature_names):
            feature_value = prepared_data[i]
            feature_importance = self.model.feature_importances_[i]
            
            criteria_analysis[feature] = {
                'value': float(feature_value),
                'importance': float(feature_importance),
                'contribution': float(feature_value * feature_importance)
            }
        
        # Identifier les facteurs clés
        key_factors = self.identify_key_factors(criteria_analysis)
        
        # Déterminer le niveau de risque
        if risk_proba < 0.2:
            risk_level = "Très Faible"
        elif risk_proba < 0.4:
            risk_level = "Faible"
        elif risk_proba < 0.6:
            risk_level = "Moyen"
        elif risk_proba < 0.8:
            risk_level = "Élevé"
        else:
            risk_level = "Critique"
        
        return {
            'risk_score': float(risk_proba),
            'risk_level': risk_level,
            'will_leave': bool(risk_prediction),
            'confidence': float(max(risk_proba, 1-risk_proba)),
            'key_factors': key_factors,
            'criteria_analysis': criteria_analysis,
            'recommendations': self.generate_recommendations(key_factors, risk_proba)
        }
    
    def prepare_employee_data(self, employee_data):
        """Prépare les données d'un employé pour la prédiction"""
        
        # Créer un DataFrame temporaire pour les calculs
        df_temp = pd.DataFrame([employee_data])
        
        # Calculer les critères enrichis
        df_temp = self.calculate_enriched_criteria(df_temp)
        
        # Préparer les données selon les features du modèle
        prepared_data = []
        for feature in self.feature_names:
            if feature in df_temp.columns:
                prepared_data.append(df_temp[feature].iloc[0])
            else:
                prepared_data.append(0)
        
        return prepared_data
    
    def calculate_enriched_criteria(self, df):
        """Calcule les critères enrichis pour un employé"""
        
        # Critères salariaux (simulés)
        df['salary_vs_avg_dept'] = 1.0  # Valeur par défaut
        df['salary_vs_avg_level'] = 1.0
        df['salary_percentile'] = 0.5
        
        # Critères de performance
        df['performance_trend'] = df.get('performance_rating', 3.5).iloc[0] - 3.0
        # performance_consistency retiré - critère peu clair
        df['goals_achievement_rate'] = df.get('goals_achieved', 80).iloc[0] / 100
        
        # Critères de formation
        df['training_intensity'] = df.get('training_hours', 20).iloc[0] / (df.get('tenure_years', 2).iloc[0] + 0.1)
        df['training_quality'] = df.get('training_score', 85).iloc[0]
        df['training_frequency'] = df.get('training_count', 5).iloc[0] / (df.get('tenure_years', 2).iloc[0] + 0.1)
        
        # Critères de charge de travail
        df['overtime_intensity'] = df.get('overtime_hours', 10).iloc[0] / (df.get('tenure_years', 2).iloc[0] + 0.1)
        df['overtime_frequency'] = df.get('overtime_count', 5).iloc[0] / (df.get('tenure_years', 2).iloc[0] + 0.1)
        df['workload_indicator'] = df['overtime_intensity'] * df.get('projects_count', 3).iloc[0]
        
        # Critères d'absences
        df['absence_rate'] = df.get('absence_days', 5).iloc[0] / (df.get('tenure_years', 2).iloc[0] * 365 + 1)
        df['absence_frequency'] = df.get('absence_count', 3).iloc[0] / (df.get('tenure_years', 2).iloc[0] + 0.1)
        
        # Critères de satisfaction
        df['satisfaction_score'] = (
            df.get('performance_rating', 3.5).iloc[0] * 0.3 +
            df['goals_achievement_rate'] * 0.2 +
            df['salary_percentile'] * 0.2 +
            (1 - df['overtime_intensity'] / 10) * 0.15 +
            df['training_quality'] / 100 * 0.15
        )
        
        # Critères de relation manager
        df['manager_relationship'] = (
            df.get('feedback_score', 3.5).iloc[0] * 0.5 +
            (1 - df['absence_rate']) * 0.5
        )
        
        # Critères de marché
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
        
        dept = df['department'].iloc[0] if 'department' in df.columns else 'IT'
        level = df['job_level'].iloc[0] if 'job_level' in df.columns else 'Mid'
        df['market_salary'] = market_salaries.get(dept, {}).get(level, 50000)
        df['salary_vs_market'] = df.get('salary', 50000).iloc[0] / df['market_salary']
        
        # Critères de carrière
        level_values = {'Junior': 1, 'Mid': 2, 'Senior': 3, 'Lead': 4, 'Manager': 5, 'Director': 6}
        df['career_progression'] = df.get('tenure_years', 2).iloc[0] / (level_values.get(level, 2) + 0.1)
        df['promotion_likelihood'] = df.get('performance_rating', 3.5).iloc[0] * df['goals_achievement_rate'] * df['training_intensity']
        
        # Critères de stress et équilibre
        df['stress_indicator'] = df['overtime_intensity'] + df['absence_rate']
        df['work_life_balance'] = 1 / (1 + df['stress_indicator'])
        
        return df
    
    def identify_key_factors(self, criteria_analysis):
        """Identifie les facteurs clés pour un employé"""
        sorted_criteria = sorted(criteria_analysis.items(), 
                               key=lambda x: abs(x[1]['contribution']), 
                               reverse=True)
        
        key_factors = []
        for feature, data in sorted_criteria[:5]:
            if data['contribution'] > 0:
                impact = "Augmente le risque"
            else:
                impact = "Réduit le risque"
            
            key_factors.append({
                'factor': feature,
                'value': data['value'],
                'impact': impact,
                'contribution': data['contribution']
            })
        
        return key_factors
    
    def generate_recommendations(self, key_factors, risk_score):
        """Génère des recommandations basées sur l'analyse"""
        recommendations = []
        
        if risk_score > 0.8:
            recommendations.append("ALERTE CRITIQUE - Action immédiate requise")
            recommendations.append("Entretien urgent avec le manager")
            recommendations.append("Évaluation immédiate de l'augmentation")
            recommendations.append("Plan de rétention personnalisé")
        elif risk_score > 0.6:
            recommendations.append("Risque élevé - Surveillance renforcée")
            recommendations.append("Entretien dans les 48h")
            recommendations.append("Discussion sur l'évolution de carrière")
            recommendations.append("Programme de reconnaissance")
        elif risk_score > 0.4:
            recommendations.append("Risque modéré - Actions préventives")
            recommendations.append("Entretien de suivi mensuel")
            recommendations.append("Plan de développement des compétences")
            recommendations.append("Amélioration de l'environnement de travail")
        else:
            recommendations.append("Risque faible - Maintenir la satisfaction")
            recommendations.append("Continuer le développement professionnel")
            recommendations.append("Renforcer l'engagement d'équipe")
            recommendations.append("Monitoring régulier")
        
        # Recommandations spécifiques basées sur les facteurs clés
        for factor in key_factors:
            factor_name = factor['factor']
            
            if 'salary' in factor_name.lower():
                if factor['contribution'] > 0:
                    recommendations.append(f"Améliorer la compensation ({factor_name})")
            
            if 'performance' in factor_name.lower():
                if factor['contribution'] > 0:
                    recommendations.append(f"Soutenir la performance ({factor_name})")
            
            if 'training' in factor_name.lower():
                if factor['contribution'] < 0:
                    recommendations.append(f"Augmenter la formation ({factor_name})")
            
            if 'overtime' in factor_name.lower():
                if factor['contribution'] > 0:
                    recommendations.append(f"Réduire la charge de travail ({factor_name})")
            
            if 'manager' in factor_name.lower():
                if factor['contribution'] > 0:
                    recommendations.append(f"Améliorer la relation manager ({factor_name})")
        
        return recommendations

def main():
    """Fonction principale avec exemples d'utilisation"""
    
    print("\n" + "="*60)
    print("ANALYSEUR DE PREDICTION DU TURNOVER")
    print("="*60)
    
    # Initialiser l'analyseur
    analyzer = TurnoverAnalyzer()
    
    if analyzer.model is None:
        print("[ERREUR] Impossible de charger le modèle")
        print("Exécutez d'abord: python criteria_analysis_model.py")
        return
    
    # Exemples d'employés à analyser
    employees_to_analyze = [
        {
            'name': 'Marie Dubois',
            'age': 28,
            'tenure_years': 2.5,
            'salary': 45000,
            'department': 'IT',
            'job_level': 'Mid',
            'performance_rating': 4.2,
            'goals_achieved': 85,
            'training_hours': 25,
            'training_score': 88,
            'training_count': 6,
            'overtime_hours': 8,
            'overtime_count': 3,
            'projects_count': 3,
            'absence_days': 3,
            'absence_count': 2,
            'feedback_score': 4.0
        },
        {
            'name': 'Jean Martin',
            'age': 45,
            'tenure_years': 8.0,
            'salary': 65000,
            'department': 'Sales',
            'job_level': 'Senior',
            'performance_rating': 3.8,
            'goals_achieved': 70,
            'training_hours': 5,
            'training_score': 75,
            'training_count': 2,
            'overtime_hours': 25,
            'overtime_count': 8,
            'projects_count': 7,
            'absence_days': 8,
            'absence_count': 4,
            'feedback_score': 3.2
        },
        {
            'name': 'Sophie Leroy',
            'age': 35,
            'tenure_years': 5.5,
            'salary': 75000,
            'department': 'HR',
            'job_level': 'Lead',
            'performance_rating': 4.8,
            'goals_achieved': 95,
            'training_hours': 40,
            'training_score': 92,
            'training_count': 8,
            'overtime_hours': 5,
            'overtime_count': 2,
            'projects_count': 4,
            'absence_days': 2,
            'absence_count': 1,
            'feedback_score': 4.7
        }
    ]
    
    print(f"\nAnalyse de {len(employees_to_analyze)} employés:")
    print("-" * 60)
    
    results = []
    
    for i, emp in enumerate(employees_to_analyze, 1):
        print(f"\n[{i}] {emp['name']} ({emp['department']} - {emp['job_level']})")
        
        # Analyse
        analysis = analyzer.analyze_employee(emp)
        
        print(f"  Score de risque: {analysis['risk_score']:.1%}")
        print(f"  Niveau de risque: {analysis['risk_level']}")
        print(f"  Prédiction: {'Partira' if analysis['will_leave'] else 'Restera'}")
        print(f"  Confiance: {analysis['confidence']:.1%}")
        
        print(f"\n  Facteurs clés:")
        for j, factor in enumerate(analysis['key_factors'], 1):
            factor_name = factor['factor']
            factor_fr = translate_criteria(factor_name)
            print(f"    {j}. {factor_fr:<35} : {factor['impact']}")
        
        print(f"\n  Recommandations:")
        for j, rec in enumerate(analysis['recommendations'][:4], 1):
            print(f"    {j}. {rec}")
        
        results.append({
            'employee': emp['name'],
            'analysis': analysis
        })
    
    # Sauvegarder les résultats
    with open('employee_analysis_results.json', 'w') as f:
        json.dump({
            'analyses': results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n[OK] Résultats sauvegardés dans 'employee_analysis_results.json'")
    
    # Statistiques globales
    high_risk = sum(1 for r in results if r['analysis']['risk_score'] > 0.6)
    medium_risk = sum(1 for r in results if 0.4 < r['analysis']['risk_score'] <= 0.6)
    low_risk = sum(1 for r in results if r['analysis']['risk_score'] <= 0.4)
    
    print(f"\n" + "="*60)
    print("RESUME DES ANALYSES")
    print("="*60)
    print(f"[INFO] Employés à risque élevé: {high_risk}")
    print(f"[INFO] Employés à risque modéré: {medium_risk}")
    print(f"[INFO] Employés à faible risque: {low_risk}")
    
    avg_risk = sum(r['analysis']['risk_score'] for r in results) / len(results)
    print(f"[INFO] Risque moyen: {avg_risk:.1%}")
    
    print("\n" + "="*60)
    print("ANALYSE TERMINEE")
    print("="*60)

if __name__ == "__main__":
    main()
