#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lecteur de base de données SQLite pour le système de prédiction du turnover
Permet de lire et analyser les données de la base turnover_data.db
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

class TurnoverDatabase:
    """Classe pour interagir avec la base de données de turnover"""
    
    def __init__(self, db_path='turnover_data.db'):
        self.db_path = db_path
        self.conn = None
    
    def connect(self):
        """Se connecte à la base de données"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            print(f"OK Connexion a la base de donnees: {self.db_path}")
            return True
        except Exception as e:
            print(f"ERREUR: Impossible de se connecter a la base: {e}")
            return False
    
    def disconnect(self):
        """Ferme la connexion à la base de données"""
        if self.conn:
            self.conn.close()
            print("OK Connexion fermee")
    
    def get_employees(self, limit=None):
        """Récupère les données des employés"""
        query = "SELECT * FROM employees"
        if limit:
            query += f" LIMIT {limit}"
        
        df = pd.read_sql_query(query, self.conn)
        return df
    
    def get_performance_data(self, employee_id=None):
        """Récupère les données de performance"""
        query = "SELECT * FROM performance"
        if employee_id:
            query += f" WHERE employee_id = '{employee_id}'"
        
        df = pd.read_sql_query(query, self.conn)
        return df
    
    def get_training_data(self, employee_id=None):
        """Récupère les données de formation"""
        query = "SELECT * FROM training"
        if employee_id:
            query += f" WHERE employee_id = '{employee_id}'"
        
        df = pd.read_sql_query(query, self.conn)
        return df
    
    def get_turnover_data(self):
        """Récupère les données de turnover"""
        query = "SELECT * FROM turnover"
        df = pd.read_sql_query(query, self.conn)
        return df
    
    def get_overtime_data(self, employee_id=None):
        """Récupère les données d'heures supplémentaires"""
        query = "SELECT * FROM overtime"
        if employee_id:
            query += f" WHERE employee_id = '{employee_id}'"
        
        df = pd.read_sql_query(query, self.conn)
        return df
    
    def get_absences_data(self, employee_id=None):
        """Récupère les données d'absences"""
        query = "SELECT * FROM absences"
        if employee_id:
            query += f" WHERE employee_id = '{employee_id}'"
        
        df = pd.read_sql_query(query, self.conn)
        return df
    
    def get_employee_profile(self, employee_id):
        """Récupère le profil complet d'un employé"""
        profile = {}
        
        # Données de base
        profile['basic'] = self.get_employees().query(f"employee_id == '{employee_id}'")
        
        # Performance
        profile['performance'] = self.get_performance_data(employee_id)
        
        # Formation
        profile['training'] = self.get_training_data(employee_id)
        
        # Heures supplémentaires
        profile['overtime'] = self.get_overtime_data(employee_id)
        
        # Absences
        profile['absences'] = self.get_absences_data(employee_id)
        
        return profile
    
    def get_dataset_for_ml(self):
        """Prépare le dataset pour le machine learning"""
        print("Preparation du dataset pour le ML...")
        
        # Récupérer les données de base
        employees_df = self.get_employees()
        turnover_df = self.get_turnover_data()
        
        # Créer la variable cible (left_company)
        employees_df['left_company'] = employees_df['employee_id'].isin(turnover_df['employee_id']).astype(int)
        
        # Calculer l'ancienneté
        employees_df['hire_date'] = pd.to_datetime(employees_df['hire_date'])
        employees_df['tenure_years'] = (datetime.now() - employees_df['hire_date']).dt.days / 365.25
        
        # Agréger les données de performance
        perf_df = self.get_performance_data()
        perf_agg = perf_df.groupby('employee_id').agg({
            'performance_rating': ['mean', 'std', 'max'],
            'goals_achieved': ['mean', 'max'],
            'feedback_score': ['mean', 'std']
        }).round(2)
        
        # Flatten les colonnes multi-niveau
        perf_agg.columns = ['_'.join(col).strip() for col in perf_agg.columns]
        perf_agg = perf_agg.reset_index()
        
        # Agréger les données de formation
        training_df = self.get_training_data()
        training_agg = training_df.groupby('employee_id').agg({
            'hours_completed': ['sum', 'mean', 'count'],
            'score': ['mean', 'std']
        }).round(2)
        
        training_agg.columns = ['_'.join(col).strip() for col in training_agg.columns]
        training_agg = training_agg.reset_index()
        
        # Agréger les heures supplémentaires
        overtime_df = self.get_overtime_data()
        overtime_agg = overtime_df.groupby('employee_id').agg({
            'hours': ['sum', 'mean', 'count']
        }).round(2)
        
        overtime_agg.columns = ['_'.join(col).strip() for col in overtime_agg.columns]
        overtime_agg = overtime_agg.reset_index()
        
        # Agréger les absences
        absences_df = self.get_absences_data()
        absences_agg = absences_df.groupby('employee_id').agg({
            'duration_days': ['sum', 'mean', 'count']
        }).round(2)
        
        absences_agg.columns = ['_'.join(col).strip() for col in absences_agg.columns]
        absences_agg = absences_agg.reset_index()
        
        # Fusionner toutes les données
        ml_dataset = employees_df.copy()
        
        # Joindre les données agrégées
        ml_dataset = ml_dataset.merge(perf_agg, on='employee_id', how='left')
        ml_dataset = ml_dataset.merge(training_agg, on='employee_id', how='left')
        ml_dataset = ml_dataset.merge(overtime_agg, on='employee_id', how='left')
        ml_dataset = ml_dataset.merge(absences_agg, on='employee_id', how='left')
        
        # Remplacer les NaN par 0
        ml_dataset = ml_dataset.fillna(0)
        
        print(f"OK Dataset ML prepare: {len(ml_dataset)} employes, {len(ml_dataset.columns)} colonnes")
        
        return ml_dataset
    
    def get_database_stats(self):
        """Affiche les statistiques de la base de données"""
        print("\n" + "=" * 60)
        print("STATISTIQUES DE LA BASE DE DONNEES")
        print("=" * 60)
        
        # Compter les enregistrements par table
        tables = ['employees', 'performance', 'training', 'turnover', 'overtime', 'absences']
        
        for table in tables:
            try:
                cursor = self.conn.cursor()
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"{table.capitalize()}: {count:,} enregistrements")
            except:
                print(f"{table.capitalize()}: Table non trouvee")
        
        # Statistiques sur les employés
        employees_df = self.get_employees()
        turnover_df = self.get_turnover_data()
        
        print(f"\nStatistiques employes:")
        print(f"  Total employes: {len(employees_df):,}")
        print(f"  Employes partis: {len(turnover_df):,}")
        print(f"  Taux de turnover: {len(turnover_df)/len(employees_df):.1%}")
        
        print(f"\nPar departement:")
        dept_stats = employees_df['department'].value_counts()
        for dept, count in dept_stats.items():
            turnover_count = len(turnover_df[turnover_df['employee_id'].isin(
                employees_df[employees_df['department'] == dept]['employee_id']
            )])
            turnover_rate = turnover_count / count if count > 0 else 0
            print(f"  {dept}: {count} employes, {turnover_count} departs ({turnover_rate:.1%})")
        
        print(f"\nPar niveau:")
        level_stats = employees_df['job_level'].value_counts()
        for level, count in level_stats.items():
            turnover_count = len(turnover_df[turnover_df['employee_id'].isin(
                employees_df[employees_df['job_level'] == level]['employee_id']
            )])
            turnover_rate = turnover_count / count if count > 0 else 0
            print(f"  {level}: {count} employes, {turnover_count} departs ({turnover_rate:.1%})")

def demo_database():
    """Démonstration de l'utilisation de la base de données"""
    print("=" * 60)
    print("DEMONSTRATION DE LA BASE DE DONNEES")
    print("=" * 60)
    
    # Initialiser la base de données
    db = TurnoverDatabase()
    
    if not db.connect():
        print("ERREUR: Impossible de se connecter a la base de donnees")
        print("Executez d'abord: python create_database.py")
        return False
    
    try:
        # Afficher les statistiques
        db.get_database_stats()
        
        # Préparer le dataset pour ML
        ml_dataset = db.get_dataset_for_ml()
        
        print(f"\nColonnes disponibles pour le ML:")
        for i, col in enumerate(ml_dataset.columns, 1):
            print(f"  {i:2d}. {col}")
        
        # Afficher quelques exemples
        print(f"\nExemples d'employes:")
        sample_employees = ml_dataset.sample(3)
        for _, emp in sample_employees.iterrows():
            status = "PARTI" if emp['left_company'] else "ACTIF"
            print(f"  {emp['employee_id']}: {emp['first_name']} {emp['last_name']} - {emp['department']} - {status}")
        
        # Analyser les facteurs de turnover
        print(f"\nAnalyse des facteurs de turnover:")
        
        # Par ancienneté
        ml_dataset['tenure_group'] = pd.cut(ml_dataset['tenure_years'], 
                                          bins=[0, 1, 3, 5, 10, 100], 
                                          labels=['<1an', '1-3ans', '3-5ans', '5-10ans', '>10ans'])
        
        tenure_turnover = ml_dataset.groupby('tenure_group')['left_company'].agg(['count', 'sum', 'mean'])
        print(f"\nPar anciennete:")
        for tenure, row in tenure_turnover.iterrows():
            print(f"  {tenure}: {row['count']} employes, {row['sum']} departs ({row['mean']:.1%})")
        
        # Par département
        dept_turnover = ml_dataset.groupby('department')['left_company'].agg(['count', 'sum', 'mean'])
        print(f"\nPar departement:")
        for dept, row in dept_turnover.iterrows():
            print(f"  {dept}: {row['count']} employes, {row['sum']} departs ({row['mean']:.1%})")
        
        return True
        
    except Exception as e:
        print(f"ERREUR: {e}")
        return False
    
    finally:
        db.disconnect()

if __name__ == "__main__":
    success = demo_database()
    if success:
        print("\nPour utiliser la base de donnees dans vos scripts:")
        print("  from database_reader import TurnoverDatabase")
        print("  db = TurnoverDatabase()")
        print("  db.connect()")
        print("  data = db.get_dataset_for_ml()")
        print("  db.disconnect()")
