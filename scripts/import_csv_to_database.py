#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour importer les données du CSV d'archive dans la base de données SQLite
Remplace les données existantes par celles du fichier CSV
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
from pathlib import Path
from paths import get_db_path, get_data_path

def import_csv_to_database(csv_path=None, db_path=None):
    if csv_path is None:
        csv_path = str(Path(__file__).parent.parent / "data" / "archive" / "WA_Fn-UseC_-HR-Employee-Attrition.csv")
    if db_path is None:
        db_path = get_db_path()
    """Importe les données du CSV dans la base de données SQLite"""
    
    print("=" * 60)
    print("IMPORTATION DES DONNEES CSV DANS LA BASE DE DONNEES")
    print("=" * 60)
    
    # Vérifier que le fichier CSV existe
    if not os.path.exists(csv_path):
        print(f"ERREUR: Le fichier {csv_path} n'existe pas!")
        return False
    
    # Lire le CSV
    print(f"\n1. Lecture du fichier CSV: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        print(f"   OK: {len(df)} lignes lues")
    except Exception as e:
        print(f"   ERREUR lors de la lecture: {e}")
        return False
    
    # Connexion à la base de données
    print("\n2. Connexion à la base de données...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Supprimer les tables existantes pour repartir à zéro
    print("\n3. Suppression des anciennes données...")
    tables = ['turnover', 'performance', 'training', 'overtime', 'absences', 'promotions', 
              'projects', 'satisfaction_surveys', 'csv_original_data', 'employees']
    for table in tables:
        try:
            cursor.execute(f'DROP TABLE IF EXISTS {table}')
        except:
            pass
    print("   OK: Anciennes données supprimées")
    
    # Créer les tables (même structure que create_database.py)
    print("\n4. Création des tables...")
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS employees (
        employee_id TEXT PRIMARY KEY,
        employee_number INTEGER UNIQUE NOT NULL,
        age INTEGER NOT NULL,
        gender TEXT NOT NULL,
        hire_date DATE NOT NULL,
        department TEXT NOT NULL,
        job_level TEXT NOT NULL,
        manager_id TEXT,
        location TEXT NOT NULL,
        salary REAL NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS performance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id TEXT NOT NULL,
        year INTEGER NOT NULL,
        quarter INTEGER NOT NULL,
        performance_rating REAL NOT NULL,
        goals_achieved INTEGER NOT NULL,
        feedback_score REAL NOT NULL,
        review_date DATE NOT NULL,
        FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS promotions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id TEXT NOT NULL,
        old_position TEXT NOT NULL,
        new_position TEXT NOT NULL,
        promotion_date DATE NOT NULL,
        salary_increase REAL NOT NULL,
        FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS training (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id TEXT NOT NULL,
        training_name TEXT NOT NULL,
        training_type TEXT NOT NULL,
        hours_completed INTEGER NOT NULL,
        completion_date DATE NOT NULL,
        score REAL,
        FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS absences (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id TEXT NOT NULL,
        absence_date DATE NOT NULL,
        absence_type TEXT NOT NULL,
        duration_days INTEGER NOT NULL,
        reason TEXT,
        FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS overtime (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id TEXT NOT NULL,
        date DATE NOT NULL,
        hours REAL NOT NULL,
        project_name TEXT,
        approved_by TEXT,
        FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id TEXT NOT NULL,
        project_name TEXT NOT NULL,
        start_date DATE NOT NULL,
        end_date DATE,
        role TEXT NOT NULL,
        status TEXT NOT NULL,
        FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS satisfaction_surveys (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id TEXT NOT NULL,
        survey_date DATE NOT NULL,
        satisfaction_score REAL NOT NULL,
        work_life_balance REAL NOT NULL,
        manager_relationship REAL NOT NULL,
        career_development REAL NOT NULL,
        compensation_fairness REAL NOT NULL,
        team_collaboration REAL NOT NULL,
        comments TEXT,
        FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS turnover (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id TEXT NOT NULL,
        departure_date DATE NOT NULL,
        departure_reason TEXT NOT NULL,
        voluntary BOOLEAN NOT NULL,
        notice_period_days INTEGER,
        exit_interview_score REAL,
        FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
    )
    ''')
    
    # Table pour stocker toutes les variables originales du CSV
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS csv_original_data (
        employee_id TEXT PRIMARY KEY,
        Age INTEGER,
        Attrition TEXT,
        BusinessTravel TEXT,
        DailyRate REAL,
        Department TEXT,
        DistanceFromHome INTEGER,
        Education INTEGER,
        EducationField TEXT,
        EmployeeCount INTEGER,
        EmployeeNumber INTEGER,
        EnvironmentSatisfaction INTEGER,
        Gender TEXT,
        HourlyRate REAL,
        JobInvolvement INTEGER,
        JobLevel INTEGER,
        JobRole TEXT,
        JobSatisfaction INTEGER,
        MaritalStatus TEXT,
        MonthlyIncome REAL,
        MonthlyRate INTEGER,
        NumCompaniesWorked INTEGER,
        Over18 TEXT,
        OverTime TEXT,
        PercentSalaryHike INTEGER,
        PerformanceRating INTEGER,
        RelationshipSatisfaction INTEGER,
        StandardHours INTEGER,
        StockOptionLevel INTEGER,
        TotalWorkingYears INTEGER,
        TrainingTimesLastYear INTEGER,
        WorkLifeBalance INTEGER,
        YearsAtCompany INTEGER,
        YearsInCurrentRole INTEGER,
        YearsSinceLastPromotion INTEGER,
        YearsWithCurrManager INTEGER,
        FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
    )
    ''')
    
    print("   OK: Tables créées")
    
    # Préparer les données des employés
    print("\n5. Préparation des données d'employés...")
    
    employees_data = []
    turnover_data = []
    performance_data = []
    training_data = []
    csv_original_data = []
    
    # Mapping des niveaux de poste
    job_level_mapping = {
        1: 'Junior',
        2: 'Mid',
        3: 'Senior',
        4: 'Lead',
        5: 'Manager'
    }
    
    # Mapping des départements
    department_mapping = {
        'Sales': 'Sales',
        'Research & Development': 'R&D',
        'Human Resources': 'HR'
    }
    
    for idx, row in df.iterrows():
        # ID employé
        employee_id = f'EMP_{row["EmployeeNumber"]:04d}'
        
        # Informations de base
        age = int(row['Age'])
        gender = 'M' if row['Gender'] == 'Male' else 'F'
        department = department_mapping.get(row['Department'], row['Department'])
        
        # Job level
        job_level_num = int(row['JobLevel'])
        job_level = job_level_mapping.get(job_level_num, 'Mid')
        
        # Salaire annuel (MonthlyIncome * 12)
        salary = float(row['MonthlyIncome']) * 12
        
        # Ancienneté (YearsAtCompany)
        years_at_company = float(row['YearsAtCompany'])
        hire_date = datetime.now() - timedelta(days=int(years_at_company * 365.25))
        
        # Utiliser DistanceFromHome comme proxy pour la localisation
        # Catégoriser la distance en zones (0-5: Proche, 6-10: Moyenne, 11-15: Loin, 16+: Très loin)
        distance = int(row['DistanceFromHome'])
        if distance <= 5:
            location = 'Proche'
        elif distance <= 10:
            location = 'Moyenne'
        elif distance <= 15:
            location = 'Loin'
        else:
            location = 'Tres_Loin'
        
        employees_data.append((
            employee_id, int(row['EmployeeNumber']), age, gender,
            hire_date.strftime('%Y-%m-%d'), department, job_level, None,
            location, salary
        ))
        
        # Stocker toutes les données originales du CSV
        csv_original_data.append((
            employee_id,
            int(row['Age']),
            str(row['Attrition']),
            str(row['BusinessTravel']),
            float(row['DailyRate']),
            str(row['Department']),
            int(row['DistanceFromHome']),
            int(row['Education']),
            str(row['EducationField']),
            int(row['EmployeeCount']),
            int(row['EmployeeNumber']),
            int(row['EnvironmentSatisfaction']),
            str(row['Gender']),
            float(row['HourlyRate']),
            int(row['JobInvolvement']),
            int(row['JobLevel']),
            str(row['JobRole']),
            int(row['JobSatisfaction']),
            str(row['MaritalStatus']),
            float(row['MonthlyIncome']),
            int(row['MonthlyRate']),
            int(row['NumCompaniesWorked']),
            str(row['Over18']),
            str(row['OverTime']),
            int(row['PercentSalaryHike']),
            int(row['PerformanceRating']),
            int(row['RelationshipSatisfaction']),
            int(row['StandardHours']),
            int(row['StockOptionLevel']),
            int(row['TotalWorkingYears']),
            int(row['TrainingTimesLastYear']),
            int(row['WorkLifeBalance']),
            int(row['YearsAtCompany']),
            int(row['YearsInCurrentRole']),
            int(row['YearsSinceLastPromotion']),
            int(row['YearsWithCurrManager'])
        ))
        
        # Données de turnover (si Attrition = Yes)
        if row['Attrition'] == 'Yes':
            # Date de départ approximative (basée sur YearsAtCompany)
            # S'assurer qu'il y a au moins quelques jours entre l'embauche et le départ
            days_at_company = max(30, int(years_at_company * 365.25))
            departure_date = hire_date + timedelta(days=days_at_company)
            
            # S'assurer que la date de départ n'est pas dans le futur
            today = datetime.now()
            if departure_date > today:
                departure_date = today - timedelta(days=np.random.randint(1, 90))
            
            # Raisons possibles basées sur les données disponibles
            reasons = ['Better opportunity', 'Career change', 'Dissatisfaction', 
                      'Work-life balance', 'Management issues', 'Company culture']
            departure_reason = random.choice(reasons)
            
            turnover_data.append((
                employee_id, departure_date.strftime('%Y-%m-%d'), departure_reason,
                True, 30, 3.0  # voluntary, notice_period, exit_interview_score
            ))
        
        # Données de performance (PerformanceRating)
        current_year = datetime.now().year
        performance_rating = float(row['PerformanceRating'])
        
        # Générer des évaluations trimestrielles pour les dernières années
        for year in range(max(current_year - 2, hire_date.year), current_year + 1):
            for quarter in range(1, 5):
                if year == hire_date.year and quarter < ((hire_date.month - 1) // 3) + 1:
                    continue
                if year == current_year and quarter > ((datetime.now().month - 1) // 3) + 1:
                    continue
                
                # Variation autour du rating moyen
                rating = performance_rating + np.random.normal(0, 0.3)
                rating = max(1.0, min(5.0, rating))
                
                goals_achieved = int(60 + (rating - 1) * 10 + np.random.normal(0, 5))
                goals_achieved = max(60, min(100, goals_achieved))
                
                feedback_score = rating + np.random.normal(0, 0.2)
                feedback_score = max(1.0, min(5.0, feedback_score))
                
                review_date = datetime(year, quarter * 3, 1)
                
                performance_data.append((
                    employee_id, year, quarter, rating,
                    goals_achieved, feedback_score, review_date.strftime('%Y-%m-%d')
                ))
        
        # Données de formation (TrainingTimesLastYear)
        training_times = int(row['TrainingTimesLastYear'])
        if training_times > 0:
            training_names = [
                'Leadership Fundamentals', 'Project Management', 'Data Analysis', 
                'Communication Skills', 'Technical Skills Update', 'Agile Methodology'
            ]
            
            # Calculer la durée maximale en jours
            max_days = max(30, int(years_at_company * 365.25))
            today = datetime.now()
            days_since_hire = (today - hire_date).days
            
            for i in range(training_times):
                if days_since_hire > 30:
                    days_offset = np.random.randint(30, min(days_since_hire, max_days) + 1)
                else:
                    days_offset = 30
                training_date = hire_date + timedelta(days=days_offset)
                # S'assurer que la date n'est pas dans le futur
                if training_date > today:
                    training_date = today - timedelta(days=np.random.randint(1, 30))
                
                training_name = random.choice(training_names)
                hours = np.random.randint(4, 20)
                score = np.random.normal(85, 10)
                score = max(60, min(100, score))
                
                training_data.append((
                    employee_id, training_name, 'Online', hours,
                    training_date.strftime('%Y-%m-%d'), score
                ))
    
    # Insérer les employés
    print(f"\n6. Insertion de {len(employees_data)} employés...")
    cursor.executemany('''
    INSERT INTO employees (employee_id, employee_number, age, gender, 
                         hire_date, department, job_level, manager_id, location, salary)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', employees_data)
    
    print(f"   OK: {len(employees_data)} employés insérés")
    
    # Insérer les données originales du CSV
    if csv_original_data:
        print(f"\n6b. Insertion de {len(csv_original_data)} enregistrements de données CSV originales...")
        cursor.executemany('''
        INSERT INTO csv_original_data (
            employee_id, Age, Attrition, BusinessTravel, DailyRate, Department,
            DistanceFromHome, Education, EducationField, EmployeeCount, EmployeeNumber,
            EnvironmentSatisfaction, Gender, HourlyRate, JobInvolvement, JobLevel,
            JobRole, JobSatisfaction, MaritalStatus, MonthlyIncome, MonthlyRate,
            NumCompaniesWorked, Over18, OverTime, PercentSalaryHike, PerformanceRating,
            RelationshipSatisfaction, StandardHours, StockOptionLevel, TotalWorkingYears,
            TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany, YearsInCurrentRole,
            YearsSinceLastPromotion, YearsWithCurrManager
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', csv_original_data)
        print(f"   OK: {len(csv_original_data)} enregistrements CSV insérés")
    
    # Insérer les données de turnover
    if turnover_data:
        print(f"\n7. Insertion de {len(turnover_data)} départs...")
        cursor.executemany('''
        INSERT INTO turnover (employee_id, departure_date, departure_reason, 
                            voluntary, notice_period_days, exit_interview_score)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', turnover_data)
        print(f"   OK: {len(turnover_data)} départs insérés")
    
    # Insérer les données de performance
    if performance_data:
        print(f"\n8. Insertion de {len(performance_data)} évaluations de performance...")
        cursor.executemany('''
        INSERT INTO performance (employee_id, year, quarter, performance_rating, 
                               goals_achieved, feedback_score, review_date)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', performance_data)
        print(f"   OK: {len(performance_data)} évaluations insérées")
    
    # Insérer les données de formation
    if training_data:
        print(f"\n9. Insertion de {len(training_data)} formations...")
        cursor.executemany('''
        INSERT INTO training (employee_id, training_name, training_type, hours_completed, 
                           completion_date, score)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', training_data)
        print(f"   OK: {len(training_data)} formations insérées")
    
    # Commit et fermeture
    conn.commit()
    conn.close()
    
    print("\n" + "=" * 60)
    print("IMPORTATION TERMINEE AVEC SUCCES!")
    print("=" * 60)
    print(f"Fichier CSV: {csv_path}")
    print(f"Base de données: {db_path}")
    print(f"Employés: {len(employees_data)}")
    print(f"Départs: {len(turnover_data)}")
    print(f"Évaluations: {len(performance_data)}")
    print(f"Formations: {len(training_data)}")
    
    return True

if __name__ == "__main__":
    success = import_csv_to_database()
    if success:
        print("\n[OK] Les donnees du CSV ont ete importees avec succes!")
        print("  Vous pouvez maintenant utiliser la base de donnees avec:")
        print("    python main.py")
        print("    python dashboard.py")
    else:
        print("\n[ERREUR] Erreur lors de l'importation")

