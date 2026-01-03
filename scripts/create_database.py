#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Générateur de base de données SQLite pour le système de prédiction du turnover
Crée une base de données réaliste avec des données d'employés générées par LLM
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json
import os
import yaml
from typing import List, Dict, Any, Optional
import time
from paths import get_config_path, get_db_path

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("ATTENTION: Bibliothèque openai non disponible. Installation: pip install openai")


class LLMDataGenerator:
    """Générateur de données réalistes utilisant une LLM"""
    
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = get_config_path()
        self.config = self.load_config(config_path)
        self.llm_config = self.config.get('llm_data_generation', {})
        self.enabled = self.llm_config.get('enabled', False) and OPENAI_AVAILABLE
        
        if self.enabled:
            api_key = os.getenv('OPENAI_API_KEY') or self.llm_config.get('api_key', '').replace('${OPENAI_API_KEY}', os.getenv('OPENAI_API_KEY', ''))
            if api_key:
                self.client = OpenAI(api_key=api_key)
                self.model = self.llm_config.get('model', 'gpt-4o-mini')
                self.temperature = self.llm_config.get('temperature', 0.7)
                self.max_tokens = self.llm_config.get('max_tokens', 500)
                self.batch_size = self.llm_config.get('batch_size', 20)
                print(f"✓ LLM activé: {self.model}")
            else:
                print("⚠ Clé API OpenAI non trouvée. Utilisation de la génération aléatoire.")
                self.enabled = False
        else:
            print("ℹ Génération LLM désactivée. Utilisation de la génération aléatoire.")
    
    def load_config(self, config_path):
        """Charge la configuration"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"⚠ Erreur chargement config: {e}. Utilisation des valeurs par défaut.")
            return {}
    
    def generate_employee_profiles_batch(self, n_profiles: int, departments: List[str], 
                                        job_levels: List[str], locations: List[str]) -> List[Dict]:
        """Génère un batch de profils d'employés réalistes avec la LLM"""
        if not self.enabled:
            return []
        
        prompt = f"""Génère {n_profiles} profils d'employés français réalistes pour une entreprise. 
Pour chaque profil, fournis les informations suivantes au format JSON strict:

{{
  "first_name": "Prénom français réaliste",
  "last_name": "Nom de famille français réaliste",
  "age": nombre entre 22 et 65,
  "gender": "M" ou "F",
  "department": un parmi {departments},
  "job_level": un parmi {job_levels},
  "location": un parmi {locations},
  "hire_date_years_ago": nombre entre 0.08 et 10 (années depuis l'embauche)
}}

Important:
- Les noms doivent être variés et réalistes (noms français courants)
- L'âge doit être cohérent avec le niveau de poste (Junior: 22-30, Mid: 26-40, Senior: 30-50, Lead/Manager: 35-55, Director: 40-65)
- Le salaire sera calculé automatiquement, ne l'inclut pas
- Retourne un tableau JSON valide avec exactement {n_profiles} objets
- Format: [{{...}}, {{...}}, ...]
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Tu es un expert en génération de données RH réalistes. Tu retournes uniquement du JSON valide, sans texte supplémentaire."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens * n_profiles // 5  # Ajustement pour batch
            )
            
            content = response.choices[0].message.content.strip()
            # Nettoyer le contenu (enlever markdown si présent)
            if content.startswith('```json'):
                content = content[7:]
            if content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            profiles = json.loads(content)
            if not isinstance(profiles, list):
                profiles = [profiles]
            
            return profiles[:n_profiles]
            
        except Exception as e:
            print(f"⚠ Erreur LLM: {e}. Retour à la génération aléatoire pour ce batch.")
            return []
    
    def generate_all_employees(self, n_employees: int) -> List[Dict]:
        """Génère tous les profils d'employés"""
        if not self.enabled:
            return []
        
        departments = ['IT', 'Sales', 'HR', 'Finance', 'Operations', 'Marketing', 'Legal', 'R&D']
        job_levels = ['Junior', 'Mid', 'Senior', 'Lead', 'Manager', 'Director']
        locations = ['Paris', 'Lyon', 'Marseille', 'Toulouse', 'Nice', 'Nantes', 'Strasbourg', 'Montpellier']
        
        all_profiles = []
        n_batches = (n_employees + self.batch_size - 1) // self.batch_size
        
        print(f"Génération de {n_employees} profils avec LLM ({n_batches} batches)...")
        
        for i in range(n_batches):
            batch_size = min(self.batch_size, n_employees - len(all_profiles))
            print(f"  Batch {i+1}/{n_batches} ({batch_size} profils)...", end=' ', flush=True)
            
            profiles = self.generate_employee_profiles_batch(
                batch_size, departments, job_levels, locations
            )
            
            if profiles:
                all_profiles.extend(profiles)
                print(f"✓ {len(profiles)} profils générés")
            else:
                print("⚠ Échec, génération aléatoire pour ce batch")
                # Fallback: générer quelques profils aléatoires
                for _ in range(batch_size):
                    all_profiles.append({
                        'first_name': random.choice(['Jean', 'Marie', 'Pierre', 'Sophie']),
                        'last_name': random.choice(['Martin', 'Bernard', 'Dubois', 'Moreau']),
                        'age': np.random.randint(22, 66),
                        'gender': random.choice(['M', 'F']),
                        'department': random.choice(departments),
                        'job_level': random.choice(job_levels),
                        'location': random.choice(locations),
                        'hire_date_years_ago': np.random.uniform(0.08, 10)
                    })
            
            # Petite pause pour éviter les limites de taux
            if i < n_batches - 1:
                time.sleep(0.5)
        
        return all_profiles[:n_employees]


def create_database():
    """Crée la base de données SQLite avec des données réalistes"""
    
    print("=" * 60)
    print("CREATION DE LA BASE DE DONNEES TURNOVER")
    print("=" * 60)
    
    # Connexion à la base de données
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    
    print("\n1. Creation des tables...")
    
    # Table des employés
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS employees (
        employee_id TEXT PRIMARY KEY,
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
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
    
    # Table des performances
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
    
    # Table des promotions
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
    
    # Table des formations
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
    
    # Table des absences
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
    
    # Table des heures supplémentaires
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
    
    # Table des projets
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
    
    # Table des enquêtes de satisfaction
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
    
    # Table des départs
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
    
    print("OK Tables creees")
    
    print("\n2. Generation des donnees d'employes...")
    
    # Initialiser le générateur LLM
    llm_generator = LLMDataGenerator()
    
    # Génération de données réalistes
    np.random.seed(42)
    n_employees = 1000
    
    # Données de base (pour fallback)
    first_names = ['Jean', 'Marie', 'Pierre', 'Sophie', 'Lucas', 'Emma', 'Thomas', 'Camille', 
                   'Antoine', 'Julie', 'Nicolas', 'Sarah', 'Alexandre', 'Laura', 'Maxime',
                   'Chloé', 'Julien', 'Manon', 'Romain', 'Léa', 'Paul', 'Clara', 'Hugo', 'Inès']
    
    last_names = ['Martin', 'Bernard', 'Thomas', 'Petit', 'Robert', 'Richard', 'Durand', 'Dubois',
                  'Moreau', 'Laurent', 'Simon', 'Michel', 'Lefebvre', 'Leroy', 'Roux', 'David',
                  'Bertrand', 'Morel', 'Fournier', 'Girard', 'André', 'Lopez', 'Rousseau', 'Vincent']
    
    departments = ['IT', 'Sales', 'HR', 'Finance', 'Operations', 'Marketing', 'Legal', 'R&D']
    job_levels = ['Junior', 'Mid', 'Senior', 'Lead', 'Manager', 'Director']
    locations = ['Paris', 'Lyon', 'Marseille', 'Toulouse', 'Nice', 'Nantes', 'Strasbourg', 'Montpellier']
    genders = ['M', 'F']
    
    # Générer les profils avec LLM si activé
    llm_profiles = []
    if llm_generator.enabled:
        llm_profiles = llm_generator.generate_all_employees(n_employees)
    
    employees_data = []
    
    # Salaire de base par niveau
    base_salaries = {'Junior': 35000, 'Mid': 45000, 'Senior': 60000, 'Lead': 75000, 'Manager': 90000, 'Director': 120000}
    
    for i in range(n_employees):
        # Utiliser le profil LLM si disponible, sinon génération aléatoire
        if i < len(llm_profiles) and llm_profiles[i]:
            profile = llm_profiles[i]
            first_name = profile.get('first_name', random.choice(first_names))
            last_name = profile.get('last_name', random.choice(last_names))
            age = int(profile.get('age', np.random.randint(22, 66)))
            gender = profile.get('gender', random.choice(genders))
            department = profile.get('department', random.choice(departments))
            job_level = profile.get('job_level', random.choice(job_levels))
            location = profile.get('location', random.choice(locations))
            
            # Calculer la date d'embauche à partir des années
            years_ago = profile.get('hire_date_years_ago', np.random.uniform(0.08, 10))
            hire_date = datetime.now() - timedelta(days=int(years_ago * 365.25))
        else:
            # Fallback: génération aléatoire
            first_name = random.choice(first_names)
            last_name = random.choice(last_names)
            age = np.random.randint(22, 66)
            gender = random.choice(genders)
            department = random.choice(departments)
            job_level = random.choice(job_levels)
            location = random.choice(locations)
            hire_date = datetime.now() - timedelta(days=np.random.randint(30, 3650))
        
        employee_id = f'EMP_{i+1:04d}'
        email = f'{first_name.lower().replace(" ", "")}.{last_name.lower().replace(" ", "")}.{i+1}@company.com'
        
        # Calcul du salaire basé sur le niveau et l'ancienneté
        base_salary = base_salaries.get(job_level, 50000)
        tenure_years = (datetime.now() - hire_date).days / 365.25
        salary = base_salary * (1 + tenure_years * 0.03) + np.random.normal(0, base_salary * 0.1)
        salary = max(30000, salary)  # Salaire minimum
        
        employees_data.append((
            employee_id, first_name, last_name, email, age, gender,
            hire_date.strftime('%Y-%m-%d'), department, job_level, None,
            location, salary
        ))
    
    # Insertion des employés
    cursor.executemany('''
    INSERT INTO employees (employee_id, first_name, last_name, email, age, gender, 
                         hire_date, department, job_level, manager_id, location, salary)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', employees_data)
    
    print(f"OK {n_employees} employes generes")
    
    print("\n3. Generation des donnees de performance...")
    
    performance_data = []
    for emp in employees_data:
        employee_id = emp[0]
        hire_date = datetime.strptime(emp[6], '%Y-%m-%d')
        
        # Générer des évaluations trimestrielles depuis l'embauche
        current_date = datetime.now()
        quarters = []
        
        start_year = hire_date.year
        start_quarter = ((hire_date.month - 1) // 3) + 1
        
        for year in range(start_year, current_date.year + 1):
            for quarter in range(1, 5):
                if year == start_year and quarter < start_quarter:
                    continue
                if year == current_date.year and quarter > ((current_date.month - 1) // 3) + 1:
                    continue
                
                quarters.append((year, quarter))
        
        for year, quarter in quarters:
            # Performance rating entre 1 et 5
            performance_rating = np.random.normal(3.5, 0.8)
            performance_rating = max(1.0, min(5.0, performance_rating))
            
            goals_achieved = np.random.randint(60, 101)
            feedback_score = np.random.normal(3.2, 0.6)
            feedback_score = max(1.0, min(5.0, feedback_score))
            
            review_date = datetime(year, quarter * 3, 1)
            
            performance_data.append((
                employee_id, year, quarter, performance_rating,
                goals_achieved, feedback_score, review_date.strftime('%Y-%m-%d')
            ))
    
    cursor.executemany('''
    INSERT INTO performance (employee_id, year, quarter, performance_rating, 
                           goals_achieved, feedback_score, review_date)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', performance_data)
    
    print(f"OK {len(performance_data)} evaluations de performance generees")
    
    print("\n4. Generation des donnees de formation...")
    
    training_data = []
    training_names = [
        'Leadership Fundamentals', 'Project Management', 'Data Analysis', 'Communication Skills',
        'Technical Skills Update', 'Agile Methodology', 'Customer Service', 'Financial Analysis',
        'Digital Marketing', 'Cybersecurity Basics', 'Team Building', 'Time Management',
        'Presentation Skills', 'Negotiation Techniques', 'Strategic Thinking'
    ]
    
    training_types = ['Online', 'Classroom', 'Workshop', 'Conference', 'Certification']
    
    for emp in employees_data:
        employee_id = emp[0]
        hire_date = datetime.strptime(emp[6], '%Y-%m-%d')
        
        # Nombre de formations (entre 2 et 8)
        n_trainings = np.random.randint(2, 9)
        
        for _ in range(n_trainings):
            training_name = random.choice(training_names)
            training_type = random.choice(training_types)
            hours = np.random.randint(4, 40)
            
            # Date de formation après l'embauche
            training_date = hire_date + timedelta(days=np.random.randint(30, (datetime.now() - hire_date).days))
            
            score = np.random.normal(85, 10)
            score = max(60, min(100, score))
            
            training_data.append((
                employee_id, training_name, training_type, hours,
                training_date.strftime('%Y-%m-%d'), score
            ))
    
    cursor.executemany('''
    INSERT INTO training (employee_id, training_name, training_type, hours_completed, 
                        completion_date, score)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', training_data)
    
    print(f"OK {len(training_data)} formations generees")
    
    print("\n5. Generation des donnees de turnover...")
    
    # Générer des départs pour environ 15% des employés
    n_departures = int(n_employees * 0.15)
    departed_employees = random.sample(employees_data, n_departures)
    
    turnover_data = []
    departure_reasons = [
        'Better opportunity', 'Career change', 'Relocation', 'Family reasons',
        'Dissatisfaction', 'Retirement', 'Health issues', 'Startup venture',
        'Higher salary', 'Work-life balance', 'Management issues', 'Company culture'
    ]
    
    for emp in departed_employees:
        employee_id = emp[0]
        hire_date = datetime.strptime(emp[6], '%Y-%m-%d')
        
        # Date de départ (entre 6 mois et 5 ans après l'embauche)
        tenure_days = np.random.randint(180, 1825)
        departure_date = hire_date + timedelta(days=tenure_days)
        
        departure_reason = random.choice(departure_reasons)
        voluntary = random.choice([True, False])
        
        notice_period = np.random.randint(14, 90) if voluntary else 0
        exit_interview_score = np.random.normal(3.0, 1.0)
        exit_interview_score = max(1.0, min(5.0, exit_interview_score))
        
        turnover_data.append((
            employee_id, departure_date.strftime('%Y-%m-%d'), departure_reason,
            voluntary, notice_period, exit_interview_score
        ))
    
    cursor.executemany('''
    INSERT INTO turnover (employee_id, departure_date, departure_reason, 
                        voluntary, notice_period_days, exit_interview_score)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', turnover_data)
    
    print(f"OK {len(turnover_data)} departs generes")
    
    print("\n6. Generation des donnees supplementaires...")
    
    # Générer des heures supplémentaires
    overtime_data = []
    for emp in employees_data:
        employee_id = emp[0]
        hire_date = datetime.strptime(emp[6], '%Y-%m-%d')
        
        # Nombre d'entrées d'heures sup (entre 5 et 50)
        n_overtime = np.random.randint(5, 51)
        
        for _ in range(n_overtime):
            overtime_date = hire_date + timedelta(days=np.random.randint(30, (datetime.now() - hire_date).days))
            hours = np.random.exponential(4)  # La plupart des heures sup sont courtes
            hours = min(12, hours)  # Maximum 12h par jour
            
            project_name = f'Project_{np.random.randint(1, 21)}'
            approved_by = f'Manager_{np.random.randint(1, 11)}'
            
            overtime_data.append((
                employee_id, overtime_date.strftime('%Y-%m-%d'), hours, project_name, approved_by
            ))
    
    cursor.executemany('''
    INSERT INTO overtime (employee_id, date, hours, project_name, approved_by)
    VALUES (?, ?, ?, ?, ?)
    ''', overtime_data)
    
    # Générer des absences
    absence_data = []
    absence_types = ['Sick leave', 'Vacation', 'Personal', 'Maternity/Paternity', 'Bereavement', 'Medical']
    
    for emp in employees_data:
        employee_id = emp[0]
        hire_date = datetime.strptime(emp[6], '%Y-%m-%d')
        
        # Nombre d'absences (entre 2 et 15)
        n_absences = np.random.randint(2, 16)
        
        for _ in range(n_absences):
            absence_date = hire_date + timedelta(days=np.random.randint(30, (datetime.now() - hire_date).days))
            absence_type = random.choice(absence_types)
            duration = np.random.randint(1, 10) if absence_type == 'Sick leave' else np.random.randint(1, 21)
            
            reasons = ['Medical appointment', 'Family emergency', 'Personal matter', 'Mental health day', 'Car trouble']
            reason = random.choice(reasons)
            
            absence_data.append((
                employee_id, absence_date.strftime('%Y-%m-%d'), absence_type, duration, reason
            ))
    
    cursor.executemany('''
    INSERT INTO absences (employee_id, absence_date, absence_type, duration_days, reason)
    VALUES (?, ?, ?, ?, ?)
    ''', absence_data)
    
    print(f"OK {len(overtime_data)} heures supplementaires generees")
    print(f"OK {len(absence_data)} absences generees")
    
    # Commit et fermeture
    conn.commit()
    conn.close()
    
    print("\n" + "=" * 60)
    print("BASE DE DONNEES CREE AVEC SUCCES!")
    print("=" * 60)
    print(f"Fichier: {get_db_path()}")
    print(f"Employes: {n_employees}")
    print(f"Evaluations: {len(performance_data)}")
    print(f"Formations: {len(training_data)}")
    print(f"Departs: {len(turnover_data)}")
    print(f"Heures sup: {len(overtime_data)}")
    print(f"Absences: {len(absence_data)}")
    
    return True

def show_database_info():
    """Affiche les informations sur la base de données"""
    conn = sqlite3.connect(get_db_path())
    
    print("\n" + "=" * 60)
    print("INFORMATIONS SUR LA BASE DE DONNEES")
    print("=" * 60)
    
    # Lister les tables
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print("\nTables disponibles:")
    for table in tables:
        table_name = table[0]
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"  - {table_name}: {count} enregistrements")
    
    # Afficher quelques exemples
    print("\nExemples de donnees:")
    
    # Employés
    cursor.execute("SELECT employee_id, first_name, last_name, department, job_level, salary FROM employees LIMIT 5")
    employees = cursor.fetchall()
    print("\nEmployes:")
    for emp in employees:
        print(f"  {emp[0]}: {emp[1]} {emp[2]} - {emp[3]} {emp[4]} - {emp[5]:.0f}€")
    
    # Performance
    cursor.execute("SELECT employee_id, year, quarter, performance_rating FROM performance LIMIT 5")
    performances = cursor.fetchall()
    print("\nPerformances:")
    for perf in performances:
        print(f"  {perf[0]}: Q{perf[2]} {perf[1]} - {perf[3]:.1f}/5")
    
    # Turnover
    cursor.execute("SELECT employee_id, departure_date, departure_reason FROM turnover LIMIT 5")
    turnovers = cursor.fetchall()
    print("\nDeparts:")
    for turn in turnovers:
        print(f"  {turn[0]}: {turn[1]} - {turn[2]}")
    
    conn.close()

if __name__ == "__main__":
    success = create_database()
    if success:
        show_database_info()
        print("\nPour utiliser la base de donnees:")
        print("  python demo_ia.py")
        print("  # ou")
        print("  python main.py")
