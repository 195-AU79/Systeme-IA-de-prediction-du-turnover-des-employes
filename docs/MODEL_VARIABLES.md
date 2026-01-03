# Variables du Modèle de Prédiction du Turnover

## Table des matières
1. [Vue d'ensemble](#vue-densemble)
2. [Variables Numériques](#variables-numériques)
3. [Variables Catégorielles](#variables-catégorielles)
4. [Importance des Variables](#importance-des-variables)
5. [Préparation des Données](#préparation-des-données)

---

## Vue d'ensemble

Le modèle de prédiction du turnover utilise **30 variables** sélectionnées automatiquement parmi un ensemble initial de variables disponibles. Ces variables sont sélectionnées à l'aide de la méthode `SelectKBest` avec `mutual_info_classif` pour capturer les relations non-linéaires entre les features et la variable cible.

**Méthode de sélection :** `SelectKBest` avec `mutual_info_classif` (k=30)

**Total de variables sélectionnées :** 30 features

---

## Variables Numériques

Les variables numériques sont normalisées à l'aide de `StandardScaler` avant l'entraînement du modèle.

### 1. Démographie et Caractéristiques Personnelles

#### `age`
- **Type :** Numérique (entier)
- **Description :** Âge de l'employé en années
- **Importance :** 6.45% (4ème variable la plus importante)
- **Plage typique :** 18-65 ans
- **Utilité :** Les jeunes employés et les employés proches de la retraite ont tendance à avoir des taux de turnover différents

#### `total_working_years`
- **Type :** Numérique (entier)
- **Description :** Nombre total d'années d'expérience professionnelle
- **Importance :** 6.25% (7ème variable la plus importante)
- **Plage typique :** 0-40 ans
- **Utilité :** L'expérience professionnelle globale influence la stabilité de l'emploi

---

### 2. Ancienneté et Stabilité

#### `tenure_years`
- **Type :** Numérique (décimal)
- **Description :** Nombre d'années passées dans l'entreprise actuelle
- **Importance :** 6.56% (3ème variable la plus importante)
- **Plage typique :** 0-40 ans
- **Utilité :** L'ancienneté est un indicateur fort de la probabilité de départ

#### `years_with_curr_manager`
- **Type :** Numérique (décimal)
- **Description :** Nombre d'années avec le manager actuel
- **Importance :** 7.67% (2ème variable la plus importante)
- **Plage typique :** 0-20 ans
- **Utilité :** La stabilité de la relation manager-employé est un facteur clé de rétention

#### `years_in_current_role`
- **Type :** Numérique (décimal)
- **Description :** Nombre d'années dans le poste actuel
- **Importance :** 5.22% (11ème variable la plus importante)
- **Plage typique :** 0-20 ans
- **Utilité :** La stagnation dans un poste peut être un facteur de départ

---

### 3. Salaire et Compensation

#### `salary`
- **Type :** Numérique (décimal)
- **Description :** Salaire annuel de l'employé (en unités monétaires)
- **Importance :** 6.30% (6ème variable la plus importante)
- **Plage typique :** Variable selon le poste et le niveau
- **Utilité :** Le niveau de rémunération influence directement la satisfaction et la rétention

#### `percent_salary_hike`
- **Type :** Numérique (décimal)
- **Description :** Pourcentage d'augmentation salariale lors de la dernière révision
- **Importance :** 3.42% (16ème variable la plus importante)
- **Plage typique :** 0-25%
- **Utilité :** Les augmentations salariales récentes peuvent influencer la décision de rester

#### `stock_option_level`
- **Type :** Numérique (entier)
- **Description :** Niveau d'options d'achat d'actions (0-3)
- **Importance :** 7.94% (1ère variable la plus importante)
- **Plage typique :** 0-3
- **Utilité :** Les avantages financiers à long terme sont un facteur de rétention important

#### `DailyRate`
- **Type :** Numérique (décimal)
- **Description :** Taux journalier de rémunération
- **Importance :** 4.44% (12ème variable la plus importante)
- **Plage typique :** Variable selon le poste
- **Utilité :** Indicateur complémentaire du niveau de rémunération

#### `HourlyRate`
- **Type :** Numérique (décimal)
- **Description :** Taux horaire de rémunération
- **Importance :** 3.93% (13ème variable la plus importante)
- **Plage typique :** Variable selon le poste
- **Utilité :** Indicateur complémentaire du niveau de rémunération

#### `MonthlyRate`
- **Type :** Numérique (décimal)
- **Description :** Taux mensuel de rémunération
- **Importance :** 3.71% (14ème variable la plus importante)
- **Plage typique :** Variable selon le poste
- **Utilité :** Indicateur complémentaire du niveau de rémunération

---

### 4. Satisfaction et Engagement

#### `environment_satisfaction`
- **Type :** Numérique (entier)
- **Description :** Niveau de satisfaction avec l'environnement de travail (1-4)
- **Importance :** 6.19% (8ème variable la plus importante)
- **Plage typique :** 1 (Très insatisfait) à 4 (Très satisfait)
- **Utilité :** L'environnement de travail est un facteur clé de rétention

---

### 5. Autres Variables Numériques Disponibles

Les variables suivantes sont disponibles dans le dataset mais peuvent ne pas être sélectionnées dans le modèle final selon leur importance :

- `distance_from_home` : Distance du domicile au travail (en km)
- `performance_rating` : Note de performance (1-4)
- `training_times` : Nombre de formations suivies l'année dernière
- `work_life_balance` : Équilibre vie/travail (1-4)
- `job_satisfaction` : Satisfaction au travail (1-4)
- `relationship_satisfaction` : Satisfaction avec les relations professionnelles (1-4)
- `job_involvement` : Implication dans le travail (1-4)
- `years_since_last_promotion` : Années depuis la dernière promotion
- `num_companies_worked` : Nombre d'entreprises précédentes
- `education` : Niveau d'éducation (1-5)
- `StandardHours` : Heures standard de travail

---

## Variables Catégorielles

Les variables catégorielles sont encodées en variables binaires (one-hot encoding) avant d'être utilisées dans le modèle.

### 1. Département (`department`)

**Type :** Catégoriel (one-hot encoding)

**Catégories disponibles :**
- `department_Research & Development` (Importance : 3.10%)
- `department_Human Resources` (Importance : 0.07%)
- Autres départements (Sales, Research & Development, etc.)

**Description :** Département dans lequel l'employé travaille

**Utilité :** Certains départements peuvent avoir des taux de turnover différents selon leur culture et leurs conditions de travail

---

### 2. Niveau Hiérarchique (`job_level`)

**Type :** Catégoriel (one-hot encoding)

**Catégories disponibles :**
- `job_level_1` (Importance : 5.40% - 10ème variable la plus importante)
- `job_level_5` (Importance : 0.11%)
- Autres niveaux (2, 3, 4)

**Description :** Niveau hiérarchique dans l'entreprise (1 = junior, 5 = senior executive)

**Utilité :** Le niveau hiérarchique influence les opportunités de carrière et la satisfaction

---

### 3. Heures Supplémentaires (`overtime`)

**Type :** Catégoriel (one-hot encoding)

**Catégories disponibles :**
- `overtime_No` (Importance : 6.38% - 5ème variable la plus importante)
- `overtime_Yes` (Importance : 6.12% - 9ème variable la plus importante)

**Description :** Indique si l'employé fait régulièrement des heures supplémentaires

**Utilité :** Les heures supplémentaires excessives sont un facteur de stress et de turnover

---

### 4. Statut Matrimonial (`marital_status`)

**Type :** Catégoriel (one-hot encoding)

**Catégories disponibles :**
- `marital_status_Single` (Importance : 3.42% - 15ème variable la plus importante)
- `marital_status_Married` (Importance : 1.52%)
- `marital_status_Divorced` (Importance : 0.99%)

**Description :** Statut matrimonial de l'employé

**Utilité :** Le statut matrimonial peut influencer les priorités et la stabilité professionnelle

---

### 5. Domaine d'Études (`education_field`)

**Type :** Catégoriel (one-hot encoding)

**Catégories disponibles :**
- `education_field_Medical` (Importance : 1.37%)
- `education_field_Life Sciences` (Importance : 1.30%)
- `education_field_Technical Degree` (Importance : 0.40%)
- `education_field_Other` (Importance : 0.12%)
- `education_field_Human Resources` (Importance : 0.06%)
- Autres domaines (Marketing, Technical Degree, etc.)

**Description :** Domaine d'études de l'employé

**Utilité :** Le domaine d'études peut influencer les opportunités de carrière et la mobilité

---

### 6. Rôle Professionnel (`job_role`)

**Type :** Catégoriel (one-hot encoding)

**Catégories disponibles :**
- `job_role_Manufacturing Director` (Importance : 0.59%)
- `job_role_Sales Representative` (Importance : 0.44%)
- `job_role_Manager` (Importance : 0.28%)
- `job_role_Research Director` (Importance : 0.27%)
- Autres rôles (Healthcare Representative, Research Scientist, etc.)

**Description :** Rôle professionnel spécifique de l'employé

**Utilité :** Certains rôles peuvent avoir des taux de turnover plus élevés que d'autres

---

### 7. Autres Variables Catégorielles Disponibles

Les variables suivantes sont disponibles dans le dataset mais peuvent ne pas être sélectionnées dans le modèle final :

- `location` : Localisation géographique (Proche, Moyenne, Loin, Tres_Loin)
- `gender` : Genre (Male, Female)
- `business_travel` : Fréquence des voyages professionnels (Non-Travel, Travel_Rarely, Travel_Frequently)

---

## Importance des Variables

### Top 10 Variables les Plus Importantes

| Rang | Variable | Importance | Type |
|------|----------|------------|------|
| 1 | `stock_option_level` | 7.94% | Numérique |
| 2 | `years_with_curr_manager` | 7.67% | Numérique |
| 3 | `tenure_years` | 6.56% | Numérique |
| 4 | `age` | 6.45% | Numérique |
| 5 | `overtime_No` | 6.38% | Catégoriel |
| 6 | `salary` | 6.30% | Numérique |
| 7 | `total_working_years` | 6.25% | Numérique |
| 8 | `environment_satisfaction` | 6.19% | Numérique |
| 9 | `overtime_Yes` | 6.12% | Catégoriel |
| 10 | `job_level_1` | 5.40% | Catégoriel |

### Analyse de l'Importance

**Variables les plus critiques :**
1. **Compensation financière** : `stock_option_level` et `salary` sont parmi les variables les plus importantes
2. **Stabilité relationnelle** : `years_with_curr_manager` est la 2ème variable la plus importante
3. **Ancienneté** : `tenure_years` est un indicateur fort de la probabilité de départ
4. **Charge de travail** : Les variables `overtime_No` et `overtime_Yes` sont très importantes

**Insights clés :**
- Les facteurs financiers (stock options, salaire) sont cruciaux
- La relation avec le manager est un facteur de rétention majeur
- L'ancienneté et l'âge sont des indicateurs importants
- Les heures supplémentaires sont un facteur de risque significatif

---

## Préparation des Données

### Traitement des Variables Numériques

1. **Normalisation** : Toutes les variables numériques sont normalisées à l'aide de `StandardScaler`
   - Moyenne = 0
   - Écart-type = 1

2. **Gestion des valeurs manquantes** : Les valeurs manquantes sont remplacées par 0

3. **Sélection** : Seules les variables disponibles dans le dataset sont utilisées

### Traitement des Variables Catégorielles

1. **One-Hot Encoding** : Chaque catégorie devient une variable binaire (0 ou 1)
   - Exemple : `department` → `department_Research & Development`, `department_Sales`, etc.

2. **Préfixe** : Chaque variable dummy est préfixée par le nom de la variable originale
   - Format : `{variable}_{valeur}`

3. **Gestion des valeurs manquantes** : Les valeurs manquantes sont converties en chaînes de caractères avant l'encodage

### Sélection des Features

**Méthode :** `SelectKBest` avec `mutual_info_classif`

**Critères :**
- Sélection des 30 meilleures features (ou moins si moins de features disponibles)
- Basé sur l'information mutuelle avec la variable cible
- Capture les relations non-linéaires

**Résultat :** 30 variables sélectionnées automatiquement parmi toutes les variables disponibles

---

## Mapping des Colonnes CSV

Les colonnes du CSV original sont mappées vers les noms de variables utilisés dans le modèle :

| Colonne CSV | Variable Modèle | Type |
|-------------|-----------------|------|
| `Age` | `age` | Numérique |
| `MonthlyIncome` | `salary` (×12 pour annuel) | Numérique |
| `YearsAtCompany` | `tenure_years` | Numérique |
| `DistanceFromHome` | `distance_from_home` | Numérique |
| `PerformanceRating` | `performance_rating` | Numérique |
| `TrainingTimesLastYear` | `training_times` | Numérique |
| `WorkLifeBalance` | `work_life_balance` | Numérique |
| `JobSatisfaction` | `job_satisfaction` | Numérique |
| `EnvironmentSatisfaction` | `environment_satisfaction` | Numérique |
| `RelationshipSatisfaction` | `relationship_satisfaction` | Numérique |
| `JobInvolvement` | `job_involvement` | Numérique |
| `YearsInCurrentRole` | `years_in_current_role` | Numérique |
| `YearsSinceLastPromotion` | `years_since_last_promotion` | Numérique |
| `YearsWithCurrManager` | `years_with_curr_manager` | Numérique |
| `NumCompaniesWorked` | `num_companies_worked` | Numérique |
| `TotalWorkingYears` | `total_working_years` | Numérique |
| `PercentSalaryHike` | `percent_salary_hike` | Numérique |
| `StockOptionLevel` | `stock_option_level` | Numérique |
| `Education` | `education` | Numérique |
| `DailyRate` | `DailyRate` | Numérique |
| `HourlyRate` | `HourlyRate` | Numérique |
| `MonthlyRate` | `MonthlyRate` | Numérique |
| `StandardHours` | `StandardHours` | Numérique |
| `Department` | `department` | Catégoriel |
| `JobLevel` | `job_level` | Catégoriel |
| `OverTime` | `overtime` | Catégoriel |
| `MaritalStatus` | `marital_status` | Catégoriel |
| `EducationField` | `education_field` | Catégoriel |
| `JobRole` | `job_role` | Catégoriel |
| `Gender` | `gender` | Catégoriel |
| `BusinessTravel` | `business_travel` | Catégoriel |

**Note :** La variable `location` est créée à partir de `distance_from_home` :
- 0-5 km → "Proche"
- 5-10 km → "Moyenne"
- 10-15 km → "Loin"
- >15 km → "Tres_Loin"

---

## Utilisation dans le Modèle

### Pipeline de Traitement

1. **Chargement** : Données chargées depuis la base de données SQLite
2. **Mapping** : Colonnes CSV mappées vers les noms de variables
3. **Transformation** : 
   - Variables numériques : Normalisation avec `StandardScaler`
   - Variables catégorielles : One-hot encoding
4. **Sélection** : `SelectKBest` sélectionne les 30 meilleures features
5. **Entraînement** : Modèle RandomForest entraîné sur les features sélectionnées

### Variables Finales Utilisées

Le modèle final utilise exactement **30 variables** sélectionnées automatiquement. La liste exacte des variables peut varier légèrement selon les données, mais inclut généralement :

- Les variables numériques les plus importantes (age, salary, tenure_years, etc.)
- Les variables catégorielles encodées les plus importantes (overtime_No, overtime_Yes, job_level_1, etc.)

Pour obtenir la liste exacte des variables utilisées dans votre modèle, consultez le fichier `data/criteria_analysis_results.json` dans la section `feature_importance`.

---

## Notes Importantes

1. **Normalisation** : Toutes les variables numériques sont normalisées avant l'entraînement
2. **Encodage** : Les variables catégorielles sont encodées en variables binaires
3. **Sélection automatique** : Les 30 meilleures variables sont sélectionnées automatiquement
4. **Cohérence** : Les mêmes transformations doivent être appliquées aux nouvelles données pour les prédictions
5. **Valeurs manquantes** : Les valeurs manquantes sont remplacées par 0 pour les variables numériques

---

## Références

- Fichier de configuration : `src/main.py` (fonction `select_features`)
- Résultats d'importance : `data/criteria_analysis_results.json`
- Traduction des variables : `src/dashboard.py` (dictionnaire `CRITERIA_TRANSLATION`)
