# Système d'IA de Prédiction du Turnover des Employés

Un système complet d'intelligence artificielle pour prédire le risque de départ (turnover) des employés, avec analyse des critères clés, dashboard interactif et recommandations d'actions de rétention.

## 📋 Table des matières

- [Description](#description)
- [Fonctionnalités](#fonctionnalités)
- [Installation](#installation)
- [Structure du projet](#structure-du-projet)
- [Utilisation](#utilisation)
- [Configuration](#configuration)
- [Documentation](#documentation)
- [Modèles et Métriques](#modèles-et-métriques)
- [Technologies utilisées](#technologies-utilisées)

## 🎯 Description

Ce système utilise des algorithmes de machine learning avancés (Random Forest, XGBoost, LightGBM) pour analyser les données RH et prédire la probabilité qu'un employé quitte l'entreprise. Il identifie les facteurs de risque clés et propose des actions de rétention personnalisées.

### Critères analysés

- **Salaire et compensation** : Comparaison avec le marché, compétitivité salariale
- **Ancienneté** : Facteurs de risque selon l'ancienneté
- **Performance** : Notes, tendances, atteinte des objectifs
- **Formation et développement** : Intensité, qualité, fréquence
- **Charge de travail** : Heures supplémentaires, projets, indicateurs de charge
- **Satisfaction** : Satisfaction au travail, environnement, relations
- **Relation manager** : Qualité de la relation avec le manager
- **Équilibre vie/travail** : Indicateurs de stress et d'équilibre
- **Carrière** : Progression, promotions, stagnation
- **Concurrence** : Attractivité des offres concurrentes
- **Image de l'entreprise** : Perception de l'entreprise

## ✨ Fonctionnalités

- 🤖 **Prédiction du turnover** : Modèles ML entraînés avec plusieurs algorithmes
- 📊 **Dashboard interactif** : Interface Streamlit pour visualiser les données et prédictions
- 🔍 **Analyse d'importance** : Identification des facteurs les plus influents
- 📈 **Analyse SHAP** : Interprétabilité des prédictions
- 👥 **Analyse d'employés** : Analyse individuelle avec recommandations
- 📉 **Monitoring** : Suivi des métriques et alertes
- 🔄 **Apprentissage continu** : Mise à jour automatique des modèles
- 🔒 **Confidentialité** : Conformité GDPR avec anonymisation des données

## 🚀 Installation

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

### Installation des dépendances

1. Clonez le dépôt :
```bash
git clone <url-du-repo>
cd Systeme-IA-de-prediction-du-turnover-des-employes-main
```

2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

### Configuration de la base de données

1. Importez les données CSV dans la base de données :
```bash
python scripts/import_csv_to_database.py
```

2. Vérifiez que la base de données a été créée :
```bash
python scripts/database_reader.py
```

## 📁 Structure du projet

```
Systeme-IA-de-prediction-du-turnover-des-employes-main/
├── src/                    # Code principal de l'application
│   ├── main.py            # Script d'entraînement du modèle principal
│   ├── dashboard.py       # Dashboard Streamlit interactif
│   ├── start.py           # Point d'entrée principal
│   └── paths.py           # Helpers pour les chemins de fichiers
│
├── scripts/               # Scripts utilitaires
│   ├── employee_analyzer_simple.py    # Analyse d'employés spécifiques
│   ├── check_importance.py           # Vérification de l'importance des features
│   ├── calculate_turnover_percentage.py  # Calcul du taux de turnover
│   ├── create_database.py             # Création de la base de données
│   ├── database_reader.py             # Lecture de la base de données
│   ├── import_csv_to_database.py      # Import CSV vers base de données
│   ├── continuous_learning.py         # Apprentissage continu
│   ├── monitoring.py                  # Monitoring du système
│   ├── privacy_preserving.py         # Fonctions de confidentialité
│   └── paths.py                       # Helpers pour les chemins
│
├── models/                # Modèles ML sauvegardés (.pkl)
│   ├── turnover_criteria_model.pkl
│   ├── criteria_scaler.pkl
│   ├── criteria_encoder_*.pkl
│   └── ...
│
├── data/                  # Données et résultats
│   ├── turnover_data.db   # Base de données SQLite
│   ├── *.json             # Résultats d'analyse
│   └── archive/           # Fichiers CSV archivés
│
├── config/                # Fichiers de configuration
│   └── config.yaml
│
├── docs/                  # Documentation
│   ├── MODEL_VARIABLES.md
│   └── METRICS_AND_MODEL_SELECTION.md
│
├── requirements.txt       # Dépendances Python
└── README.md             # Ce fichier
```

## 💻 Utilisation

### Démarrage rapide

Lancez le script de démarrage interactif :

```bash
python src/start.py
```

Vous aurez le choix entre :
1. **Entraîner le modèle principal** : Entraîne le modèle avec les données disponibles
2. **Analyser des employés spécifiques** : Analyse des employés individuels
3. **Démarrer le dashboard** : Lance l'interface web Streamlit
4. **Quitter**

### Entraîner le modèle

Pour entraîner le modèle de prédiction :

```bash
python src/main.py
```

Ce script va :
- Charger et préparer les données depuis la base de données
- Sélectionner les features les plus importantes
- Entraîner plusieurs modèles (Random Forest, XGBoost, LightGBM)
- Sélectionner le meilleur modèle ou créer un ensemble
- Analyser l'importance des features
- Générer l'analyse SHAP
- Sauvegarder le modèle et les résultats

### Lancer le dashboard

Pour accéder au dashboard interactif :

```bash
streamlit run src/dashboard.py
```

Le dashboard sera accessible sur `http://localhost:8501`

### Analyser un employé spécifique

Pour analyser un employé individuel :

```bash
python scripts/employee_analyzer_simple.py
```

## ⚙️ Configuration

Le fichier `config/config.yaml` contient toutes les configurations du système :

- **Sources de données** : Configuration des systèmes HR (SAP, Workday, etc.)
- **Paramètres des modèles** : Hyperparamètres pour chaque algorithme
- **Seuils de risque** : Définition des niveaux de risque (critique, élevé, moyen, faible)
- **Features** : Liste des variables utilisées par catégorie
- **Actions de rétention** : Recommandations par niveau de risque
- **Monitoring** : Configuration des alertes et KPIs
- **Éthique et conformité** : Paramètres GDPR et biais

Consultez `config/config.yaml` pour personnaliser ces paramètres.

## 📚 Documentation

### Documentation détaillée

- **[MODEL_VARIABLES.md](docs/MODEL_VARIABLES.md)** : Description complète des variables utilisées par le modèle
- **[METRICS_AND_MODEL_SELECTION.md](docs/METRICS_AND_MODEL_SELECTION.md)** : Explication des métriques et de la sélection des modèles
- **[REORGANIZATION.md](REORGANIZATION.md)** : Documentation de la réorganisation du code

### Variables du modèle

Le modèle utilise **30 variables** sélectionnées automatiquement parmi un ensemble initial plus large. Ces variables sont sélectionnées à l'aide de `SelectKBest` avec `mutual_info_classif` pour capturer les relations non-linéaires.

Les principales catégories de variables incluent :
- Variables démographiques (âge, ancienneté, département, niveau hiérarchique)
- Performance (notes, tendances, objectifs)
- Compensation (salaire, augmentations, promotions)
- Satisfaction (travail, environnement, relations)
- Charge de travail (heures supplémentaires, projets)
- Formation et développement

## 📊 Modèles et Métriques

### Algorithmes utilisés

Le système teste plusieurs algorithmes et sélectionne le meilleur :

- **Random Forest** : Ensemble d'arbres de décision
- **XGBoost** : Gradient boosting optimisé
- **LightGBM** : Gradient boosting rapide et efficace
- **Ensemble (Voting)** : Combinaison des meilleurs modèles

### Métriques de performance

Le modèle est évalué sur plusieurs métriques :

- **Accuracy** : Précision globale
- **ROC-AUC** : Aire sous la courbe ROC
- **Precision** : Précision des prédictions positives
- **Recall** : Taux de détection des départs
- **F1-Score** : Moyenne harmonique de précision et recall

### Seuil optimal

Le système trouve automatiquement le seuil optimal de décision pour équilibrer accuracy, F1-score et recall.

## 🛠️ Technologies utilisées

### Core Data Science
- **pandas** : Manipulation de données
- **numpy** : Calculs numériques
- **scipy** : Statistiques et optimisation

### Machine Learning
- **scikit-learn** : Algorithmes ML classiques
- **xgboost** : Gradient boosting optimisé
- **lightgbm** : Gradient boosting rapide
- **imbalanced-learn** : Gestion des classes déséquilibrées (SMOTE)

### Interprétabilité
- **shap** : Analyse SHAP pour l'explicabilité

### Visualisation
- **matplotlib** : Graphiques statiques
- **seaborn** : Visualisations statistiques
- **plotly** : Graphiques interactifs

### Dashboard
- **streamlit** : Interface web interactive

### Utilitaires
- **joblib** : Sauvegarde/chargement de modèles
- **pyyaml** : Gestion de configuration
- **sqlite3** : Base de données

## 📝 Notes importantes

### Données

- Les données doivent être importées dans la base de données SQLite avant l'entraînement
- Le format CSV attendu doit contenir les colonnes standardisées (voir `MODEL_VARIABLES.md`)

### Performance

- L'entraînement peut prendre plusieurs minutes selon la taille du dataset
- Les modèles sont sauvegardés automatiquement dans le dossier `models/`

### Confidentialité

- Le système inclut des fonctionnalités de préservation de la confidentialité
- Les données peuvent être anonymisées selon les besoins GDPR

## 🤝 Contribution

Pour contribuer au projet :

1. Forkez le dépôt
2. Créez une branche pour votre fonctionnalité (`git checkout -b feature/AmazingFeature`)
3. Committez vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Pushez vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

## 📄 Licence

Ce projet est fourni tel quel pour usage éducatif et professionnel.

## 👤 Auteur

Système développé pour l'analyse et la prédiction du turnover des employés.

## 🙏 Remerciements

- Bibliothèques open-source utilisées
- Communauté Python pour le support

---

**Note** : Ce système est un outil d'aide à la décision. Les prédictions doivent être utilisées en complément de l'expertise RH et ne doivent pas être le seul critère de décision.

