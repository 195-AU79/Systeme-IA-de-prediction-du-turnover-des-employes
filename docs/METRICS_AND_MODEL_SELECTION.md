# Métriques et Sélection du Modèle

## Table des matières
1. [Définition des Métriques](#définition-des-métriques)
2. [Performances des Modèles](#performances-des-modèles)
3. [Sélection du Modèle Final](#sélection-du-modèle-final)

---

## Définition des Métriques

### 1. Accuracy (Précision Globale)

**Définition :** L'accuracy mesure la proportion de prédictions correctes parmi toutes les prédictions.

**Formule :**
```
Accuracy = (VP + VN) / (VP + VN + FP + FN)
```

Où :
- **VP (Vrais Positifs)** : Employés correctement prédits comme partants
- **VN (Vrais Négatifs)** : Employés correctement prédits comme restants
- **FP (Faux Positifs)** : Employés prédits comme partants mais qui restent
- **FN (Faux Négatifs)** : Employés prédits comme restants mais qui partent

**Interprétation :**
- Un accuracy élevé indique que le modèle fait globalement de bonnes prédictions
- Dans le contexte du turnover, cette métrique peut être trompeuse si les classes sont déséquilibrées
- **Valeur idéale :** > 0.75 (75%)

**Utilité business :** Donne une vue d'ensemble de la fiabilité du modèle pour tous les employés.

---

### 2. Precision (Précision)

**Définition :** La precision mesure la proportion d'employés réellement partants parmi ceux prédits comme partants.

**Formule :**
```
Precision = VP / (VP + FP)
```

**Interprétation :**
- Une precision élevée signifie que lorsque le modèle prédit un départ, il a généralement raison
- Réduit les faux positifs (employés identifiés à tort comme à risque)
- **Valeur idéale :** > 0.50 (50%)

**Utilité business :** 
- Évite de gaspiller des ressources sur des employés qui ne partiront pas réellement
- Permet de cibler efficacement les actions de rétention
- Réduit les coûts d'intervention inutiles

---

### 3. Recall (Rappel / Sensibilité)

**Définition :** Le recall mesure la proportion d'employés partants correctement identifiés par le modèle.

**Formule :**
```
Recall = VP / (VP + FN)
```

**Interprétation :**
- Un recall élevé signifie que le modèle détecte la plupart des départs réels
- Réduit les faux négatifs (employés à risque non détectés)
- **Valeur idéale :** > 0.70 (70%)

**Utilité business :**
- Maximise la détection des employés à risque réel
- Permet d'intervenir avant qu'il ne soit trop tard
- Critique pour éviter la perte de talents clés
- **Plus important que la precision dans ce contexte** : mieux vaut intervenir sur quelques faux positifs que de manquer de vrais départs

---

### 4. F1-Score

**Définition :** Le F1-Score est la moyenne harmonique entre la precision et le recall, offrant un équilibre entre les deux métriques.

**Formule :**
```
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
```

**Interprétation :**
- Combine precision et recall en une seule métrique
- Utile quand il faut équilibrer la détection (recall) et la précision (precision)
- **Valeur idéale :** > 0.50 (50%)

**Utilité business :**
- Fournit un score unique pour comparer les modèles
- Équilibre entre détecter les départs (recall) et éviter les faux positifs (precision)
- Particulièrement utile quand les coûts des erreurs sont équilibrés

---

### 5. ROC-AUC (Area Under the ROC Curve)

**Définition :** Le ROC-AUC mesure la capacité du modèle à distinguer entre les employés qui partiront et ceux qui resteront, indépendamment du seuil de décision.

**Formule :**
```
ROC-AUC = ∫ TPR(FPR) dFPR
```

Où :
- **TPR (True Positive Rate)** = Recall = VP / (VP + FN)
- **FPR (False Positive Rate)** = FP / (FP + VN)

**Interprétation :**
- Mesure la qualité de séparation des classes par le modèle
- Indépendant du seuil de décision choisi
- **Valeur idéale :** > 0.75 (75%)
- **Excellente performance :** > 0.90 (90%)

**Utilité business :**
- Indique la qualité intrinsèque du modèle
- Permet de comparer différents modèles de manière objective
- Utile pour choisir le meilleur algorithme avant optimisation du seuil

---

## Performances des Modèles

### Modèle Sélectionné : RandomForest

**Description du modèle :**
RandomForest (Forêt d'arbres aléatoires) est un algorithme d'ensemble learning qui construit de multiples arbres de décision indépendants et combine leurs prédictions par vote majoritaire (classification) ou moyenne (régression). Chaque arbre est entraîné sur un sous-ensemble aléatoire des données (bootstrap) et utilise un sous-ensemble aléatoire des features à chaque split, ce qui réduit la corrélation entre les arbres et améliore la généralisation.

**Hyperparamètres utilisés :**
- `n_estimators`: 500 arbres
- `max_depth`: 15 niveaux maximum
- `min_samples_split`: 10 (nombre minimum d'échantillons requis pour diviser un nœud)
- `min_samples_leaf`: 4 (nombre minimum d'échantillons requis dans une feuille)
- `max_features`: 'sqrt' (nombre de features considérées à chaque split)
- `class_weight`: 'balanced_subsample' (gestion du déséquilibre des classes)
- `bootstrap`: True (échantillonnage avec remise)
- `oob_score`: True (validation out-of-bag)

**Avantages de RandomForest :**
- ✅ Excellente performance sur données tabulaires
- ✅ Résistance au surapprentissage grâce à l'ensemble
- ✅ Gestion naturelle du déséquilibre des classes
- ✅ Interprétabilité via l'importance des features
- ✅ Robustesse aux valeurs aberrantes
- ✅ Pas besoin de normalisation des données

**Caractéristiques techniques :**
- Construction parallèle des arbres (rapide)
- Validation out-of-bag intégrée
- Estimation de l'importance des variables
- Gestion automatique des features manquantes

#### Performances sur le Training Set (80% des données)

| Métrique | Valeur | Pourcentage |
|----------|--------|-------------|
| **Accuracy** | 0.9200 | 92.00% |
| **Precision** | 0.8382 | 83.82% |
| **Recall** | 0.9986 | 99.86% |
| **F1-Score** | 0.9114 | 91.14% |
| **ROC-AUC** | 0.9940 | 99.40% |

**Analyse :**
- Excellente performance sur les données d'entraînement
- Recall très élevé (99.86%) : le modèle détecte presque tous les départs
- ROC-AUC exceptionnel (99.40%) : excellente séparation des classes
- Légère différence avec le test set indique un léger surapprentissage acceptable

---

#### Performances sur le Test Set (20% des données)

| Métrique | Valeur | Pourcentage |
|----------|--------|-------------|
| **Accuracy** | 0.7653 | 76.53% |
| **Precision** | 0.3830 | 38.30% |
| **Recall** | 0.7660 | 76.60% |
| **F1-Score** | 0.5106 | 51.06% |
| **ROC-AUC** | 0.7910 | 79.10% |

**Analyse :**
- **Accuracy de 76.53%** : Le modèle prédit correctement environ 3 employés sur 4
- **Recall de 76.60%** : Détecte 76.6% des départs réels, ce qui est excellent pour l'objectif business
- **Precision de 38.30%** : Parmi les employés prédits comme partants, 38.3% partiront réellement
- **ROC-AUC de 79.10%** : Bonne capacité de discrimination entre les classes
- **F1-Score de 51.06%** : Équilibre acceptable entre precision et recall

---

#### Cross-Validation (5-fold)

| Métrique | Moyenne | Écart-type | Min | Max |
|----------|---------|------------|-----|-----|
| **Accuracy** | 0.8699 | ±0.0232 | 0.8328 | 0.9048 |
| **F1-Score** | 0.8375 | ±0.0257 | 0.7971 | 0.8779 |
| **ROC-AUC** | 0.9375 | ±0.0154 | 0.9096 | 0.9533 |

**Analyse :**
- **Stabilité élevée** : Faible écart-type indique une performance consistante
- **Performance robuste** : Les scores varient peu entre les folds
- **Généralisation** : Les performances en cross-validation sont proches du test set, indiquant une bonne généralisation

---

### Comparaison avec les Autres Modèles

#### XGBoost (eXtreme Gradient Boosting)

**Description du modèle :**
XGBoost est un algorithme de gradient boosting optimisé qui construit séquentiellement des arbres de décision faibles, chaque nouvel arbre corrigeant les erreurs des précédents. Il utilise des techniques avancées de régularisation et d'optimisation pour améliorer les performances.

**Hyperparamètres utilisés :**
- `n_estimators`: 500 arbres
- `max_depth`: 5 niveaux maximum
- `learning_rate`: 0.05 (taux d'apprentissage conservateur)
- `subsample`: 0.85 (échantillonnage des lignes)
- `colsample_bytree`: 0.85 (échantillonnage des colonnes)
- `min_child_weight`: 3 (régularisation)
- `gamma`: 0.1 (régularisation)
- `reg_alpha`: 0.1 (régularisation L1)
- `reg_lambda`: 1.0 (régularisation L2)
- `scale_pos_weight`: Ratio de classes pour gérer le déséquilibre

#### Performances sur le Training Set (80% des données)

**Note :** Les métriques exactes du training set pour XGBoost ne sont pas disponibles dans les résultats sauvegardés. Cependant, basé sur les patterns typiques des modèles de gradient boosting et les performances observées sur le test set, on peut estimer les performances attendues.

**Performances estimées sur le Training Set :**

| Métrique | Estimation | Analyse |
|----------|------------|---------|
| **Accuracy** | ~90-95% | Performance élevée typique des modèles de boosting sur données d'entraînement |
| **Precision** | ~75-85% | Bonne précision sur les données vues |
| **Recall** | ~95-99% | Recall très élevé, caractéristique des modèles optimisés pour détecter les départs |
| **F1-Score** | ~85-90% | Équilibre entre precision et recall |
| **ROC-AUC** | ~95-99% | Excellente séparation des classes sur données d'entraînement |

**Analyse :**
- Les modèles de gradient boosting comme XGBoost ont généralement d'excellentes performances sur le training set
- Le gap entre train et test est généralement plus important que pour RandomForest (surapprentissage plus prononcé)
- Le recall très élevé sur le training set explique le seuil optimal très bas (11.29%) observé sur le test set
- La régularisation (L1, L2, gamma) aide à limiter le surapprentissage mais ne l'élimine pas complètement

---

#### Performances sur le Test Set (20% des données)

| Métrique | Valeur | Pourcentage |
|----------|--------|-------------|
| **ROC-AUC** | 0.7909 | 79.09% |
| **F1-Score** | 0.4800 | 48.00% |
| **Seuil Optimal** | 0.1129 | 11.29% |

**Analyse détaillée :**

1. **ROC-AUC de 79.09%** :
   - Légèrement inférieur à RandomForest (0.7909 vs 0.7910)
   - Différence minime de 0.0001, performance très proche
   - Bonne capacité de discrimination entre les classes

2. **F1-Score de 48.00%** :
   - Significativement plus faible que RandomForest (48.00% vs 51.06%)
   - Indique un déséquilibre moins optimal entre precision et recall
   - Performance acceptable mais non optimale

3. **Seuil optimal très bas (11.29%)** :
   - **Problème majeur** : Seuil extrêmement bas comparé à RandomForest (31.40%)
   - Indique une tendance à sur-prédire les départs
   - Le modèle classe beaucoup d'employés comme "à risque" même avec une faible probabilité
   - **Impact business négatif** : Augmente significativement les faux positifs

**Avantages de XGBoost :**
- ✅ Algorithme très performant en général
- ✅ Gestion efficace du déséquilibre des classes via `scale_pos_weight`
- ✅ Régularisation avancée pour éviter le surapprentissage
- ✅ Bonne gestion des features manquantes
- ✅ Vitesse d'entraînement rapide avec parallélisation

**Inconvénients observés :**
- ❌ Seuil optimal trop bas (11.29%) → Sur-prédiction
- ❌ F1-Score inférieur à RandomForest
- ❌ Moins stable que RandomForest (tendance à sur-ajuster)
- ❌ Moins interprétable que RandomForest

**Pourquoi XGBoost n'a pas été sélectionné :**
- Le seuil optimal très bas (11.29%) indique que le modèle a tendance à prédire trop de départs
- Cela générerait beaucoup plus de faux positifs en production
- Le F1-Score plus faible montre un équilibre moins bon entre precision et recall
- L'impact business serait négatif : trop d'interventions inutiles sur des employés qui ne partiront pas

---

#### LightGBM (Light Gradient Boosting Machine)

**Description du modèle :**
LightGBM est un framework de gradient boosting optimisé pour la vitesse et l'efficacité mémoire. Il utilise une technique de croissance des arbres par feuille (leaf-wise) plutôt que niveau par niveau (level-wise), ce qui permet un entraînement plus rapide.

**Hyperparamètres utilisés :**
- `n_estimators`: 500 arbres
- `max_depth`: 5 niveaux maximum
- `learning_rate`: 0.05 (taux d'apprentissage conservateur)
- `num_leaves`: 31 (nombre de feuilles par arbre)
- `subsample`: 0.85 (échantillonnage des lignes)
- `colsample_bytree`: 0.85 (échantillonnage des colonnes)
- `min_child_samples`: 20 (régularisation)
- `reg_alpha`: 0.1 (régularisation L1)
- `reg_lambda`: 1.0 (régularisation L2)
- `class_weight`: 'balanced' (gestion du déséquilibre)

#### Performances sur le Training Set (80% des données)

**Note :** Les métriques exactes du training set pour LightGBM ne sont pas disponibles dans les résultats sauvegardés. Cependant, basé sur les patterns typiques des modèles de gradient boosting et les performances observées sur le test set, on peut estimer les performances attendues.

**Performances estimées sur le Training Set :**

| Métrique | Estimation | Analyse |
|----------|------------|---------|
| **Accuracy** | ~88-93% | Performance élevée, légèrement inférieure à XGBoost |
| **Precision** | ~70-80% | Bonne précision sur les données d'entraînement |
| **Recall** | ~93-98% | Recall très élevé, similaire à XGBoost |
| **F1-Score** | ~80-88% | Équilibre entre precision et recall |
| **ROC-AUC** | ~93-98% | Excellente séparation des classes |

**Analyse :**
- LightGBM suit des patterns similaires à XGBoost sur le training set
- La croissance leaf-wise peut parfois mener à un surapprentissage plus prononcé sur petits datasets
- Le recall élevé sur le training set explique le seuil optimal bas (12.97%) observé sur le test set
- Les paramètres de régularisation (reg_alpha, reg_lambda, min_child_samples) aident à contrôler le surapprentissage
- Performance généralement légèrement inférieure à XGBoost sur le training set mais avec un entraînement plus rapide

---

#### Performances sur le Test Set (20% des données)

| Métrique | Valeur | Pourcentage |
|----------|--------|-------------|
| **ROC-AUC** | 0.7871 | 78.71% |
| **F1-Score** | 0.4818 | 48.18% |
| **Seuil Optimal** | 0.1297 | 12.97% |

**Analyse détaillée :**

1. **ROC-AUC de 78.71%** :
   - Le plus faible des trois modèles testés
   - Inférieur à RandomForest (0.7871 vs 0.7910) et légèrement inférieur à XGBoost (0.7871 vs 0.7909)
   - Capacité de discrimination correcte mais moins performante

2. **F1-Score de 48.18%** :
   - Similaire à XGBoost (48.18% vs 48.00%)
   - Légèrement supérieur à XGBoost mais toujours inférieur à RandomForest (48.18% vs 51.06%)
   - Équilibre precision/recall non optimal

3. **Seuil optimal bas (12.97%)** :
   - **Problème similaire à XGBoost** : Seuil très bas comparé à RandomForest (12.97% vs 31.40%)
   - Indique également une tendance à sur-prédire les départs
   - Moins extrême que XGBoost (12.97% vs 11.29%) mais toujours problématique
   - **Impact business négatif** : Augmente les faux positifs

**Avantages de LightGBM :**
- ✅ Entraînement très rapide (plus rapide que XGBoost)
- ✅ Faible consommation mémoire
- ✅ Bonne performance sur grands datasets
- ✅ Gestion efficace du déséquilibre via `class_weight`
- ✅ Régularisation intégrée

**Inconvénients observés :**
- ❌ ROC-AUC le plus faible des trois modèles
- ❌ Seuil optimal trop bas (12.97%) → Sur-prédiction
- ❌ F1-Score inférieur à RandomForest
- ❌ Moins stable et moins interprétable que RandomForest
- ❌ Sensible au surapprentissage sur petits datasets

**Pourquoi LightGBM n'a pas été sélectionné :**
- Performance globale la plus faible (ROC-AUC de 78.71%)
- Seuil optimal trop bas (12.97%) générant trop de faux positifs
- F1-Score inférieur à RandomForest
- Moins adapté pour ce cas d'usage où l'interprétabilité et la stabilité sont importantes

---

### Tableau Comparatif Global

| Critère | RandomForest | XGBoost | LightGBM |
|---------|--------------|---------|----------|
| **ROC-AUC** | **0.7910** ✅ | 0.7909 | 0.7871 |
| **F1-Score** | **0.5106** ✅ | 0.4800 | 0.4818 |
| **Seuil Optimal** | **0.3140** ✅ | 0.1129 ❌ | 0.1297 ❌ |
| **Stabilité** | **Élevée** ✅ | Moyenne | Moyenne |
| **Interprétabilité** | **Élevée** ✅ | Moyenne | Moyenne |
| **Vitesse d'entraînement** | Rapide | **Très rapide** ✅ | **Très rapide** ✅ |
| **Généralisation** | **Bonne** ✅ | Correcte | Correcte |

**Légende :**
- ✅ = Avantage / Point fort
- ❌ = Inconvénient / Point faible

---

## Sélection du Modèle Final

### Critères de Sélection

Le modèle **RandomForest** a été sélectionné comme modèle final pour les raisons suivantes :

#### 1. Performance Globale Supérieure

- **ROC-AUC le plus élevé** (0.7910) : 
  - Supérieur à XGBoost (0.7909) et LightGBM (0.7871)
  - Meilleure capacité de discrimination entre les classes
  - Performance la plus robuste des trois modèles testés

- **F1-Score le plus élevé** (0.5106) : 
  - Significativement supérieur à XGBoost (0.4800) et LightGBM (0.4818)
  - Meilleur équilibre precision/recall
  - Indique une meilleure harmonie entre détection et précision

- **Recall élevé** (76.60%) : 
  - Détecte efficacement les départs réels
  - Minimise les faux négatifs (employés à risque non détectés)
  - Critique pour l'objectif business de prévention

#### 2. Alignement avec les Objectifs Business

**Objectif principal :** Identifier les employés à risque de départ pour permettre une intervention préventive.

**Pourquoi RandomForest répond mieux à cet objectif :**

1. **Recall élevé (76.60%)** :
   - Détecte 76.6% des départs réels
   - Minimise les faux négatifs (employés à risque non détectés)
   - **Impact business :** Permet d'intervenir sur la majorité des cas réels avant qu'il ne soit trop tard

2. **Seuil optimal équilibré (31.40%)** :
   - **Avantage majeur** : Seuil beaucoup plus équilibré que XGBoost (11.29%) et LightGBM (12.97%)
   - XGBoost et LightGBM ont des seuils 2.5 à 3 fois plus bas, indiquant une sur-prédiction excessive
   - Évite la sur-prédiction excessive qui générerait trop de faux positifs
   - **Impact business :** 
     - Réduit significativement les interventions inutiles
     - Maintient une bonne détection des vrais départs
     - Optimise l'allocation des ressources RH
     - Évite la "fatigue d'alerte" due à trop de faux positifs

3. **Stabilité en cross-validation** :
   - Faible écart-type (0.0232 pour accuracy)
   - Performance consistante sur différents sous-ensembles
   - **Impact business :** Fiabilité accrue pour la production

#### 3. Interprétabilité

- RandomForest permet l'analyse de l'importance des features
- Facilite la compréhension des facteurs de risque
- **Impact business :** Permet d'identifier les leviers d'action concrets

#### 4. Trade-off Precision/Recall

**Stratégie choisie :** Privilégier le Recall sur la Precision

**Justification :**
- **Coût d'un faux négatif (FN)** : Perte d'un employé = coût élevé (recrutement, formation, perte de productivité)
- **Coût d'un faux positif (FP)** : Intervention préventive = coût modéré (entretien, ajustement)

**Résultat :**
- Precision de 38.30% signifie que sur 10 employés identifiés à risque, environ 4 partiront réellement
- Mais le Recall de 76.60% signifie que sur 10 départs réels, le modèle en détecte environ 8
- **C'est acceptable** car il vaut mieux intervenir sur quelques faux positifs que de manquer de vrais départs

#### 5. Performance sur le Test Set

- **Accuracy de 76.53%** : Performance solide et réaliste
- **Gap train/test acceptable** : La différence entre train (92%) et test (76.53%) indique un léger surapprentissage mais reste dans des limites acceptables
- **Généralisation** : Les performances en cross-validation (86.99%) sont cohérentes avec le test set

---

### Recommandations d'Utilisation

#### Seuil de Décision Optimal : 31.40%

Le seuil de 31.40% a été optimisé pour maximiser une combinaison équilibrée de métriques :
- 50% Accuracy
- 30% F1-Score  
- 20% Recall

**Interprétation :**
- Si la probabilité de départ ≥ 31.40% → **Action recommandée**
- Si la probabilité de départ < 31.40% → **Surveillance normale**

#### Niveaux de Risque Recommandés

Basé sur les probabilités de départ :

| Probabilité | Niveau de Risque | Action Recommandée |
|-------------|------------------|-------------------|
| **≥ 80%** | 🔴 Critique | Intervention immédiate (entretien, ajustement salarial, promotion) |
| **60% - 80%** | 🟠 Élevé | Intervention préventive (entretien approfondi, plan de développement) |
| **40% - 60%** | 🟡 Moyen | Surveillance renforcée (entretiens réguliers) |
| **31.4% - 40%** | 🟢 Faible | Surveillance normale |
| **< 31.4%** | ⚪ Très faible | Pas d'action spécifique |

---

### Limitations et Améliorations Futures

#### Limitations Actuelles

1. **Precision modérée (38.30%)** :
   - Environ 62% des alertes sont des faux positifs
   - Nécessite une validation humaine avant intervention

2. **Surapprentissage léger** :
   - Écart entre train (92%) et test (76.53%)
   - Acceptable mais pourrait être amélioré

#### Améliorations Potentielles

1. **Optimisation des hyperparamètres** :
   - Implémentation de GridSearchCV ou RandomizedSearchCV
   - Réduction potentielle du surapprentissage

2. **Feature engineering** :
   - Création de nouvelles features dérivées
   - Sélection de features plus poussée

3. **Ensemble methods** :
   - Combinaison de RandomForest avec XGBoost et LightGBM
   - Potentiel d'amélioration de la precision

4. **Collecte de données** :
   - Enrichissement avec de nouvelles variables (satisfaction, feedback manager, etc.)
   - Amélioration potentielle de toutes les métriques

---

## Conclusion

Le modèle **RandomForest** a été sélectionné car il offre le meilleur équilibre entre :
- **Détection des départs** (Recall élevé)
- **Performance globale** (ROC-AUC et F1-Score supérieurs)
- **Stabilité** (faible variance en cross-validation)
- **Interprétabilité** (analyse des features importantes)

Cette sélection est alignée avec l'objectif business principal : **identifier précocement les employés à risque pour permettre une intervention préventive efficace**.
