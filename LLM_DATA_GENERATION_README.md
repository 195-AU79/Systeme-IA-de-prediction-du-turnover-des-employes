# Génération de Données avec LLM

Ce projet utilise désormais une LLM (Language Model) pour générer des données d'employés plus réalistes pour la base de données.

## 🚀 Configuration

### 1. Installation des dépendances

```bash
pip install -r requirements.txt
```

Le package `openai` sera installé automatiquement.

### 2. Configuration de la clé API OpenAI

**Option A : Variable d'environnement (recommandée)**

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="votre_cle_api_openai"

# Windows CMD
set OPENAI_API_KEY=votre_cle_api_openai

# Linux/Mac
export OPENAI_API_KEY=votre_cle_api_openai
```

**Option B : Fichier .env (optionnel)**

Créez un fichier `.env` à la racine du projet :
```
OPENAI_API_KEY=votre_cle_api_openai
```

### 3. Configuration dans config.yaml

La configuration LLM se trouve dans `config.yaml` :

```yaml
llm_data_generation:
  enabled: true                    # Active/désactive la génération LLM
  provider: "openai"              # Fournisseur LLM
  model: "gpt-4o-mini"           # Modèle à utiliser (gpt-4o-mini est économique)
  api_key: "${OPENAI_API_KEY}"   # Variable d'environnement
  temperature: 0.7                # Créativité (0-1)
  max_tokens: 500                  # Tokens max par réponse
  batch_size: 20                  # Profils générés par requête
```

**Modèles recommandés :**
- `gpt-4o-mini` : Économique et rapide (recommandé)
- `gpt-4o` : Plus performant mais plus cher
- `gpt-3.5-turbo` : Alternative économique

## 📊 Utilisation

### Générer la base de données avec LLM

```bash
python create_database.py
```

Le script va :
1. Vérifier si la LLM est activée et disponible
2. Générer les profils d'employés par batches (20 par défaut)
3. Utiliser la génération aléatoire en fallback si la LLM échoue
4. Créer la base de données avec les données générées

### Exemple de sortie

```
============================================================
CREATION DE LA BASE DE DONNEES TURNOVER
============================================================

1. Creation des tables...
OK Tables creees

2. Generation des donnees d'employes...
✓ LLM activé: gpt-4o-mini
Génération de 1000 profils avec LLM (50 batches)...
  Batch 1/50 (20 profils)... ✓ 20 profils générés
  Batch 2/50 (20 profils)... ✓ 20 profils générés
  ...
OK 1000 employes generes
```

## 🎯 Avantages de la génération LLM

1. **Données plus réalistes** : 
   - Noms variés et cohérents
   - Âges adaptés aux niveaux de poste
   - Distributions naturelles

2. **Cohérence** :
   - Corrélations réalistes entre variables
   - Profils crédibles

3. **Variété** :
   - Nombreux noms français différents
   - Diversité dans les profils

## ⚙️ Désactiver la génération LLM

Si vous souhaitez utiliser uniquement la génération aléatoire (rapide, gratuit) :

```yaml
llm_data_generation:
  enabled: false
```

Ou commentez la ligne dans `config.yaml`.

## 💰 Coûts estimés

Pour 1000 employés avec `gpt-4o-mini` :
- ~50 appels API (20 profils par batch)
- Coût estimé : **~0.10-0.20 USD** (selon OpenAI)

Pour réduire les coûts :
- Augmentez `batch_size` (ex: 30-50) dans `config.yaml`
- Utilisez `gpt-3.5-turbo` si disponible

## 🔧 Dépannage

### Erreur : "Clé API OpenAI non trouvée"

**Solution** : Vérifiez que la variable d'environnement est bien définie :
```bash
echo $OPENAI_API_KEY  # Linux/Mac
echo %OPENAI_API_KEY%  # Windows CMD
$env:OPENAI_API_KEY    # Windows PowerShell
```

### Erreur : "Bibliothèque openai non disponible"

**Solution** :
```bash
pip install openai>=1.0.0
```

### La génération est lente

C'est normal ! La LLM prend du temps. Pour accélérer :
- Réduisez `batch_size` (mais augmente le nombre d'appels)
- Ou désactivez la LLM avec `enabled: false`

### Certains profils échouent

Le script passe automatiquement en mode fallback (génération aléatoire) pour les batches qui échouent. C'est normal et garanti de fonctionner.

## 📝 Notes

- Les données générées sont **anonymes** et **fictives**
- La LLM respecte les contraintes (départements, niveaux, etc.)
- Le calcul des salaires reste automatique et cohérent
- Compatible avec l'ancien système (fallback automatique)


