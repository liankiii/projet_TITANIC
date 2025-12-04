# Analyse et prédiction de la survie Titanic

## Résumé exécutif

Ce projet implémente un pipeline complet d'apprentissage supervisé pour prédire la survie des passagers du Titanic à partir de caractéristiques démographiques et socioéconomiques. Les modèles sont entraînés via validation croisée stratifiée (5 folds) et évalués sur un ensemble de test indépendant (20%), assurant une estimation fiable de la performance de généralisation.

**Performance finale (test set):**
- Gradient Boosting: accuracy=0.843, F1=0.786
- Random Forest: accuracy=0.834, F1=0.776
- Logistic Regression: accuracy=0.806, F1=0.735

--------------------------------------------------------------------------------------------------------

## Architecture générale

```
projet_titanic/
├── main.py                     # Point d'entrée principal
├── config.py                   # Configuration centralisée
├── encoders.py                 # Gestion encodeurs/scalers
├── pipeline.py                 # Orchestration du pipeline
├── utils.py                    # Utilitaires réutilisables
├── core/
│   ├── train.py                # Entraînement et évaluation des modèles
│   ├── features.py             # Ingénierie des features
│   ├── predict.py              # Prédictions interactives
│   ├── visualize.py            # Visualisations analytiques
│   └── __pycache__/
├── data/
│   ├── data_loader.py          # Chargement des données depuis xls
│   └── Titanic-Dataset.xls     # Données source
├── outputs/                    # Dossier des graphiques générés
├── requirements.txt            # Dépendances
└── __pycache__/
```

--------------------------------------------------------------------------------------------------------

## Méthodologie

### 1. Préparation des données

#### 1.1 Ingénierie des features (`core/features.py`)

**Nouvelles variables créées:**

- **`age_group`**: Catégorisation discrète de l'âge (enfant, ado, jeune adulte, adulte, senior).
- **`is_child`, `is_mother`**: Variables binaires pour identifier les enfants et mères.
- **`family_size`**: Total du groupe familial (SibSp + Parch + 1).
- **`fare_per_person`**: Tarif normalisé par personne dans le groupe familial.
- **`title`**: civilité extrait du nom (Mr, Mrs, Miss, Master, Rare, Unknown) — proxy social et démographique puissant.
- **`deck`**: Première lettre du numéro de cabine (A-G, T, U) — proxy de localisation et classe.
- **`ticket_prefix`**: Préfixe du numéro de ticket — variable catégorique haute fréquence.

**Imputation de l'âge:**
L'âge manquant (19.9% des données) est imputé par la **médiane du groupe (Sex, Pclass)** avant une imputation globale. Cette stratégie préserve les patterns sociodémographiques observés (ex. âge médian différent entre hommes/femmes, classes).

**Profils de passagers:**
Assignation manuelle de 21 profils sociodémographiques (ex. "femme_riche_jeune", "jeune_homme_seul") basée sur âge, sexe, classe et contexte familial. Ces profils capturent les interactions complexes entre variables et facilitent l'interprétabilité.

#### 1.2 Sélection des features pour l'entraînement

```python
numeric_cols = ['Age', 'Fare', 'fare_per_person', 'Pclass', 'SibSp', 'Parch', 'family_size']
categorical_cols = ['Sex', 'Embarked', 'ticket_class', 'age_group', 'title', 'deck', 'ticket_prefix']
```

**Justification:**
- Variables numériques: capturent les dimensions de prix, démographie et classe.
- Variables catégorique: capturent le genre (forte corrélation avec survie), l'embarquement, le statut social via titre et pont.

#### 1.3 Normalisation et encodage

- **StandardScaler**: appliqué sur toutes les variables numériques et catégorique encodées pour ramener à une échelle [−σ, +σ] centrée en 0.
- **LabelEncoder**: encodage ordinal des catégories avec figeage des catégories connues pour éviter les dérives en prédiction.
- **PCA optionnelle**: réduction à 4 composantes principales pour le clustering (améliore la séparation via réduction du bruit).

### 2. Entraînement et validation

#### 2.1 Protocole de validation croisée

- **StratifiedKFold(n_splits=5, shuffle=True, random_state=42)**: scinde les données en 5 folds stratifiés.
- **Avantage**: préserve les proportions de classe (38.4% survie) dans chaque fold; moyennes les métriques pour une meilleure estimation .
- **Résultats CV** (moyennes sur 5 folds):
  - Gradient Boosting: acc=0.843, F1=0.786
  - Random Forest: acc=0.834, F1=0.776
  - Logistic Regression: acc=0.806, F1=0.735

#### 2.2 Split train/test

- **Ratio**: 80% entraînement, 20% test (stratifié sur `Survived`).
- **Taille**: ~713 samples train, ~178 samples test.
- **Approche**: les modèles finaux sont réentraînés sur l'ensemble train complet (post-CV) puis évalués sur test indépendant.

#### 2.3 Hyperparamètres des modèles

**Logistic Regression:**
```python
LogisticRegression(
    random_state=42,
    max_iter=2000,
    class_weight='balanced'  # Gère le déséquilibre de classes (38% survie)
)
```
Justification: `class_weight='balanced'` ajuste les poids de classes inversement proportionnels à leur fréquence, on neutralise le biais vers la classe majoritaire.

**Random Forest:**
```python
RandomForestClassifier(
    n_estimators=300,        # Nombre élevé pour meilleure stabilité
    max_depth=None,          # Pas de limite → capture patterns complexes
    min_samples_split=2,     # Bas → flexibilité
    min_samples_leaf=1,      # Bas → régression fine
    max_features='sqrt',     # sqrt(n_features) ≈ 3.7 → décorrélation entre arbres
    random_state=42,
    n_jobs=-1                # Parallélisation
)
```
Justification: configuration volontairement exploratrice (300 arbres, pas de limite de profondeur) pour capturer interactions "complexes"; `max_features='sqrt'` réduit la corrélation inter-arbres et améliore la généralisation.

**Gradient Boosting:**
```python
GradientBoostingClassifier(
    n_estimators=200,
    random_state=42
    # Autres paramètres par défaut: learning_rate=0.1, max_depth=3
)
```
Justification: 200 estimateurs sequentiels; depth=3 limite le sur-apprentissage (weak learners); learning_rate=0.1 par défaut = bon compromis.

### 3. Clustering K-Means avec PCA

#### 3.1 Approche

- **Algorithme**: K-Means (simple, efficace mais aussi sensible à l'initialisation).
- **Réduction dimensionnelle**: PCA avec 4 composantes avant clustering.
  - **Pourquoi PCA?** Réduction du bruit, amélioration de la séparation des clusters en espace de faible dimension.
  - **4 composantes**: expliquent ~70% de la variance; bon compromis exploitation/bruit mais assez faible en soit.

#### 3.2 Sélection du nombre de clusters

- **Grille**: k ∈ [2, 3, 4, 5, 6, 7, 8].
- **Métrique**: silhouette score (mesure de cohésion intra-cluster et séparation inter-cluster).
- **Résultat**: k=5, silhouette=0.371 (structure modérée/faible).
  - Interprétation: clusters pas tr-s bien séparés avec chevauchement; mais structure naturelle présente.

#### 3.3 Interprétation des clusters

| Cluster | Survivants | Total | Taux survie | Caractéristiques |
|---------|-----------|-------|-------------|------------------|
| 3       | 69        | 87    | 79.3%       | Riches, tarif élevé (~143£), classe 1 |
| 4       | 98        | 132   | 74.2%       | Jeunes, tarif bas, classe 2-3 mélangée |
| 1       | 112       | 249   | 45.0%       | Adultes, tarif moyen, classe 1-2 |
| 2       | 58        | 368   | 15.8%       | Classe populaire, tarif bas (~11£) |
| 0       | 5         | 55    | 9.1%        | Enfants seuls ou marginalisés |

**Observations**: clusters alignés avec facteurs socioéconomiques (tarif, classe); le cluster "riche" (3) a 79% survie vs 9% pour le cluster "marginal" (0) — écart de 70 points.

--------------------------------------------------------------------------------------------------------

## Résultats et analyse critique

### 1. Performance globale

**Test set (hold-out 20%):**
- GradientBoosting: acc=0.843, F1=0.786, prec=0.822, recall=0.754
- RandomForest: acc=0.834, F1=0.776, prec=0.807, recall=0.749
- LogisticRegression: acc=0.806, F1=0.735, prec=0.770, recall=0.705

### 3. Limitations et axes d'amélioration

1. **Taille des données**: 891 samples (train ~713) est petit pour les modèles modernes; augmentation synthétique (SMOTE) pourrait aider mais il faudrait les générées avec prudence surtout vu la complexité du dataset.
2. **Déséquilibre de classes**: 38% survie; bien que `class_weight='balanced'` aide.
3. **Catégories rares**: `title` et `deck` ont bcp de valeurs uniques; one-hot encoding ou embedding pourraient être utilisé pour mieux capturer les interactions.
4. **Interactions non-capturées**: Random Forest/GB capturent certaines interactions; des features d'interaction explicites (ex. `Pclass * Sex`) ou réseaux de neurones pourraient améliorer.
5. **Instabilité du clustering**: silhouette score (0.371) plut^t faible; essayer d'autres algo ou plus de composantes PCA.

--------------------------------------------------------------------------------------------------------

## Exécution

### Installation des dépendances

```bash
pip install -r requirements.txt
```

### Pipeline complet

```bash
python main.py
```

### Test des modèles seul

```bash
python core/train.py
```

### Visualisations

Les graphiques sont générés dans `outputs/`:
- `01_survie_par_profil.png`: taux survie par profil
- `02_survie_demographic.png`: survie par genre, classe, âge, solitude
- `03_fare_analysis.png`: impact du tarif
- `04_correlation_heatmap.png`: corrélations variables
- `06_survival_factors_analysis.png`: analyse croisée facteurs

--------------------------------------------------------------------------------------------------------

## Fichiers clés

### Fichiers racine

| Fichier               | Rôle |
|----------------------|------|
| `main.py`            | Point d'entrée : orchestration complète du pipeline |
| `config.py`          | Configuration centralisée (features, modèles, labels) |
| `encoders.py`        | Gestion des encodeurs et scalers |
| `pipeline.py`        | Classe TitanicPipeline : orchestration du pipeline |
| `utils.py`           | Utilitaires réutilisables (transformations, formatage) |

### Module core/

| Fichier               | Rôle |
|----------------------|------|
| `core/train.py`      | Classe `TitanicModelTrainer`: entraînement CV, test eval, clustering |
| `core/features.py`   | Classe `FeatureEngineer`: création des 21 profils, features dérivées |
| `core/predict.py`    | Classe `PassengerPredictor`: interface de prédiction interactive |
| `core/visualize.py`  | Classe `TitanicVisualizer`: visualisations analytiques |

### Module data/

| Fichier               | Rôle |
|----------------------|------|
| `data/data_loader.py` | Classe `TitanicDataLoader`: chargement flexible (XLS, CSV) |

--------------------------------------------------------------------------------------------------------

## Dépendances

- `pandas>=1.5`
- `numpy>=1.22`
- `scikit-learn>=1.0`
- `matplotlib>=3.5`
- `seaborn>=0.11`
- `openpyxl>=3.0` (pour XLS)
- `xlrd>=2.0` (pour XLS hérité)

--------------------------------------------------------------------------------------------------------