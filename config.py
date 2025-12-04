"""
config.py: Configuration centralisée pour le pipeline Titanic.
Permet une meilleure factorisation et maintenabilité.
"""
from dataclasses import dataclass
from typing import List, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


@dataclass
class FeatureConfig:
    """Configuration des features du projet."""
    
    # Features numériques
    numeric_cols: List[str] = None
    # Features catégoriques
    categorical_cols: List[str] = None
    # Colonnes à supprimer pour les visualisations
    drop_cols: List[str] = None
    # Tailles des bins pour les groupes d'âge
    age_bins: List[int] = None
    age_labels: List[str] = None
    
    def __post_init__(self):
        if self.numeric_cols is None:
            self.numeric_cols = ['Age', 'Fare', 'fare_per_person', 'Pclass', 'SibSp', 'Parch', 'family_size']
        if self.categorical_cols is None:
            self.categorical_cols = ['Sex', 'Embarked', 'ticket_class', 'age_group', 'title', 'deck', 'ticket_prefix']
        if self.drop_cols is None:
            self.drop_cols = ['PassengerId', 'Ticket', 'Cabin', 'Name']
        if self.age_bins is None:
            self.age_bins = [0, 12, 18, 35, 60, 150]
        if self.age_labels is None:
            self.age_labels = ['enfant', 'ado', 'jeune_adulte', 'adulte', 'senior']


@dataclass
class TitleMapping:
    """Mapping pour normalisation des titres."""
    mapping: Dict[str, str] = None
    
    def __post_init__(self):
        if self.mapping is None:
            self.mapping = {
                'Mlle': 'Miss', 'Mme': 'Mrs', 'Ms': 'Miss',
                'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare', 'Col': 'Rare',
                'Don': 'Rare', 'Dr': 'Rare', 'Major': 'Rare', 'Rev': 'Rare', 'Sir': 'Rare',
                'Jonkheer': 'Rare', 'Dona': 'Rare'
            }


@dataclass
class ClassNameMapping:
    """Mapping pour les classes de billet."""
    mapping: Dict[int, str] = None
    
    def __post_init__(self):
        if self.mapping is None:
            self.mapping = {1: 'premiere', 2: 'deuxieme', 3: 'troisieme'}


class ModelConfig:
    """Configuration des modèles de ML."""
    
    MODELS = {
        'logistic_regression': {
            'model': LogisticRegression(random_state=42, max_iter=2000, n_jobs=None),
            'params': {'random_state': 42, 'max_iter': 2000, 'n_jobs': None}
        },
        'random_forest': {
            'model': RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'params': {
                'n_estimators': 300,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'random_state': 42,
                'n_jobs': -1
            }
        },
        'gradient_boosting': {
            'model': GradientBoostingClassifier(n_estimators=200, random_state=42),
            'params': {'n_estimators': 200, 'random_state': 42}
        }
    }
    
    CV_SPLITS = 5
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CLUSTERING_COMPONENTS = 4
    OPTIMAL_CLUSTERS = 5


class LabelConfig:
    """Configuration des labels pour encodeurs."""
    
    LABELS = {
        'Sex': ['male', 'female'],
        'Embarked': ['S', 'C', 'Q'],
        'ticket_class': ['premiere', 'deuxieme', 'troisieme'],
        'title': ['Mr', 'Mrs', 'Miss', 'Master', 'Dr', 'Rev', 'Rare', 'Unknown'],
        'deck': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'U'],
        'ticket_prefix': ['NUM', 'PC', 'A', 'PP', 'CA', 'STONO', 'SC', 'CASOTON', 'F', 'SC/Paris', 'SCParis', 'C', 'SOPP', 'SW/PP', 'SOP', 'SOTONO2', 'FC', 'WEP', 'STONO2', 'SC/AH', 'LINE', 'UNK', 'Unknown'],
        'age_group': ['enfant', 'ado', 'jeune_adulte', 'adulte', 'senior']
    }
    
    DEFAULT_VALUES = {
        'title': 'Unknown',
        'deck': 'U',
        'ticket_prefix': 'UNK'
    }
