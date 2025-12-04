"""
encoders.py: Manager centralisé pour encodeurs et scalers.
Factorisation de la gestion des preprocessing.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Dict, Tuple, Optional
from config import LabelConfig


class EncoderManager:
    """Gère tous les encodeurs et scalers du pipeline."""
    
    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: list = []
    
    def fit_encoders(self, df: pd.DataFrame, categorical_cols: list) -> None:
        """
        Entraîne les label encoders sur les données.
        
        Args:
            df: DataFrame source
            categorical_cols: Colonnes catégoriques à encoder
        """
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                le.fit(df[col].astype(str))
                self.label_encoders[col] = le
    
    def fit_scaler(self, X: np.ndarray) -> None:
        """Entraîne le StandardScaler."""
        self.scaler = StandardScaler()
        self.scaler.fit(X)
    
    def encode_categorical(self, df: pd.DataFrame, categorical_cols: list, fit: bool = False) -> np.ndarray:
        """
        Encode les variables catégoriques.
        
        Args:
            df: DataFrame à traiter
            categorical_cols: Colonnes à encoder
            fit: Si True, entraîne les encodeurs
        
        Returns:
            Array encodé (n_samples, n_categorical_cols)
        """
        encoded = np.empty((len(df), len(categorical_cols)))
        
        for idx, col in enumerate(categorical_cols):
            if fit:
                le = LabelEncoder()
                encoded[:, idx] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                # Remplacer les labels inconnus par le premier
                df_col = df[col].astype(str).copy()
                unknown_mask = ~df_col.isin(le.classes_)
                df_col[unknown_mask] = le.classes_[0]
                encoded[:, idx] = le.transform(df_col)
        
        return encoded
    
    def encode_single_sample(self, data: dict) -> np.ndarray:
        """
        Encode un seul échantillon (pour prédictions).
        
        Args:
            data: Dictionnaire avec les valeurs
        
        Returns:
            Array encodé (1, n_features)
        """
        df = pd.DataFrame([data])
        encoded = np.empty((1, len(self.label_encoders)))
        
        for idx, col in enumerate(self.label_encoders.keys()):
            le = self.label_encoders[col]
            value = df[col].values[0]
            
            # Remplacer les labels inconnus
            if value not in le.classes_:
                value = le.classes_[0]
            
            encoded[0, idx] = le.transform([value])[0]
        
        return encoded
    
    def preprocess_features(self, df: pd.DataFrame, numeric_cols: list, 
                           categorical_cols: list, fit: bool = False) -> np.ndarray:
        """
        Preprocessing complet (features numériques + encodage catégorique + scaling).
        
        Args:
            df: DataFrame source
            numeric_cols: Colonnes numériques
            categorical_cols: Colonnes catégoriques
            fit: Si True, entraîne encoders et scaler
        
        Returns:
            Array normalisé des features
        """
        df_proc = df.copy()
        
        # Remplir valeurs manquantes
        for col in numeric_cols:
            if col in df_proc.columns:
                df_proc[col] = df_proc[col].fillna(df_proc[col].median())
        
        for col in categorical_cols:
            if col in df_proc.columns:
                mode_val = df_proc[col].mode()
                if not mode_val.empty:
                    df_proc[col] = df_proc[col].fillna(mode_val.iloc[0])
        
        # Extraire features numériques
        X = df_proc[numeric_cols].values.astype(float)
        
        # Encoder features catégoriques
        X_cat = self.encode_categorical(df_proc, categorical_cols, fit=fit)
        X = np.column_stack([X, X_cat])
        
        # Normaliser
        if fit:
            self.fit_scaler(X)
        
        if self.scaler:
            X = self.scaler.transform(X)
        
        self.feature_names = numeric_cols + categorical_cols
        return X
    
    def get_config_dict(self) -> dict:
        """Retourne la configuration sous forme de dictionnaire."""
        return {
            'encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
