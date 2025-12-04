"""
utils.py: Utilitaires factorisés.
Fonctions réutilisables pour conversions et transformations.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from config import LabelConfig, ModelConfig


class FeatureTransformer:
    """Transformations de features réutilisables."""
    
    @staticmethod
    def create_ticket_class(df: pd.DataFrame, pclass_col: str = 'Pclass') -> pd.Series:
        """Crée la colonne ticket_class à partir de Pclass."""
        mapping = {1: 'premiere', 2: 'deuxieme', 3: 'troisieme'}
        return df[pclass_col].map(mapping)
    
    @staticmethod
    def fill_missing_values(df: pd.DataFrame, numeric_cols: list, categorical_cols: list) -> pd.DataFrame:
        """Remplit les valeurs manquantes."""
        df_copy = df.copy()
        
        for col in numeric_cols:
            if col in df_copy.columns and df_copy[col].isna().any():
                df_copy[col] = df_copy[col].fillna(df_copy[col].median())
        
        for col in categorical_cols:
            if col in df_copy.columns and df_copy[col].isna().any():
                mode_val = df_copy[col].mode()
                if not mode_val.empty:
                    df_copy[col] = df_copy[col].fillna(mode_val.iloc[0])
        
        return df_copy
    
    @staticmethod
    def handle_unknown_labels(value: Any, le_classes: np.ndarray, default: Optional[Any] = None) -> Any:
        """Gère les labels inconnus en les remplaçant par la première classe."""
        if value in le_classes:
            return value
        return default if default else le_classes[0]


class MetricsFormatter:
    """Formatage des résultats."""
    
    @staticmethod
    def format_model_results(results: Dict[str, Dict[str, float]]) -> str:
        """Formate les résultats en string."""
        output = []
        for model_name, metrics in results.items():
            output.append(f"{model_name.upper().replace('_', ' ')}:")
            for metric, value in metrics.items():
                output.append(f"  {metric:12s}: {value:.3f}")
            output.append("")
        return "\n".join(output)
    
    @staticmethod
    def compare_cv_vs_test(cv_results: Dict, test_results: Dict) -> str:
        """Compare CV et test pour détecter overfitting."""
        output = []
        output.append("COMPARAISON CV vs TEST (détection overfitting)")
        output.append("-" * 50)
        
        for model_name in cv_results:
            cv_f1 = cv_results[model_name].get('f1', 0)
            test_f1 = test_results[model_name].get('f1', 0)
            diff = cv_f1 - test_f1
            
            status = "OK" if diff < 0.05 else "OVERFITTING LÉGER" if diff < 0.15 else "OVERFITTING"
            output.append(f"{model_name:25s} CV:{cv_f1:.3f} → Test:{test_f1:.3f} ({status})")
        
        return "\n".join(output)


class InputValidator:
    """Validation d'inputs."""
    
    @staticmethod
    def validate_passenger_dict(data: dict, required_cols: list) -> bool:
        """Valide un dictionnaire passager."""
        missing = [col for col in required_cols if col not in data]
        if missing:
            raise ValueError(f"Colonnes manquantes: {missing}")
        return True
    
    @staticmethod
    def normalize_values(data: dict, config: Dict) -> dict:
        """Normalise les valeurs selon la configuration."""
        normalized = data.copy()
        
        if 'Pclass' in normalized:
            normalized['ticket_class'] = FeatureTransformer.create_ticket_class(
                pd.DataFrame([normalized])
            ).values[0]
        
        return normalized
