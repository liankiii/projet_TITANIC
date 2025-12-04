"""
pipeline.py: Classe principale pour l'orchestration (factorisée).
Encapsule la logique globale en une seule classe.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'data'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

from data.data_loader import TitanicDataLoader
from core.features import FeatureEngineer
from core.visualize import TitanicVisualizer
from config import FeatureConfig, ModelConfig, ClassNameMapping
from encoders import EncoderManager


class TitanicPipeline:
    """
    Pipeline complet avec meilleure factorisation.
    Encapsule: chargement, features, encodage, entraînement, clustering.
    """
    
    def __init__(self, data_path: str = "Titanic-Dataset.xls"):
        """Initialise le pipeline."""
        self.data_path = data_path
        self.df_raw = None
        self.df_engineered = None
        self.train_df = None
        self.test_df = None
        
        self.config = FeatureConfig()
        self.encoder_manager = EncoderManager()
        
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
        self.models = {}
        self.cv_results = {}
        self.test_results = {}
        self.clusters = None
    
    def load_data(self) -> pd.DataFrame:
        """Charge et retourne les données brutes."""
        loader = TitanicDataLoader(self.data_path)
        self.df_raw = loader.load_data()
        print(f"Données chargées: {len(self.df_raw)} lignes, {len(self.df_raw.columns)} colonnes")
        return self.df_raw
    
    def engineer_features(self) -> pd.DataFrame:
        """Feature engineering."""
        engineer = FeatureEngineer(self.df_raw)
        self.df_engineered = engineer.create_demographic_features()
        self.df_engineered = engineer.create_passenger_profiles()
        print(f"Features engineering: {len(self.df_engineered.columns)} colonnes")
        return self.df_engineered
    
    def split_data(self, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split train/test stratifié."""
        self.train_df, self.test_df = train_test_split(
            self.df_engineered,
            test_size=test_size,
            stratify=self.df_engineered['Survived'],
            random_state=ModelConfig.RANDOM_STATE
        )
        print(f"Split: {len(self.train_df)} train, {len(self.test_df)} test")
        return self.train_df, self.test_df
    
    def preprocess_features(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prétraitement complet des features."""
        # Entraîner encoders sur train_df
        self.X_train = self.encoder_manager.preprocess_features(
            self.train_df,
            self.config.numeric_cols,
            self.config.categorical_cols,
            fit=True
        )
        self.y_train = self.train_df['Survived'].values
        
        # Appliquer sur test_df
        self.X_test = self.encoder_manager.preprocess_features(
            self.test_df,
            self.config.numeric_cols,
            self.config.categorical_cols,
            fit=False
        )
        self.y_test = self.test_df['Survived'].values
        
        print(f"Features: {self.X_train.shape}")
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def train_models_cv(self) -> Dict[str, Dict[str, float]]:
        """Entraînement avec validation croisée stratifiée."""
        skf = StratifiedKFold(n_splits=ModelConfig.CV_SPLITS, shuffle=True, random_state=ModelConfig.RANDOM_STATE)
        results = {}
        
        for model_name, config in ModelConfig.MODELS.items():
            acc_scores = []
            f1_scores = []
            prec_scores = []
            rec_scores = []
            
            for train_idx, val_idx in skf.split(self.X_train, self.y_train):
                X_tr, X_val = self.X_train[train_idx], self.X_train[val_idx]
                y_tr, y_val = self.y_train[train_idx], self.y_train[val_idx]
                
                model = type(config['model'])(**config['params'])
                model.fit(X_tr, y_tr)
                
                y_pred = model.predict(X_val)
                acc_scores.append(accuracy_score(y_val, y_pred))
                f1_scores.append(f1_score(y_val, y_pred))
                prec_scores.append(precision_score(y_val, y_pred))
                rec_scores.append(recall_score(y_val, y_pred))
            
            # Moyennes
            self.cv_results[model_name] = {
                'accuracy': float(np.mean(acc_scores)),
                'f1_score': float(np.mean(f1_scores)),
                'precision': float(np.mean(prec_scores)),
                'recall': float(np.mean(rec_scores))
            }
            results[model_name] = self.cv_results[model_name]
        
        return results
    
    def evaluate_on_test(self) -> Dict[str, Dict[str, float]]:
        """Évaluation sur test set."""
        results = {}
        
        for model_name, config in ModelConfig.MODELS.items():
            model = type(config['model'])(**config['params'])
            model.fit(self.X_train, self.y_train)
            
            self.models[model_name] = model
            
            y_pred = model.predict(self.X_test)
            
            self.test_results[model_name] = {
                'accuracy': float(accuracy_score(self.y_test, y_pred)),
                'f1_score': float(f1_score(self.y_test, y_pred)),
                'precision': float(precision_score(self.y_test, y_pred)),
                'recall': float(recall_score(self.y_test, y_pred))
            }
            results[model_name] = self.test_results[model_name]
        
        return results
    
    def perform_clustering(self, n_clusters: int = 5) -> np.ndarray:
        """Clustering k-means avec PCA."""
        pca = PCA(n_components=ModelConfig.CLUSTERING_COMPONENTS, random_state=ModelConfig.RANDOM_STATE)
        X_pca = pca.fit_transform(self.X_train)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=ModelConfig.RANDOM_STATE, n_init=10)
        self.clusters = kmeans.fit_predict(X_pca)
        
        silhouette = silhouette_score(X_pca, self.clusters)
        print(f"Clustering: {n_clusters} clusters, silhouette={silhouette:.3f}")
        
        return self.clusters
    
    def print_results(self):
        """Affiche tous les résultats."""
        print("\n" + "="*70)
        print("VALIDATION CROISÉE (5-fold)")
        print("="*70)
        
        for model_name, metrics in self.cv_results.items():
            print(f"\n{model_name.upper().replace('_', ' ')}:")
            for metric, value in metrics.items():
                print(f"  {metric:12s}: {value:.3f}")
        
        print("\n" + "="*70)
        print("ÉVALUATION SUR TEST SET")
        print("="*70)
        
        for model_name, metrics in self.test_results.items():
            print(f"\n{model_name.upper().replace('_', ' ')}:")
            for metric, value in metrics.items():
                print(f"  {metric:12s}: {value:.3f}")
    
    def get_best_model(self, metric: str = 'f1_score') -> Tuple[str, Any]:
        """Retourne le meilleur modèle selon une métrique."""
        best_name = max(self.test_results, key=lambda x: self.test_results[x][metric])
        return best_name, self.models[best_name]
    
    def run_full_pipeline(self):
        """Exécute le pipeline complet."""
        print("="*70)
        print("PIPELINE TITANIC - FACTORISATION AMÉLIORÉE")
        print("="*70)
        
        self.load_data()
        self.engineer_features()
        self.split_data()
        self.preprocess_features()
        self.train_models_cv()
        self.evaluate_on_test()
        self.perform_clustering()
        
        # Générer les visualisations des clusters
        print("\n" + "="*70)
        print("GÉNÉRATION DES VISUALISATIONS")
        print("="*70)
        visualizer = TitanicVisualizer(self.df_engineered, output_dir="outputs")
        visualizer.plot_clusters(self.X_train, self.clusters)
        visualizer.plot_clusters_with_survival(self.X_train, self.clusters, self.y_train)
        
        self.print_results()
        
        best_name, best_model = self.get_best_model()
        print(f"\nMeilleur modèle (test F1): {best_name}")
        
        return self
