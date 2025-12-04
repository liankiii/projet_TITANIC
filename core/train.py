"""
Module d'entraînement et clustering pour identifier les groupes de survie.
Entraîne des modèles prédictifs et analyse les patterns de survie.
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report
)
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from typing import Tuple, Dict, Any


class TitanicModelTrainer:
    """Entraîneur et analyseur de modèles pour la survie Titanic."""
    
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame | None = None):
        """
        Initialise le trainer.
        
        Args:
            train_df: DataFrame d'entraînement
            test_df: DataFrame de test (optionnel)
        """
        self.train_df = train_df.copy()
        self.test_df = test_df.copy() if test_df is not None else None
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.feature_names = None
    
    def preprocess_features(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Prépare les features pour l'entraînement.
        
        Args:
            df: DataFrame à traiter
            fit: Si True, crée les scalers et encodeurs
        
        Returns:
            Array des features normalisées
        """
        df_proc = df.copy()
        
        # Sélectionner les colonnes (ajout de features utiles connues sur Titanic)
        numeric_cols = ['Age', 'Fare', 'fare_per_person', 'Pclass', 'SibSp', 'Parch', 'family_size']
        categorical_cols = ['Sex', 'Embarked', 'ticket_class', 'age_group', 'title', 'deck', 'ticket_prefix']
        
        # Traiter les valeurs manquantes
        for col in numeric_cols:
            if col in df_proc.columns:
                df_proc[col] = df_proc[col].fillna(df_proc[col].median())
        
        for col in categorical_cols:
            if col in df_proc.columns:
                mode_val = df_proc[col].mode().iloc[0] if not df_proc[col].mode().empty else None
                if mode_val is not None:
                    df_proc[col] = df_proc[col].fillna(mode_val)
        
        # Encoder les variables catégoriques
        X = df_proc[numeric_cols].values.astype(float)
        
        for col in categorical_cols:
            if col in df_proc.columns:
                if fit:
                    le = LabelEncoder()
                    encoded = le.fit_transform(df_proc[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    le = self.label_encoders[col]
                    df_proc[col] = df_proc[col].astype(str)
                    unknown_mask = ~df_proc[col].isin(le.classes_)
                    df_proc.loc[unknown_mask, col] = le.classes_[0]
                    encoded = le.transform(df_proc[col])
                
                X = np.column_stack([X, encoded])
        
        # Normaliser
        if fit:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            self.scalers['main'] = scaler
        else:
            scaler = self.scalers.get('main')
            if scaler:
                X = scaler.transform(X)
        
        self.feature_names = numeric_cols + categorical_cols
        return X
    
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prépare les données d'entraînement.
        
        Returns:
            Tuple (X_train, y_train)
        """
        self.X_train = self.preprocess_features(self.train_df, fit=True)
        self.y_train = self.train_df['Survived'].values
        
        return self.X_train, self.y_train
    
    def train_models(self) -> Dict[str, Any]:
        """
        Entraîne plusieurs modèles.
        
        Returns:
            Dict avec les modèles entraînés et leurs performances
        """
        if self.X_train is None:
            self.prepare_data()
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        models_config = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=2000, n_jobs=None),
            'random_forest': RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=200, random_state=42)
        }
        
        results = {}

        for name, base_model in models_config.items():
            acc_scores = []
            f1_scores = []
            prec_scores = []
            rec_scores = []

            for train_idx, val_idx in skf.split(self.X_train, self.y_train):
                X_tr, X_val = self.X_train[train_idx], self.X_train[val_idx]
                y_tr, y_val = self.y_train[train_idx], self.y_train[val_idx]

                # Cloner le modèle pour éviter le partage d'état entre folds
                model = type(base_model)(**base_model.get_params())
                model.fit(X_tr, y_tr)

                y_pred = model.predict(X_val)
                acc_scores.append(accuracy_score(y_val, y_pred))
                f1_scores.append(f1_score(y_val, y_pred))
                prec_scores.append(precision_score(y_val, y_pred))
                rec_scores.append(recall_score(y_val, y_pred))

            # Entraîner le modèle final sur tout l'ensemble d'entraînement
            final_model = type(base_model)(**base_model.get_params())
            final_model.fit(self.X_train, self.y_train)
            self.models[name] = final_model

            results[name] = {
                'accuracy': float(np.mean(acc_scores)),
                'f1_score': float(np.mean(f1_scores)),
                'precision': float(np.mean(prec_scores)),
                'recall': float(np.mean(rec_scores))
            }

            print(f"Accuracy: {results[name]['accuracy']:.3f}, F1: {results[name]['f1_score']:.3f}")
        
        return results
    
    def analyze_feature_importance(self) -> pd.DataFrame:
        """
        Analyse l'importance des features selon le Random Forest.
        
        Returns:
            DataFrame trié par importance
        """
        if 'random_forest' not in self.models:
            self.train_models()
        
        rf = self.models['random_forest']
        importances = rf.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df

    def evaluate_on_test(self) -> Dict[str, Any]:
        """
        Evalue les modeles sur l'ensemble de test si disponible.

        Returns:
            Dict avec les performances sur le test set
        """
        if self.test_df is None or len(self.test_df) == 0:
            return {}

        self.X_test = self.preprocess_features(self.test_df, fit=False)
        if 'Survived' in self.test_df.columns:
            self.y_test = self.test_df['Survived'].values
        else:
            return {}

        test_results = {}
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            test_results[name] = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'f1_score': f1_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred)
            }

        return test_results

    @staticmethod
    def split_train_test(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple:
        """
        Scinde les donnees en train/test de maniere stratifiee.

        Args:
            df: DataFrame complet
            test_size: Proportion du test set (0.2 = 20%)
            random_state: Graine aleatoire

        Returns:
            Tuple (train_df, test_df)
        """
        if 'Survived' not in df.columns:
            raise ValueError("Colonne 'Survived' requise pour le split stratifie")

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df['Survived'],
            random_state=random_state
        )
        return train_df, test_df
    
    def identify_survival_clusters(
        self,
        n_clusters_list: list[int] = [2, 3, 4, 5, 6, 7, 8],
        random_states: list[int] = [0, 1, 2, 3, 42],
        n_clusters: int | list[int] | None = None,
        use_pca: bool = True,
        pca_components: int = 4
    ) -> np.ndarray:
        """
        Cherche le meilleur clustering K-Means via score de silhouette.
        
        Args:
            n_clusters_list: Liste de k à tester
            random_states: Liste de random_state à tester
        
        Returns:
            Array avec les labels du meilleur cluster
        """

        if self.X_train is None:
            self.prepare_data()

        # Compatibilité ascendante : accepter le mot-clé `n_clusters`
        if n_clusters is not None:
            if isinstance(n_clusters, int):
                n_clusters_list = [n_clusters]
            elif isinstance(n_clusters, list):
                n_clusters_list = n_clusters

        best_score = -1.0
        best_model = None
        best_labels = None
        best_params = None

        print("\nRecherche du meilleur K-Means par silhouette...")
        # Réduction de dimension optionnelle pour le scoring silhouette
        X_eval = self.X_train
        if use_pca:
            try:
                pca = PCA(n_components=min(pca_components, X_eval.shape[1]))
                X_eval = pca.fit_transform(X_eval)
            except Exception:
                X_eval = self.X_train
        for k in n_clusters_list:
            for rs in random_states:
                kmeans = KMeans(n_clusters=k, random_state=rs, n_init=10)
                labels = kmeans.fit_predict(X_eval)
                # silhouette nécessite au moins 2 clusters et moins que nb samples
                if k > 1 and k < len(X_eval):
                    score = silhouette_score(X_eval, labels)
                else:
                    score = -1.0
                if score > best_score:
                    best_score = score
                    best_model = kmeans
                    best_labels = labels
                    best_params = (k, rs)

        if best_model is None:
            # fallback: k=5, rs=42
            fallback_X = X_eval
            best_model = KMeans(n_clusters=5, random_state=42, n_init=10).fit(fallback_X)
            best_labels = best_model.labels_
            best_score = -1.0
            best_params = (5, 42)

        k, rs = best_params
        print(f"Meilleur K-Means: k={k}, random_state={rs}, silhouette={best_score:.3f}")
        self.models['kmeans'] = best_model
        return best_labels

    def analyze_cluster_survival(self, clusters: np.ndarray) -> pd.DataFrame:
        """
        Analyse le taux de survie par cluster.
        
        Args:
            clusters: Array avec les labels de cluster
        
        Returns:
            DataFrame avec stats de survie par cluster
        """
        df_analysis = self.train_df.copy()
        df_analysis['cluster'] = clusters
        
        cluster_summary = df_analysis.groupby('cluster').agg({
            'Survived': ['sum', 'count', 'mean'],
            'Age': 'mean',
            'Fare': 'mean',
            'Pclass': 'mean'
        }).round(3)
        
        cluster_summary.columns = [
            'survivants', 'total', 'taux_survie',
            'age_moyen', 'tarif_moyen', 'classe_moyenne'
        ]
        
        return cluster_summary.sort_values('taux_survie', ascending=False)


if __name__ == "__main__":
    # Import local
    import sys
    import os
    # Ajouter la racine du projet au PYTHONPATH pour importer les modules locaux
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from data.data_loader import TitanicDataLoader
    # importer le module local `features` (dans `core/`)
    sys.path.insert(0, os.path.dirname(__file__))
    from features import FeatureEngineer
    from sklearn.metrics import silhouette_score
    
    # Charger les données (chemin résolu depuis la racine du projet)
    data_file = os.path.join(project_root, 'data', 'Titanic-Dataset.xls')
    loader = TitanicDataLoader(data_file)
    df = loader.load_data()
    
    # Créer les features
    engineer = FeatureEngineer(df)
    df = engineer.create_demographic_features()
    df = engineer.create_passenger_profiles()
    
    # Entraîner les modèles
    trainer = TitanicModelTrainer(df, None)
    results = trainer.train_models()
    
    print("\nPerformance des modèles:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.3f}")
    
    print("\nImportance des features:")
    importance = trainer.analyze_feature_importance()
    print(importance)
    
    print("\nAnalyse des clusters:")
    clusters = trainer.identify_survival_clusters(n_clusters=5)
    cluster_summary = trainer.analyze_cluster_survival(clusters)
    print(cluster_summary)
