"""
Module de chargement du dataset Titanic.
"""
import os
import pandas as pd


class TitanicDataLoader:
    """Gestionnaire de chargement du dataset Titanic."""
    
    def __init__(self, file_path: str):
        """
        Initialise le loader avec le chemin du fichier.
        
        Args:
            file_path: Chemin vers le fichier (CSV, XLS, XLSX)
        """
        self.file_path = file_path
        self.df = None
    
    def load_data(self) -> pd.DataFrame:
        """
        Charge le fichier (détecte automatiquement le format).
        
        Returns:
            DataFrame avec les données
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Fichier non trouvé: {self.file_path}")
        
        print(f"Chargement: {self.file_path}")
        
        # Déterminer le format et charger
        try:
            if self.file_path.endswith('.xlsx'):
                self.df = pd.read_excel(self.file_path)
            elif self.file_path.endswith('.xls'):
                # Essayer d'abord en Excel, sinon comme CSV
                try:
                    self.df = pd.read_excel(self.file_path, engine='xlrd')
                except Exception:
                    # Si ça échoue, charger comme CSV
                    self.df = pd.read_csv(self.file_path)
            else:
                self.df = pd.read_csv(self.file_path)
        except Exception as e:
            # Dernier recours: essayer CSV
            try:
                self.df = pd.read_csv(self.file_path)
            except Exception:
                raise ValueError(f"Impossible de charger le fichier: {e}")
        
        print(f"OK: {self.df.shape[0]} lignes, {self.df.shape[1]} colonnes")
        
        return self.df
    
    def describe_data(self) -> dict:
        """Retourne infos sur les données."""
        if self.df is None:
            raise ValueError("Données non chargées. Appelez load_data() d'abord.")
        
        info = {
            "shape": self.df.shape,
            "columns": self.df.columns.tolist(),
            "dtypes": self.df.dtypes.to_dict(),
            "missing": self.df.isnull().sum().to_dict(),
        }
        
        if "Survived" in self.df.columns:
            info["survival_rate"] = self.df["Survived"].mean()
        
        print("\nDataset:")
        print(f"   Shape: {info['shape']}")
        if "survival_rate" in info:
            print(f"   Taux de survie: {info['survival_rate']:.1%}")
        print(f"\n   Valeurs manquantes:")
        for col, count in info["missing"].items():
            if count > 0:
                pct = 100 * count / len(self.df)
                print(f"     - {col}: {count} ({pct:.1f}%)")
        
        return info


if __name__ == "__main__":
    loader = TitanicDataLoader("Titanic-Dataset.xls")
    df = loader.load_data()
    info = loader.describe_data()
    print("\nPremiers enregistrements:")
    print(df.head())