"""
main.py: Script principal
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from pipeline import TitanicPipeline
from utils import MetricsFormatter


def main():
    """Exécute le pipeline complet."""
    
    # Initialiser et exécuter
    pipeline = TitanicPipeline("data/Titanic-Dataset.xls")
    pipeline.run_full_pipeline()
    
    # Affichage avancé
    #print("\n" + "="*70)
    #print("ANALYSE COMPARATIVE")
    #print("="*70)
    #print(MetricsFormatter.compare_cv_vs_test(pipeline.cv_results, pipeline.test_results))
    
    # Résumé final
    print("\n" + "="*70)
    print("RÉSUMÉ CONFIGURATION")
    print("="*70)
    print(f"Features numériques: {len(pipeline.config.numeric_cols)}")
    print(f"Features catégoriques: {len(pipeline.config.categorical_cols)}")
    print(f"CV splits: {5}")
    print(f"Clustering: {5} clusters")
    print(f"Scaler utilisé: StandardScaler")
    print(f"PCA components: 4")


if __name__ == "__main__":
    main()
