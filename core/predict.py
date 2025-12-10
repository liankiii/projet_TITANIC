"""
predict.py: Module de prédiction interactive pour la survie Titanic.
Permet de saisir les caractéristiques d'un passager et obtenir une prédiction.
"""
import sys
import os
import pandas as pd
import numpy as np

# Ajouter le chemin parent pour les imports (A MODIFIER SELON VOTRE STRUCTURE et si erreur d'import sur "data")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import FeatureConfig, LabelConfig, ClassNameMapping
from encoders import EncoderManager


class PassengerPredictor:
    """Prédicteur pour la survie d'un passager."""
    
    def __init__(self, pipeline):
        """
        Initialise le prédicteur.
        
        Args:
            pipeline: Instance de TitanicPipeline avec modèles entraînés
        """
        self.pipeline = pipeline
        self.model = pipeline.models.get('random_forest')
        self.encoder_manager = pipeline.encoder_manager
        self.config = pipeline.config
        
        if not self.model:
            raise ValueError("Random Forest non entraîné")
    
    def get_passenger_input(self) -> dict:
        """
        Menu interactif pour saisir les caractéristiques d'un passager.
        
        Returns:
            Dict avec les données du passager
        """
        print("\n" + "="*60)
        print("PRÉDICTION DE SURVIE TITANIC")
        print("="*60)
        print("\nEntrez les caractéristiques du passager:")
        
        # Nom
        name = input("\nNom du passager: ").strip()
        if not name:
            name = "Passager Inconnu"
        
        # Sexe
        while True:
            sex = input("Sexe (male/female): ").strip().lower()
            if sex in ['male', 'female', 'm', 'f', 'h', 'homme', 'femme']:
                sex = 'male' if sex in ['male', 'm', 'h', 'homme'] else 'female'
                break
            print("Répondez par 'male' ou 'female'")
        
        # Age
        while True:
            try:
                age = float(input("Âge (0-80): "))
                if 0 <= age <= 120:
                    break
                print("L'âge doit être entre 0 et 120")
            except ValueError:
                print("Entrez un nombre valide")
        
        # Classe
        print("\nClasses de billets:")
        print("  1 - Première classe (Luxe)")
        print("  2 - Deuxième classe (Intermédiaire)")
        print("  3 - Troisième classe (Économique)")
        
        while True:
            try:
                pclass = int(input("Classe de billet (1, 2 ou 3): "))
                if pclass in [1, 2, 3]:
                    class_names = {1: 'Première classe', 2: 'Deuxième classe', 3: 'Troisième classe'}
                    print(f"Classe sélectionnée: {class_names[pclass]}")
                    break
                print("La classe doit être 1, 2 ou 3")
            except ValueError:
                print("Entrez un nombre entier")
        
        # Tarif
        print("\nTarifs typiques selon la classe:")
        print("  1ère classe: 30-500£ (moyenne: ~85£)")
        print("  2e classe: 10-75£ (moyenne: ~20£)")
        print("  3e classe: 3-70£ (moyenne: ~13£)")
        
        while True:
            try:
                fare = float(input("Tarif payé en livres sterling (0-500): "))
                if 0 <= fare <= 1000:
                    break
                print("Le tarif doit être entre 0 et 1000")
            except ValueError:
                print("Entrez un nombre valide")
        
        # Nombre de frères/soeurs
        while True:
            try:
                sibsp = int(input("Nombre de frères/soeurs à bord: "))
                if 0 <= sibsp <= 10:
                    break
                print("Le nombre doit être entre 0 et 10")
            except ValueError:
                print("Entrez un nombre entier")
        
        # Nombre de parents/enfants
        while True:
            try:
                parch = int(input("Nombre de parents/enfants à bord: "))
                if 0 <= parch <= 10:
                    break
                print("Le nombre doit etre entre 0 et 10")
            except ValueError:
                print("Entrez un nombre entier")
        
        # Port d'embarquement
        print("\nPorts d'embarquement disponibles:")
        print("  1 - Southampton (S)")
        print("  2 - Cherbourg (C)") 
        print("  3 - Queenstown (Q)")
        
        while True:
            port_choice = input("Choisissez le port d'embarquement (1/2/3 ou S/C/Q): ").strip().upper()
            
            port_mapping = {
                '1': 'S', 'SOUTHAMPTON': 'S', 'S': 'S',
                '2': 'C', 'CHERBOURG': 'C', 'C': 'C',
                '3': 'Q', 'QUEENSTOWN': 'Q', 'Q': 'Q'
            }
            
            if port_choice in port_mapping:
                embarked = port_mapping[port_choice]
                port_names = {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}
                print(f"Port sélectionné: {port_names[embarked]} ({embarked})")
                break
            print("Choisissez 1, 2, 3 ou entrez S, C, Q")
        
        # Calculer les features dérivées
        family_size = sibsp + parch + 1
        fare_per_person = fare / family_size if family_size > 0 else fare
        
        # Déterminer le groupe d'âge
        age_bins = self.config.age_bins
        age_labels = self.config.age_labels
        age_group = age_labels[-1]  # Par défaut
        for i in range(len(age_bins) - 1):
            if age_bins[i] <= age < age_bins[i + 1]:
                age_group = age_labels[i]
                break
        
        # Classe de ticket
        class_mapping = ClassNameMapping().mapping
        ticket_class = class_mapping.get(pclass, 'troisieme')
        
        # Extraire le titre du nom (simpliste)
        title = 'Mr' if sex == 'male' else 'Miss'
        if age < 18 and sex == 'male':
            title = 'Master'
        elif 'mrs' in name.lower() or 'mme' in name.lower():
            title = 'Mrs'
        
        return {
            'name': name,
            'Sex': sex,
            'Age': age,
            'Pclass': pclass,
            'Fare': fare,
            'SibSp': sibsp,
            'Parch': parch,
            'Embarked': embarked,
            'family_size': family_size,
            'fare_per_person': fare_per_person,
            'age_group': age_group,
            'ticket_class': ticket_class,
            'title': title,
            'deck': 'U',  # Unknown par défaut
            'ticket_prefix': 'UNK'  # Unknown par défaut
        }
    
    def prepare_input(self, passenger: dict) -> np.ndarray:
        """
        Prépare les données pour la prédiction.
        
        Args:
            passenger: Dict avec les données du passager
        
        Returns:
            Array normalisé pour le modèle
        """
        # Créer un DataFrame à partir du passager
        df = pd.DataFrame([passenger])
        
        # Utiliser l'encoder_manager du pipeline
        X = self.encoder_manager.preprocess_features(
            df,
            self.config.numeric_cols,
            self.config.categorical_cols,
            fit=False  # Utiliser les encodeurs existants
        )
        
        return X
    
    def predict(self, passenger: dict) -> tuple:
        """
        Prédit la survie d'un passager.
        
        Args:
            passenger: Dict avec les données du passager
        
        Returns:
            Tuple (probabilité_survie, interprétation)
        """
        X = self.prepare_input(passenger)
        
        # Prédiction
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0]
        
        # Probabilité de survie
        prob_survive = probability[1]
        
        # Interprétation
        if prob_survive > 0.7:
            interpretation = "TRÈS BON (>70%)"
        elif prob_survive > 0.5:
            interpretation = "BON (>50%)"
        elif prob_survive > 0.3:
            interpretation = "MOYEN (30-50%)"
        else:
            interpretation = "FAIBLE (<30%)"
        
        return prob_survive, interpretation
    
    def show_profile(self, passenger: dict):
        """Affiche le profil du passager."""
        port_names = {
            'S': 'Southampton', 
            'C': 'Cherbourg', 
            'Q': 'Queenstown'
        }
        
        class_names = {
            1: '1ère classe (Luxe)',
            2: '2e classe (Intermédiaire)', 
            3: '3e classe (Économique)'
        }
        
        print("\nPROFIL DU PASSAGER:")
        print("-" * 60)
        print(f"Nom: {passenger['name']}")
        print(f"Sexe: {'Femme' if passenger['Sex'] == 'female' else 'Homme'}")
        print(f"Âge: {passenger['Age']} ans ({passenger['age_group']})")
        print(f"Classe: {class_names.get(passenger['Pclass'], passenger['Pclass'])}")
        print(f"Tarif: {passenger['Fare']:.2f} £ ({passenger['fare_per_person']:.2f} £/personne)")
        family_text = "Seul(e)" if passenger['family_size'] == 1 else f"{passenger['family_size']} personnes"
        print(f"Famille: {family_text}")
        print(f"Embarquement: {port_names.get(passenger['Embarked'], passenger['Embarked'])}")
        print(f"Titre: {passenger['title']}")
    
    def run_prediction(self):
        """Lance une prédiction interactive."""
        try:
            # Saisie
            passenger = self.get_passenger_input()
            
            # Afficher le profil
            self.show_profile(passenger)
            
            # Prédiction
            print("\nCALCUL DE LA PRÉDICTION...")
            prob, interp = self.predict(passenger)
            
            # Résultats
            print("\n" + "="*60)
            print("RÉSULTAT DE LA PRÉDICTION")
            print("="*60)
            print(f"\nChances de survie: {prob*100:.1f}%")
            print(f"Évaluation: {interp}\n")
            
            if prob > 0.5:
                print("Le passager a plus de chances de SURVIVRE")
            else:
                print("Le passager a plus de chances de NE PAS SURVIVRE\n")
            
            return prob
        
        except KeyboardInterrupt:
            print("\n\nAnnulé par l'utilisateur.")
            return None
        except Exception as e:
            print(f"Erreur: {e}")
            import traceback
            traceback.print_exc()
            return None


def interactive_menu(pipeline):
    """
    Menu interactif pour les prédictions.
    
    Args:
        pipeline: Instance de TitanicPipeline
    """
    predictor = PassengerPredictor(pipeline)
    
    # Première prédiction
    predictor.run_prediction()
    
    while True:
        try:
            choice = input("\n\nVoulez-vous faire une autre prédiction? (O/N): ").strip().upper()
            if choice in ['N', 'NON']:
                print("\nFin du programme.")
                break
            elif choice not in ['O', 'OUI', 'Y', 'YES', '']:
                print("Répondez par O (oui) ou N (non)")
                continue
            
            predictor.run_prediction()
        
        except KeyboardInterrupt:
            print("\n\nProgramme interrompu.")
            break


if __name__ == "__main__":
    print("="*60)
    print("CHARGEMENT DU MODÈLE...")
    print("="*60)
    
    # Importer et exécuter le pipeline
    from pipeline import TitanicPipeline
    
    # Chemin du fichier de données (relatif au répertoire parent)
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Titanic-Dataset.xls')
    
    # Créer et entraîner le pipeline
    pipeline = TitanicPipeline(data_path)
    pipeline.load_data()
    pipeline.engineer_features()
    pipeline.split_data()
    pipeline.preprocess_features()
    pipeline.train_models_cv()
    pipeline.evaluate_on_test()
    
    print("\nModèle chargé ")
    print(f"   Meilleur F1 score (test): {pipeline.test_results['random_forest']['f1_score']:.3f}")
    
    # Lancer le menu interactif
    interactive_menu(pipeline)
