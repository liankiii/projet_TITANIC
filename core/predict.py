"""
Module de prediction interactive pour la survie Titanic.
Permet de saisir les caracteristiques d'un passager et obtenir une prediction.
"""
import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


class PassengerPredictor:
    """Predictor pour la survie d'un passager."""
    
    def __init__(self, trainer):
        """
        Initialise le predictor.
        
        Args:
            trainer: Instance de TitanicModelTrainer avec modeles entrainess
        """
        self.trainer = trainer
        self.model = trainer.models.get('random_forest')
        if not self.model:
            raise ValueError("Random Forest non entrainnee")
    
    def get_passenger_input(self) -> dict:
        """
        Menu interactif pour saisir les caracteristiques d'un passager.
        
        Returns:
            Dict avec les donnees du passager
        """
        print("\n" + "="*60)
        print("PREDICTION DE SURVIE TITANIC")
        print("="*60)
        print("\nEntrez les caracteristiques du passager:")
        
        # Nom
        name = input("\nNom du passager: ").strip()
        
        # Sexe
        while True:
            sex = input("Sexe (male/female): ").strip().lower()
            if sex in ['male', 'female']:
                break
            print("Repondez par 'male' ou 'female'")
        
        # Age
        while True:
            try:
                age = float(input("Age (0-80): "))
                if 0 <= age <= 120:
                    break
                print("L'age doit etre entre 0 et 120")
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
        
        # Nombre de freres/soeurs
        while True:
            try:
                sibsp = int(input("Nombre de freres/soeurs a bord: "))
                if 0 <= sibsp <= 10:
                    break
                print("Le nombre doit etre entre 0 et 10")
            except ValueError:
                print("Entrez un nombre entier")
        
        # Nombre de parents/enfants
        while True:
            try:
                parch = int(input("Nombre de parents/enfants a bord: "))
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
            
            # Mapper les choix vers les codes
            port_mapping = {
                '1': 'S', 'SOUTHAMPTON': 'S', 'S': 'S',
                '2': 'C', 'CHERBOURG': 'C', 'C': 'C',
                '3': 'Q', 'QUEENSTOWN': 'Q', 'Q': 'Q'
            }
            
            if port_choice in port_mapping:
                embarked = port_mapping[port_choice]
                port_names = {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}
                print(f"Port selectionne: {port_names[embarked]} ({embarked})")
                break
            print("Choisissez 1, 2, 3 ou entrez S, C, Q")
        
        return {
            'name': name,
            'Sex': sex,
            'Age': age,
            'Pclass': pclass,
            'Fare': fare,
            'SibSp': sibsp,
            'Parch': parch,
            'Embarked': embarked,
            'family_size': sibsp + parch + 1
        }
    
    def prepare_input(self, passenger: dict) -> np.ndarray:
        """
        Prepare input data for model prediction.
        
        Args:
            passenger: Dict with passenger data
        
        Returns:
            Normalized feature array for model
        """
        # Create DataFrame from passenger dict
        df = pd.DataFrame([passenger])
        
        # Ensure numeric columns - include derived features
        numeric_cols = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch', 'family_size', 'fare_per_person']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Create ticket_class from Pclass (like FeatureEngineer does)
        class_names = {1: 'premiere', 2: 'deuxieme', 3: 'troisieme'}
        if 'ticket_class' not in df.columns:
            df['ticket_class'] = df['Pclass'].map(class_names)
        
        # Extract numeric features
        X = df[numeric_cols].values.astype(float)
        
        # Encode categorical features - match the trainer's exact columns
        categorical_cols = ['Sex', 'Embarked', 'ticket_class', 'age_group', 'title', 'deck', 'ticket_prefix']
        
        for col in categorical_cols:
            if col in self.trainer.label_encoders:
                le = self.trainer.label_encoders[col]
                # Handle unseen labels by replacing with first class
                unseen_mask = ~pd.Series([df[col].values[0]]).isin(le.classes_)
                if unseen_mask.any():
                    df[col] = le.classes_[0]
                encoded = le.transform([df[col].values[0]])
            else:
                le = LabelEncoder()
                if col == 'Sex':
                    le.fit(['male', 'female'])
                elif col == 'Embarked':
                    le.fit(['S', 'C', 'Q'])
                elif col == 'ticket_class':
                    le.fit(['premiere', 'deuxieme', 'troisieme'])
                elif col == 'title':
                    le.fit(['Mr', 'Mrs', 'Miss', 'Master', 'Dr', 'Rev', 'Rare', 'Unknown'])
                elif col == 'deck':
                    le.fit(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'U'])
                elif col == 'ticket_prefix':
                    le.fit(['NUM', 'PC', 'A', 'PP', 'CA', 'STONO', 'SC', 'CASOTON', 'F', 'SC/Paris', 'SCParis', 'C', 'SOPP', 'SW/PP', 'SOP', 'SOTONO2', 'FC', 'WEP', 'STONO2', 'SC/AH', 'LINE', 'Unknown', 'UNK'])
                elif col == 'age_group':
                    le.fit(['enfant', 'ado', 'jeune_adulte', 'adulte', 'senior'])
                
                encoded = le.transform([df[col].values[0]])
            
            X = np.column_stack([X, encoded])
        
        # Normaliser avec le scaler du trainer
        scaler = self.trainer.scalers.get('main')
        if scaler:
            X = scaler.transform(X)
        
        return X
    
    def predict(self, passenger: dict) -> tuple[float, str]:
        """
        Predit la survie d'un passager.
        
        Args:
            passenger: Dict avec les donnees du passager
        
        Returns:
            Tuple (probabilite_survie, interpretation)
        """
        X = self.prepare_input(passenger)
        
        # Prediction
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0]
        
        # Prob de survie
        prob_survive = probability[1]
        
        # Interpretation
        if prob_survive > 0.7:
            interpretation = "TRES BON (>70%)"
        elif prob_survive > 0.5:
            interpretation = "BON (>50%)"
        elif prob_survive > 0.3:
            interpretation = "MOYEN (30-50%)"
        else:
            interpretation = "FAIBLE (<30%)"
        
        return prob_survive, interpretation
    
    def show_profile(self, passenger: dict):
        """Affiche le profil du passager."""
        # Noms complets des ports
        port_names = {
            'S': 'Southampton', 
            'C': 'Cherbourg', 
            'Q': 'Queenstown'
        }
        
        # Noms des classes
        class_names = {
            1: '1ère classe (Luxe)',
            2: '2e classe (Intermédiaire)', 
            3: '3e classe (Économique)'
        }
        
        print("\nPROFIL DU PASSAGER:")
        print("-" * 60)
        print(f"Nom: {passenger['name']}")
        print(f"Sexe: {'Femme' if passenger['Sex'] == 'female' else 'Homme'}")
        print(f"Age: {passenger['Age']} ans")
        print(f"Classe: {class_names.get(passenger['Pclass'], passenger['Pclass'])}")
        print(f"Tarif: {passenger['Fare']} £")
        family_text = "Seul(e)" if passenger['family_size'] == 1 else f"{passenger['family_size']} personnes"
        print(f"Famille: {family_text}")
        print(f"Embarquement: {port_names.get(passenger['Embarked'], passenger['Embarked'])}")
    
    def run_prediction(self):
        """Lance une prediction interactive."""
        try:
            # Saisie
            passenger = self.get_passenger_input()
            
            # Afficher le profil
            self.show_profile(passenger)
            
            # Prediction
            print("\nCALCUL DE LA PREDICTION...")
            prob, interp = self.predict(passenger)
            
            # Resultats
            print("\n" + "="*60)
            print("RESULTAT DE LA PREDICTION")
            print("="*60)
            print(f"\nChances de survie: {prob*100:.1f}%")
            print(f"Evaluation: {interp}\n")
            
            if prob > 0.5:
                print("Le passager a plus de chances de SURVIVRE")
            else:
                print("Le passager a plus de chances de NE PAS SURVIVRE\n")
            
            return prob
        
        except KeyboardInterrupt:
            print("\n\nAnnule par l'utilisateur.")
            return None
        except Exception as e:
            print(f"Erreur: {e}")
            return None


def interactive_menu(trainer):
    """
    Menu interactif pour les predictions.
    
    Args:
        trainer: Instance de TitanicModelTrainer
    """
    predictor = PassengerPredictor(trainer)
    
    while True:
        try:
            choice = input("\n\nVoulez-vous faire une autre prediction? (O/N): ").strip().upper()
            if choice in ['N', 'NON']:
                print("\nFin du programme.")
                break
            elif choice not in ['O', 'OUI', 'Y', 'YES']:
                print("Repondez par O (oui) ou N (non)")
                continue
            
            predictor.run_prediction()
        
        except KeyboardInterrupt:
            print("\n\nProgramme interrompu.")
            break


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    
    from data.data_loader import TitanicDataLoader
    from core.features import FeatureEngineer
    from core.train import TitanicModelTrainer
    
    # Charger et preparer
    loader = TitanicDataLoader("data/Titanic-Dataset.xls")
    df = loader.load_data()
    
    engineer = FeatureEngineer(df)
    df = engineer.create_demographic_features()
    df = engineer.create_passenger_profiles()
    
    # Entrainner les modeles
    trainer = TitanicModelTrainer(df, None)
    trainer.train_models()
        
    # Menu
    predictor = PassengerPredictor(trainer)
    predictor.run_prediction()
    interactive_menu(trainer)
