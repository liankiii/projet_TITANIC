"""
Module d'ingénierie des features pour le dataset Titanic.
Crée de nouvelles variables pour identifier les groupes sociodémographiques.
"""
import pandas as pd
import numpy as np
from typing import Tuple


class FeatureEngineer:
    """Ingénierie des features pour l'analyse des groupes Titanic."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialise l'ingénieur de features.
        
        Args:
            df: DataFrame Titanic
        """
        self.df = df.copy()
        self.engineered_df = None
    
    def create_demographic_features(self) -> pd.DataFrame:
        """
        Crée les features démographiques clés:
        - age_group, is_child, is_woman, is_man
        - family_size, is_alone, has_children, is_mother
        - fare_per_person
        - ticket_class, title, deck, ticket_prefix
        
        Returns:
            DataFrame avec features additionnelles
        """
        df = self.df.copy()

        # Imputation de l'âge par groupe (Sex, Pclass) si Age manquant
        if 'Age' in df.columns:
            df['Age'] = df.groupby(['Sex', 'Pclass'])['Age'].transform(lambda g: g.fillna(g.median()))
            df['Age'] = df['Age'].fillna(df['Age'].median())
        
        # Groupes d'âge
        df['age_group'] = pd.cut(
            df['Age'],
            bins=[0, 12, 18, 35, 60, 150],
            labels=['enfant', 'ado', 'jeune_adulte', 'adulte', 'senior'],
            include_lowest=True
        )
        df['is_child'] = (df['Age'] < 12).astype(int)
        
        # Genre
        df['is_woman'] = (df['Sex'] == 'female').astype(int)
        df['is_man'] = (df['Sex'] == 'male').astype(int)
        
        # Voyageait seul
        df['family_size'] = df['SibSp'] + df['Parch'] + 1
        df['is_alone'] = (df['family_size'] == 1).astype(int)
        df['has_children'] = (df['Parch'] > 0).astype(int)
        df['fare_per_person'] = (df['Fare'] / df['family_size']).replace([np.inf, -np.inf], np.nan)
        df['fare_per_person'] = df['fare_per_person'].fillna(df['fare_per_person'].median())
        
        # Classe de billet
        df['ticket_class'] = df['Pclass'].map({
            1: 'premiere',
            2: 'deuxieme',
            3: 'troisieme'
        })

        # Extraction du titre depuis Name
        if 'Name' in df.columns:
            df['title'] = df['Name'].str.extract(' ([A-Za-z]+)\.')
            title_map = {
                'Mlle': 'Miss', 'Mme': 'Mrs', 'Ms': 'Miss',
                'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare', 'Col': 'Rare',
                'Don': 'Rare', 'Dr': 'Rare', 'Major': 'Rare', 'Rev': 'Rare', 'Sir': 'Rare',
                'Jonkheer': 'Rare', 'Dona': 'Rare'
            }
            df['title'] = df['title'].replace(title_map)
            df['title'] = df['title'].fillna('Unknown')
        else:
            df['title'] = 'Unknown'

        # Présence de cabine / pont
        if 'Cabin' in df.columns:
            df['deck'] = df['Cabin'].str[0].fillna('U')
        else:
            df['deck'] = 'U'

        # Préfixe de ticket
        if 'Ticket' in df.columns:
            df['ticket_prefix'] = df['Ticket'].str.replace('[./]', ' ').str.split().str[0]
            df['ticket_prefix'] = df['ticket_prefix'].where(df['ticket_prefix'].str.len() > 1, 'NUM')
        else:
            df['ticket_prefix'] = 'UNK'

        # Déterminer si la passagère est mère (female, >18, Parch>0, pas Miss)
        df['is_mother'] = ((df['Sex'] == 'female') & (df['Age'] >= 18) & (df['Parch'] > 0) & (df['title'] != 'Miss')).astype(int)
        
        self.engineered_df = df
        return df
    
    def create_passenger_profiles(self) -> pd.DataFrame:
        """
        Crée des profils de passagers basés sur les caractéristiques clés.
        Groupes identifiés:
        - young_men_alone: jeunes hommes voyageant seuls (classe basse)
        - women_children: femmes et enfants (toutes classes)
        - wealthy_men: hommes de première classe
        - third_class_families: familles en troisième classe
        - etc.
        
        Returns:
            DataFrame avec colonne 'profile'
        """
        if self.engineered_df is None:
            self.create_demographic_features()
        
        df = self.engineered_df.copy()
        
        def assign_profile(row):
            """Assigne un profil sociodémographique à chaque passager avec distinctions de classe pour femmes et enfants."""
            age = row['Age']
            pclass = row['Pclass']
            is_alone = row['is_alone']
            family_size = row['family_size']
            sex = row['Sex']
            title = row.get('title', '')

            # Priorité aux mères
            if row.get('is_mother', 0) == 1:
                return 'mere'

            # Enfants (tous sexes) -> mêmes distinctions par classe
            if pd.notna(age) and age < 12:
                if pclass == 1:
                    return 'enfant_riche'
                elif pclass == 2:
                    return 'enfant_classe_moyenne_seul' if is_alone == 1 else 'enfant_classe_moyenne_famille'
                else:
                    return 'enfant_populaire_famille' if family_size > 1 else 'enfant_populaire_seul'

            # Femmes
            if sex == 'female':
            # Femmes de première classe (mêmes paliers d'âge que pour les hommes)
                if pclass == 1:
                    if pd.notna(age) and age < 30:
                        return 'femme_riche_jeune_ado'
                    elif pd.notna(age) and age < 50:
                        return 'femme_riche_jeune'
                    else:
                        return 'femme_riche_age'
                # Femmes de deuxième classe: seul vs famille
                elif pclass == 2:
                    return 'femme_classe_moyenne_seule' if is_alone == 1 else 'femme_classe_moyenne_famille'
                # Femmes de troisième classe: famille > jeune seule > seule
                else:
                    if family_size > 1:
                        return 'femme_populaire_famille'
                    elif pd.notna(age) and age < 25:
                        return 'jeune_femme_populaire'
                    else:
                        return 'femme_populaire_seule'
            # Hommes
            else:
                # Jeunes hommes seuls (classe basse)
                if (pd.notna(age) and age < 35 and is_alone == 1 and pclass == 3):
                    return 'jeune_homme_seul'
                # Hommes de première classe
                elif pclass == 1:
                    if pd.notna(age) and age < 30:
                        return 'homme_riche_jeune_ado'
                    elif pd.notna(age) and age < 50:
                        return 'homme_riche_jeune'
                    else:
                        return 'homme_riche_age'
                # Hommes de deuxième classe
                elif pclass == 2:
                    return 'homme_classe_moyenne_seul' if is_alone == 1 else 'homme_classe_moyenne_famille'
                # Hommes de troisième classe
                else:
                    if family_size > 1:
                        return 'homme_populaire_famille'
                    elif pd.notna(age) and age < 25:
                        return 'jeune_homme_populaire'
                    else:
                        return 'homme_populaire_seul'
        df['profile'] = df.apply(assign_profile, axis=1)
        self.engineered_df = df
        return df
    
    def get_engineered_df(self) -> pd.DataFrame:
        """Retourne le DataFrame avec toutes les features."""
        if self.engineered_df is None:
            self.create_demographic_features()
            self.create_passenger_profiles()
        return self.engineered_df
    
    def summarize_profiles(self) -> pd.DataFrame:
        """
        Résumé statistique des profils (survie par groupe).
        
        Returns:
            DataFrame avec taux de survie par profil
        """
        df = self.get_engineered_df()
        
        summary = df.groupby('profile').agg({
            'Survived': ['sum', 'count', 'mean'],
            'Age': 'mean',
            'Fare': 'mean'
        }).round(3)
        
        summary.columns = ['nb_survivants', 'total', 'taux_survie', 
                          'age_moyen', 'tarif_moyen']
        summary = summary.sort_values('taux_survie', ascending=False)
        
        return summary


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))
    from data_loader import TitanicDataLoader
    
    loader = TitanicDataLoader("data/Titanic-Dataset.xls")
    df = loader.load_data()
    
    engineer = FeatureEngineer(df)
    df = engineer.create_demographic_features()
    df = engineer.create_passenger_profiles()
    
    print("\nProfils de passagers identifiés:")
    summary = engineer.summarize_profiles()
    print(summary)
    
    print("\nDistribution des profils:")
    print(df['profile'].value_counts())
