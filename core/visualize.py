"""
Module de visualisation pour l'analyse de la survie Titanic.
Génère des graphiques pour explorer les patterns de survie par groupe.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple
from sklearn.decomposition import PCA


class TitanicVisualizer:
    """Générateur de visualisations pour l'analyse Titanic."""
    
    def __init__(self, df: pd.DataFrame, output_dir: str = "outputs"):
        """
        Initialise l'outil de visualisation.
        
        Args:
            df: DataFrame avec les features et colonnes de survie
            output_dir: Dossier pour sauvegarder les figures
        """
        self.df = df.copy()
        cols_to_drop = ['PassengerId', 'Ticket', 'Cabin', 'Name']
        self.df = self.df.drop(columns=[c for c in cols_to_drop if c in self.df.columns])
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        sns.set_style("whitegrid")
        sns.set_context("talk")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.formatter.use_locale'] = True
    
    def plot_survival_by_profile(self, figsize: Tuple[int, int] = (14, 6)):
        """
        Visualise le taux de survie par profil de passager.
        
        Args:
            figsize: Taille de la figure
        """
        if 'profile' not in self.df.columns or 'Survived' not in self.df.columns:
            print("Colonnes 'profile' ou 'Survived' manquantes")
            return
        
        profile_stats = self.df.groupby('profile', observed=True)['Survived'].agg(['sum', 'count', 'mean'])
        profile_stats = profile_stats.sort_values('mean', ascending=False)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Taux de survie par profil
        colors = ['green' if x > 0.5 else 'red' for x in profile_stats['mean']]
        profile_stats['mean'].plot(kind='barh', ax=ax1, color=colors, alpha=0.7)
        ax1.set_xlabel('Taux de survie')
        ax1.set_title('Survie par profil de passager')
        ax1.set_xlim(0, 1)
        for i, v in enumerate(profile_stats['mean']):
            ax1.text(v + 0.02 if v < 0.9 else v - 0.05, i, f"{v:.0%}", va='center')
        
        # Nombre de passagers par profil
        profile_stats['count'].plot(kind='barh', ax=ax2, color='skyblue', alpha=0.7)
        ax2.set_xlabel('Nombre de passagers')
        ax2.set_title('Distribution des profils')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/01_survie_par_profil.png", dpi=150, bbox_inches='tight')
        print("Visualisation sauvegardée: 01_survie_par_profil.png")
        plt.close()
    
    def plot_survival_by_demographic(self, figsize: Tuple[int, int] = (14, 10)):
        """
        Visualise la survie selon les caractéristiques démographiques.
        
        Args:
            figsize: Taille de la figure
        """
        if 'Survived' not in self.df.columns:
            print("Colonne 'Survived' manquante")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Survie par genre
        if 'Sex' in self.df.columns:
            sex_survival = self.df.groupby('Sex')['Survived'].mean()
            sex_survival.plot(kind='bar', ax=axes[0, 0], color=['pink', 'lightblue'])
            axes[0, 0].set_title('Survie par genre')
            axes[0, 0].set_ylabel('Taux de survie')
            axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=0)
            axes[0, 0].set_ylim(0, 1)
            for i, v in enumerate(sex_survival.values):
                axes[0, 0].text(i, v + 0.02, f"{v:.0%}", ha='center')
        
        # Survie par classe
        if 'Pclass' in self.df.columns:
            class_survival = self.df.groupby('Pclass')['Survived'].mean()
            class_survival.plot(kind='bar', ax=axes[0, 1], color=['gold', 'silver', 'orange'])
            axes[0, 1].set_title('Survie par classe de billet')
            axes[0, 1].set_ylabel('Taux de survie')
            axes[0, 1].set_xticklabels(['1ère classe', '2e classe', '3e classe'], rotation=0)
            axes[0, 1].set_ylim(0, 1)
            for i, v in enumerate(class_survival.values):
                axes[0, 1].text(i, v + 0.02, f"{v:.0%}", ha='center')
        
        # Survie par groupe d'âge
        if 'age_group' in self.df.columns:
            age_survival = self.df.groupby('age_group', observed=True)['Survived'].agg(['sum', 'count', 'mean'])
            age_survival['mean'].plot(kind='bar', ax=axes[1, 0], color='teal', alpha=0.7)
            axes[1, 0].set_title('Survie par groupe d\'âge')
            axes[1, 0].set_ylabel('Taux de survie')
            axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45)
            axes[1, 0].set_ylim(0, 1)
            for i, v in enumerate(age_survival['mean'].values):
                axes[1, 0].text(i, v + 0.02, f"{v:.0%}", ha='center', rotation=0)
        
        # Survie selon solitude
        if 'is_alone' in self.df.columns:
            alone_survival = self.df.groupby('is_alone')['Survived'].mean()
            alone_survival.plot(kind='bar', ax=axes[1, 1], color=['coral', 'lightgreen'])
            axes[1, 1].set_title('Survie: seul vs en famille')
            axes[1, 1].set_ylabel('Taux de survie')
            axes[1, 1].set_xticklabels(['En famille', 'Seul'], rotation=0)
            axes[1, 1].set_ylim(0, 1)
            for i, v in enumerate(alone_survival.values):
                axes[1, 1].text(i, v + 0.02, f"{v:.0%}", ha='center')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/02_survie_demographic.png", dpi=150, bbox_inches='tight')
        print("Visualisation sauvegardée: 02_survie_demographic.png")
        plt.close()
    
    def plot_fare_distribution(self, figsize: Tuple[int, int] = (16, 10)):
        """
        Analyse détaillée du tarif et son impact sur la survie.
        
        Args:
            figsize: Taille de la figure
        """
        if 'Fare' not in self.df.columns or 'Survived' not in self.df.columns:
            print("Colonnes 'Fare' ou 'Survived' manquantes")
            return
        
        # Créer des tranches de tarif plus parlantes
        df_clean = self.df[self.df['Fare'] > 0].copy()
        df_clean.loc[:, 'Fare'] = df_clean['Fare'].clip(upper=np.percentile(df_clean['Fare'], 99))
        
        # Définir des tranches de tarif logiques
        fare_bins = [0, 10, 20, 30, 50, 100, float('inf')]
        fare_labels = ['0-10£', '10-20£', '20-30£', '30-50£', '50-100£', '100+£']
        df_clean['fare_range'] = pd.cut(df_clean['Fare'], bins=fare_bins, labels=fare_labels, include_lowest=True)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Taux de survie par tranche de tarif
        fare_survival = df_clean.groupby('fare_range', observed=True)['Survived'].agg(['mean', 'count'])
        fare_survival['mean'].plot(kind='bar', ax=axes[0, 0], color='steelblue', alpha=0.8)
        axes[0, 0].set_title('Taux de survie par tranche de tarif', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Taux de survie')
        axes[0, 0].set_xlabel('Tranche de tarif (£)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 1)
        
        # Ajouter les pourcentages sur les barres
        for i, v in enumerate(fare_survival['mean']):
            axes[0, 0].text(i, v + 0.02, f'{v:.0%}', ha='center', va='bottom')
        
        # 2. Nombre de passagers par tranche
        fare_survival['count'].plot(kind='bar', ax=axes[0, 1], color='orange', alpha=0.8)
        axes[0, 1].set_title('Nombre de passagers par tranche', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Nombre de passagers')
        axes[0, 1].set_xlabel('Tranche de tarif (£)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Ajouter les nombres sur les barres
        for i, v in enumerate(fare_survival['count']):
            axes[0, 1].text(i, v + 5, f'{v}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Distribution détaillée par classe de billet
        if 'Pclass' in df_clean.columns:
            class_colors = {'1': 'gold', '2': 'silver', '3': 'chocolate'}
            for pclass in sorted(df_clean['Pclass'].unique()):
                class_data = df_clean[df_clean['Pclass'] == pclass]
                survivors = class_data[class_data['Survived'] == 1]['Fare']
                non_survivors = class_data[class_data['Survived'] == 0]['Fare']
                axes[1, 0].hist(survivors, bins=20, alpha=0.6,
                                label=f'Classe {pclass} - Survivants',
                                color=class_colors.get(str(pclass), 'gray'))
                axes[1, 0].hist(non_survivors, bins=20, alpha=0.4,
                                label=f'Classe {pclass} - Décédés',
                                color=class_colors.get(str(pclass), 'gray'),
                                linestyle='--')
            
            axes[1, 0].set_title('Distribution des tarifs par classe et survie', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Tarif (£)')
            axes[1, 0].set_ylabel('Nombre de passagers')
            axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_xlim(0, df_clean['Fare'].quantile(0.99))
        
        # 4. Analyse statistique détaillée
        stats_data = []
        for fare_range in fare_labels:
            range_data = df_clean[df_clean['fare_range'] == fare_range]
            if len(range_data) > 0:
                survival_rate = range_data['Survived'].mean()
                count = len(range_data)
                avg_fare = range_data['Fare'].mean()
                stats_data.append({
                    'Tranche': fare_range,
                    'Taux survie': f'{survival_rate:.1%}',
                    'Passagers': count,
                    'Tarif moyen': f'{avg_fare:.1f}£'
                })
        
        # Créer un tableau récapitulatif
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        
        table_data = [[row['Tranche'], row['Taux survie'], str(row['Passagers']), row['Tarif moyen']] 
                      for row in stats_data]
        
        table = axes[1, 1].table(cellText=table_data,
                                colLabels=['Tranche tarif', 'Taux survie', 'Nb passagers', 'Tarif moyen'],
                                cellLoc='center',
                                loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Colorier les cellules selon le taux de survie
        for i in range(len(stats_data)):
            survival_rate = df_clean[df_clean['fare_range'] == stats_data[i]['Tranche']]['Survived'].mean()
            if survival_rate > 0.6:
                color = 'lightgreen'
            elif survival_rate > 0.4:
                color = 'lightyellow' 
            else:
                color = 'lightcoral'
            table[(i+1, 1)].set_facecolor(color)
        
        axes[1, 1].set_title('Analyse statistique par tranche', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/03_fare_analysis.png", dpi=150, bbox_inches='tight')
        print("Analyse des tarifs sauvegardée: 03_fare_analysis.png")
        plt.close()
    
    def plot_heatmap_correlation(self, figsize: Tuple[int, int] = (10, 8)):
        """
        Heatmap de corrélation entre variables numériques et survie.
        
        Args:
            figsize: Taille de la figure
        """
        if 'Survived' not in self.df.columns:
            print("Colonne 'Survived' manquante")
            return
        
        # Sélectionner colonnes numériques
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.df[numeric_cols].corr().clip(-1, 1)
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, ax=ax, cbar_kws={'label': 'Corrélation'})
        ax.set_title('Matrice de corrélation - Variables numériques')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/04_correlation_heatmap.png", dpi=150, bbox_inches='tight')
        print("Visualisation sauvegardée: 04_correlation_heatmap.png")
        plt.close()

    def plot_survival_factors_analysis(self, figsize: Tuple[int, int] = (16, 12)):
        """
        Analyse croisée des facteurs de survie (âge, sexe, classe, tarif).
        """
        if not all(col in self.df.columns for col in ['Age', 'Sex', 'Pclass', 'Fare', 'Survived']):
            print("Colonnes nécessaires manquantes pour l'analyse croisée")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Survie par âge et sexe
        df_clean = self.df.dropna(subset=['Age'])
        age_bins = [0, 12, 18, 30, 50, 80]
        age_labels = ['Enfant', 'Ado', 'Jeune', 'Adulte', 'Senior']
        df_clean = df_clean.copy()
        df_clean['age_category'] = pd.cut(df_clean['Age'], bins=age_bins, labels=age_labels, include_lowest=True)
        
        # Pivot table pour age/sexe
        age_sex_survival = df_clean.pivot_table(values='Survived', 
                               index='age_category', 
                               columns='Sex', 
                               aggfunc='mean', observed=True)
        
        age_sex_survival.plot(kind='bar', ax=axes[0, 0], color=['pink', 'lightblue'], alpha=0.8)
        axes[0, 0].set_title('Taux de survie par âge et sexe', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Taux de survie')
        axes[0, 0].set_xlabel('Catégorie d\'âge')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].legend(['Femme', 'Homme'])
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Survie par classe et sexe avec effectifs
        class_sex_survival = self.df.pivot_table(values='Survived', 
                            index='Pclass', 
                            columns='Sex', 
                            aggfunc='mean', observed=True)
        
        class_sex_survival.plot(kind='bar', ax=axes[0, 1], color=['pink', 'lightblue'], alpha=0.8)
        axes[0, 1].set_title('Taux de survie par classe et sexe', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Taux de survie')
        axes[0, 1].set_xlabel('Classe de billet')
        axes[0, 1].set_xticklabels(['1ère classe', '2e classe', '3e classe'], rotation=0)
        axes[0, 1].legend(['Femme', 'Homme'])
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Heatmap survie par tarif et âge
        df_fare_age = self.df[(self.df['Fare'] > 0) & (self.df['Age'].notna())].copy()
        
        fare_bins = [0, 20, 50, 100, float('inf')]
        fare_labels = ['Économique', 'Moyen', 'Élevé', 'Luxe']
        df_fare_age['fare_category'] = pd.cut(df_fare_age['Fare'], bins=fare_bins, labels=fare_labels)
        
        age_bins2 = [0, 15, 30, 50, 80]
        age_labels2 = ['0-15', '15-30', '30-50', '50+']
        df_fare_age['age_bracket'] = pd.cut(df_fare_age['Age'], bins=age_bins2, labels=age_labels2)
        
        heatmap_data = df_fare_age.pivot_table(values='Survived', 
                              index='age_bracket', 
                              columns='fare_category', 
                              aggfunc='mean', observed=True)
        
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', 
                    ax=axes[1, 0], cbar_kws={'label': 'Taux de survie'})
        axes[1, 0].set_title('Survie par âge et niveau de tarif', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Tranche d\'âge')
        axes[1, 0].set_xlabel('Niveau de tarif')
        
        # 4. Impact de la taille de famille
        if 'family_size' in self.df.columns:
            family_survival = self.df.groupby('family_size')['Survived'].agg(['mean', 'count'])
            
            # Graphique en barres
            bars = axes[1, 1].bar(family_survival.index, family_survival['mean'], 
                                 color='teal', alpha=0.8)
            axes[1, 1].set_title('Taux de survie selon la taille de la famille', fontsize=12, fontweight='bold')
            axes[1, 1].set_ylabel('Taux de survie')
            axes[1, 1].set_xlabel('Taille de la famille')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Ajouter les effectifs sur les barres
            for i, (bar, count) in enumerate(zip(bars, family_survival['count'])):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{height:.1%}\n(n={count})', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/06_survival_factors_analysis.png", dpi=150, bbox_inches='tight')
        print("Analyse des facteurs sauvegardée: 06_survival_factors_analysis.png")
        plt.close()

    def plot_clusters(self, X: np.ndarray, clusters: np.ndarray, figsize: Tuple[int, int] = (14, 6)):
        """
        Visualise les clusters en 2D avec PCA.
        
        Args:
            X: Données pour la réduction PCA
            clusters: Assignations de clusters
            figsize: Taille de la figure
        """
        if len(X) != len(clusters):
            print("Erreur: taille de X et clusters incompatible")
            return
        
        # Réduction PCA à 2 composantes pour visualisation
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Graphique 1: Clusters en 2D (PCA)
        scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, 
                                 cmap='viridis', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var.)')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var.)')
        axes[0].set_title('Visualisation des Clusters (PCA 2D)', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=axes[0])
        cbar.set_label('Cluster', fontsize=10)
        
        # Graphique 2: Distribution des clusters
        unique_clusters, counts = np.unique(clusters, return_counts=True)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_clusters)))
        bars = axes[1].bar([f'C{i}' for i in unique_clusters], counts, color=colors, alpha=0.8, edgecolor='black')
        axes[1].set_title('Distribution des Clusters', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Nombre de samples')
        axes[1].set_xlabel('Cluster')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Ajouter les effectifs sur les barres
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{count}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/07_clusters_visualization.png", dpi=150, bbox_inches='tight')
        print("Visualisation des clusters sauvegardée: 07_clusters_visualization.png")
        plt.close()

    def plot_clusters_with_survival(self, X: np.ndarray, clusters: np.ndarray, 
                                   survived: np.ndarray, figsize: Tuple[int, int] = (14, 6)):
        """
        Visualise les clusters en 2D avec informations de survie.
        
        Args:
            X: Données pour la réduction PCA
            clusters: Assignations de clusters
            survived: Informations de survie (0 ou 1)
            figsize: Taille de la figure
        """
        if len(X) != len(clusters) or len(X) != len(survived):
            print("Erreur: tailles incompatibles")
            return
        
        # Réduction PCA à 2 composantes
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Graphique 1: Clusters avec survie en couleur
        for cluster in np.unique(clusters):
            mask = clusters == cluster
            axes[0].scatter(X_pca[mask & (survived == 0), 0], 
                          X_pca[mask & (survived == 0), 1],
                          marker='x', s=100, label=f'C{cluster} (Mort)' if cluster == 0 else '', alpha=0.6)
            axes[0].scatter(X_pca[mask & (survived == 1), 0],
                          X_pca[mask & (survived == 1), 1],
                          marker='o', s=100, label=f'C{cluster} (Survécu)' if cluster == 0 else '', alpha=0.6)
        
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var.)')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var.)')
        axes[0].set_title('Clusters colorés par Survie\n(× = Mort, ○ = Survécu)', 
                         fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Graphique 2: Taux de survie par cluster
        survival_by_cluster = []
        cluster_sizes = []
        for cluster in np.unique(clusters):
            mask = clusters == cluster
            survival_rate = survived[mask].mean()
            cluster_size = mask.sum()
            survival_by_cluster.append(survival_rate)
            cluster_sizes.append(cluster_size)
        
        colors = ['red' if sr < 0.5 else 'green' for sr in survival_by_cluster]
        bars = axes[1].bar([f'C{i}' for i in np.unique(clusters)], survival_by_cluster, 
                          color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        axes[1].axhline(y=survived.mean(), color='black', linestyle='--', linewidth=2, label='Taux global')
        axes[1].set_title('Taux de Survie par Cluster', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Taux de survie')
        axes[1].set_xlabel('Cluster')
        axes[1].set_ylim([0, 1])
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].legend()
        
        # Ajouter les pourcentages et effectifs
        for bar, rate, size in zip(bars, survival_by_cluster, cluster_sizes):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{rate:.1%}\n(n={size})', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/08_clusters_survival.png", dpi=150, bbox_inches='tight')
        print("Analyse clusters-survie sauvegardée: 08_clusters_survival.png")
        plt.close()

    def generate_all_plots(self):
        """Génère toutes les visualisations."""
        print("\nGénération des visualisations :")
        print(f"Dossier de sortie: {self.output_dir}/")
        
        if 'Survived' in self.df.columns:
            self.plot_survival_by_demographic()
            self.plot_fare_distribution()
            self.plot_heatmap_correlation()
            self.plot_survival_factors_analysis()
        
        if 'profile' in self.df.columns:
            self.plot_survival_by_profile()
        
        print("Toutes les visualisations générées!")
        print("  - 01_survie_par_profil.png")
        print("  - 02_survie_demographic.png") 
        print("  - 03_fare_analysis.png")
        print("  - 04_correlation_heatmap.png")
        print("  - 06_survival_factors_analysis.png")
        print("  - 07_clusters_visualization.png (numérotation: 0-4)")
        print("  - 08_clusters_survival.png (numérotation: 0-4)")


if __name__ == "__main__":
    # Import local
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))
    from data_loader import TitanicDataLoader
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from features import FeatureEngineer
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    # Charger et préparer les données
    loader = TitanicDataLoader("data/Titanic-Dataset.xls")
    df = loader.load_data()
    
    engineer = FeatureEngineer(df)
    df = engineer.create_demographic_features()
    df = engineer.create_passenger_profiles()
    
    # Générer les visualisations
    visualizer = TitanicVisualizer(df)
    visualizer.generate_all_plots()
    
    # Générer aussi les visualisations des clusters
    print("\n" + "="*70)
    print("GÉNÉRATION DES VISUALISATIONS CLUSTERS")
    print("="*70)
    
    # Préparation des données pour clustering
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from encoders import EncoderManager
    from config import FeatureConfig
    
    config = FeatureConfig()
    encoder_manager = EncoderManager()
    
    # Prétraitement
    X = encoder_manager.preprocess_features(
        df,
        config.numeric_cols,
        config.categorical_cols,
        fit=True
    )
    y = df['Survived'].values
    
    # Clustering
    pca = PCA(n_components=4, random_state=42)
    X_pca = pca.fit_transform(X)
    
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_pca)
    
    silhouette = silhouette_score(X_pca, clusters)
    print(f"Clustering: 5 clusters, silhouette={silhouette:.3f}")
    
    # Générer les visualisations des clusters
    visualizer.plot_clusters(X, clusters)
    visualizer.plot_clusters_with_survival(X, clusters, y)
    
    # Afficher les statistiques RÉELLES des clusters
    print("\n" + "="*70)
    print("STATISTIQUES RÉELLES DES CLUSTERS:")
    print("="*70)
    for cluster in np.unique(clusters):
        mask = clusters == cluster
        size = mask.sum()
        survivors = y[mask].sum()
        survival_rate = y[mask].mean()
        avg_fare = df[mask]['Fare'].mean() if 'Fare' in df.columns else 0
        avg_age = df[mask]['Age'].mean() if 'Age' in df.columns else 0
        print(f"Cluster {cluster}: {size:3d} passagers | {survivors:2.0f} survécu(s) | Taux: {survival_rate:.1%} | Âge moy: {avg_age:5.1f} | Tarif moy: {avg_fare:6.1f}£")
