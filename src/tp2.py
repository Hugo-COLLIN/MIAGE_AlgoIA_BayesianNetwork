import pandas as pd

# ------------
# Partie 1
# ------------

# --- Q1.1 ---
# Chargement des données
data = pd.read_csv("data/music_mouv_data.csv")

# Sélectionner les données pertinentes
columns_to_keep = ['familiarity', 'danceability', 'speechiness', 'EDA Phasic Skewness', 'HRV_MaxNN', 'HRV_MinNN',
                   'emotion']
data_filtered = data[columns_to_keep]

# Filtrer les lignes pour ne garder que "Joyful activation" et "Tension"
data_filtered = data_filtered[data_filtered['emotion'].isin(['Joyful Activation', 'Tension'])]

print(data_filtered)


# --- Q1.2 ---
def binarize_column(df, column, threshold):
    return pd.cut(df[column], bins=[-float('inf'), threshold, float('inf')], labels=['basse', 'haute'])


# Appliquer la binarisation aux colonnes spécifiées
data_filtered['danceability'] = binarize_column(data_filtered, 'danceability', 0.5)
data_filtered['speechiness'] = binarize_column(data_filtered, 'speechiness', 0.33)
data_filtered['EDA Phasic Skewness'] = binarize_column(data_filtered, 'EDA Phasic Skewness', 1)
data_filtered['HRV_MaxNN'] = binarize_column(data_filtered, 'HRV_MaxNN', 500)
data_filtered['HRV_MinNN'] = binarize_column(data_filtered, 'HRV_MinNN', 1000)

# --- Q1.3 ---
from sklearn.model_selection import KFold

# Initialiser KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Créer les indices pour la validation croisée
fold_indices = list(kf.split(data_filtered))

# Les indices peuvent être utilisés plus tard pour l'apprentissage et l'évaluation du modèle


# ------------
# Partie 2
# ------------

# --- Q2.1 ---
from pgmpy.models import BayesianNetwork

# Définition de la structure du réseau
structure = [
    ('familiarity', 'emotion'),
    ('danceability', 'emotion'),
    ('speechiness', 'emotion'),
    ('emotion', 'EDA Phasic Skewness'),
    ('emotion', 'HRV_MaxNN'),
    ('emotion', 'HRV_MinNN')
]

# Création du réseau bayésien
model = BayesianNetwork(structure)

# --- Q2.2 ---
from pgmpy.estimators import BayesianEstimator
from sklearn.metrics import precision_score

precisions = []

for train_index, test_index in fold_indices:
    # Séparation des données d'apprentissage et de test
    train_data = data_filtered.iloc[train_index]
    test_data = data_filtered.iloc[test_index]

    # Apprentissage du modèle
    estimator = BayesianEstimator(model=model, data=train_data)
    model.fit(train_data, estimator=BayesianEstimator, prior_type="BDeu")

    # Prédiction sur les données de test
    predictions = model.predict(test_data.drop('emotion', axis=1))

    # Calcul de la précision micro
    precision = precision_score(test_data['emotion'], predictions['emotion'], average='micro')
    precisions.append(precision)

# Calcul de la précision moyenne
moyenne_precision = sum(precisions) / len(precisions)
print(f"Précision moyenne : {moyenne_precision}")

# --- Q2.3 ---
# # > Ajout de nouvelles variables
# structure_amelioree = structure + [('familiarity', 'HRV_MaxNN'), ('danceability', 'EDA Phasic Skewness')]
# model_ameliore = BayesianNetwork(structure_amelioree)
#
# # > Modif des seuils pour la binarisation
# data_filtered['danceability'] = binarize_column(data_filtered, 'danceability', 0.6)  # Nouveau seuil
# data_filtered['speechiness'] = binarize_column(data_filtered, 'speechiness', 0.4)  # Nouveau seuil
#
# # > Augmentation du nombre de valeurs possibles pour certaines valeurs
# def trinarize_column(df, column, thresholds):
#     return pd.cut(df[column], bins=[-float('inf')] + thresholds + [float('inf')], labels=['basse', 'moyenne', 'haute'])
#
# data_filtered['HRV_MaxNN'] = trinarize_column(data_filtered, 'HRV_MaxNN', [400, 600])
#

