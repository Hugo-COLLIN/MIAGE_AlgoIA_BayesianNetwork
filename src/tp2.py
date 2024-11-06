import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score

# ------------
# Partie 1
# ------------

# --- Q1.1 ---
# Chargement des données
data = pd.read_csv("data/music_mouv_data.csv")

# Sélectionner les données pertinentes
columns_to_keep = ['familiarity', 'danceability', 'speechiness', 'EDA Phasic Skewness', 'HRV_MaxNN', 'HRV_MinNN',
                   'emotion']
data_filtered = data[columns_to_keep].copy()  # Créer une copie pour éviter les warnings

# Filtrer les lignes pour ne garder que "Joyful activation" et "Tension"
data_filtered = data_filtered[data_filtered['emotion'].isin(['Joyful Activation', 'Tension'])]

# --- Q1.2 ---
def binarize_column(df, column, threshold):
    # Vérifier si la colonne est déjà catégorielle
    if df[column].dtype == 'object' or df[column].dtype.name == 'category':
        return df[column]
    return pd.cut(df[column], bins=[-float('inf'), threshold, float('inf')], labels=['basse', 'haute'])

def trinarize_column(df, column, thresholds):
    # Vérifier si la colonne est déjà catégorielle
    if df[column].dtype == 'object' or df[column].dtype.name == 'category':
        return df[column]
    return pd.cut(df[column], bins=[-float('inf')] + thresholds + [float('inf')], labels=['basse', 'moyenne', 'haute'])

# Appliquer la binarisation aux colonnes spécifiées
data_filtered['danceability'] = binarize_column(data_filtered, 'danceability', 0.5)
data_filtered['speechiness'] = binarize_column(data_filtered, 'speechiness', 0.33)
data_filtered['EDA Phasic Skewness'] = binarize_column(data_filtered, 'EDA Phasic Skewness', 1)
data_filtered['HRV_MaxNN'] = binarize_column(data_filtered, 'HRV_MaxNN', 500)
data_filtered['HRV_MinNN'] = binarize_column(data_filtered, 'HRV_MinNN', 1000)

print(data_filtered)

# --- Q1.3 ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_indices = list(kf.split(data_filtered))

# ------------
# Partie 2
# ------------

# --- Q2.1 ---
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
precisions = []

for train_index, test_index in fold_indices:
    # Séparation des données d'apprentissage et de test
    train_data = data_filtered.iloc[train_index]
    test_data = data_filtered.iloc[test_index]

    # Apprentissage du modèle
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
# Structure améliorée
structure_amelioree = structure + [('familiarity', 'HRV_MaxNN'), ('danceability', 'EDA Phasic Skewness')]
model_ameliore = BayesianNetwork(structure_amelioree)

# Nouvelles données avec seuils modifiés
data_amelioree = data_filtered.copy()
data_amelioree['HRV_MaxNN'] = trinarize_column(data_amelioree, 'HRV_MaxNN', [400, 600])

# Test du modèle amélioré
precisions_ameliorees = []

for train_index, test_index in fold_indices:
    train_data = data_amelioree.iloc[train_index]
    test_data = data_amelioree.iloc[test_index]

    model_ameliore.fit(train_data, estimator=BayesianEstimator, prior_type="BDeu")
    predictions = model_ameliore.predict(test_data.drop('emotion', axis=1))
    precision = precision_score(test_data['emotion'], predictions['emotion'], average='micro')
    precisions_ameliorees.append(precision)

moyenne_precision_amelioree = sum(precisions_ameliorees) / len(precisions_ameliorees)
print(f"Précision moyenne améliorée : {moyenne_precision_amelioree}")
