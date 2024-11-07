import pandas as pd
from sklearn.model_selection import KFold

################
# Partie 1
################

# 1. Chargement et filtrage des données
data = pd.read_csv("data/music_mouv_data.csv")
colonnes = ["EDA Phasic Number of Peaks", "EDA Phasic Skewness", "HRV_MaxNN", "HRV_MinNN",
            "danceability", "speechiness", "valence", "familiarity", "emotion"]
data_filtree = data[data["emotion"].isin(["Joyful Activation", "Tension"])][colonnes]

# 2. Transformation des variables quantitatives en qualitatives
def binariser(df, colonne, seuil):
    return (df[colonne] > seuil).map({True: f"{colonne}_{seuil}+", False: f"{colonne}_{seuil}-"})

data_filtree["EDA Phasic Number of Peaks"] = binariser(data_filtree, "EDA Phasic Number of Peaks", 6)
data_filtree["EDA Phasic Skewness"] = binariser(data_filtree, "EDA Phasic Skewness", 0.85)
data_filtree["HRV_MaxNN"] = binariser(data_filtree, "HRV_MaxNN", 500)
data_filtree["HRV_MinNN"] = binariser(data_filtree, "HRV_MinNN", 1000)
data_filtree["danceability"] = binariser(data_filtree, "danceability", 0.5)
data_filtree["speechiness"] = binariser(data_filtree, "speechiness", 0.33)
data_filtree["valence"] = binariser(data_filtree, "valence", 0.9)

# Conversion en transactions
transactions = list(data_filtree.apply(lambda x: set(x.dropna().astype(str)), axis=1))

# 3. Séparation en données d'apprentissage et d'évaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
folds = list(kf.split(transactions))

print("Exemple de transaction :")
print(list(transactions)[0])
print(f"\nNombre de transactions : {len(transactions)}")
print(f"Nombre de folds : {len(folds)}")


################
# Partie 2
################

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from sklearn.metrics import precision_score

# 1. Extraction des itemsets fréquents
def extraire_itemsets(transactions):
    df = pd.DataFrame([[item in transaction for item in set.union(*transactions)] for transaction in transactions])
    return apriori(df, min_support=0.01, use_colnames=True)

# 2. Extraction des règles d'association
def extraire_regles(frequent_itemsets):
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    return rules[rules['consequents'].apply(lambda x: any('emotion_' in item for item in x))]

# 3. Application des règles
def apply_rules(rules, transactions):
    predictions = []
    for transaction in transactions:
        matching_rules = rules[rules['antecedents'].apply(lambda x: x.issubset(transaction))]
        if not matching_rules.empty:
            best_rule = matching_rules.loc[matching_rules['confidence'].idxmax()]
            prediction = list(best_rule['consequents'])[0]
        else:
            prediction = 'emotion_Joyful Activation'  # Prédiction par défaut
        predictions.append(prediction)
    return predictions

# 4. Validation croisée
precisions = []
for train_index, test_index in folds:
    train_transactions = [transactions[i] for i in train_index]
    test_transactions = [transactions[i] for i in test_index]

    frequent_itemsets = extraire_itemsets(train_transactions)
    rules = extraire_regles(frequent_itemsets)

    predictions = apply_rules(rules, test_transactions)
    true_emotions = [list(t)[0] for t in test_transactions if any('emotion_' in item for item in t)]

    precision = precision_score(true_emotions, predictions, average='micro')
    precisions.append(precision)

precision_moyenne = sum(precisions) / len(precisions)
print(f"Précision moyenne : {precision_moyenne}")

# 5. Amélioration du modèle
def ternariser(df, colonne, seuils):
    return pd.cut(df[colonne], bins=[-float('inf')] + seuils + [float('inf')],
                  labels=[f"{colonne}_bas", f"{colonne}_moyen", f"{colonne}_haut"])

# Exemple d'amélioration : ternarisation de HRV_MaxNN
data_amelioree = data_filtree.copy()
data_amelioree["HRV_MaxNN"] = ternariser(data_amelioree, "HRV_MaxNN", [400, 600])

# Conversion en transactions
transactions_ameliorees = data_amelioree.apply(lambda x: set(x.dropna().astype(str)), axis=1)

# Répétition de la validation croisée avec les données améliorées
precisions_ameliorees = []
for train_index, test_index in folds:
    train_transactions = [transactions_ameliorees[i] for i in train_index]
    test_transactions = [transactions_ameliorees[i] for i in test_index]

    frequent_itemsets = extraire_itemsets(train_transactions)
    rules = extraire_regles(frequent_itemsets)

    predictions = apply_rules(rules, test_transactions)
    true_emotions = [list(t)[0] for t in test_transactions if any('emotion_' in item for item in t)]

    precision = precision_score(true_emotions, predictions, average='micro')
    precisions_ameliorees.append(precision)

precision_moyenne_amelioree = sum(precisions_ameliorees) / len(precisions_ameliorees)
print(f"Précision moyenne améliorée : {precision_moyenne_amelioree}")
