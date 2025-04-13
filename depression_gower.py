import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain
import numpy as np
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib.lines import Line2D
import gower

os.environ["LOKY_MAX_CPU_COUNT"] = "6" 

# étape 1: Chargement et nettoyage des données
data = pd.read_csv("final_depression_dataset_1.csv")
data.drop(columns=["Name", "City", "Degree", "Profession"], inplace=True)

# variables catégorielles mapping
sommeil_mapping = {"Less than 5 hours": 0, "5-6 hours": 1, "7-8 hours": 2, "More than 8 hours": 3}
repas_mapping = {"Unhealthy": 0, "Moderate": 1, "Healthy": 2}
yes_no_mapping = {"Yes": 1, "No": 0}

data["Sleep Duration"] = data["Sleep Duration"].map(sommeil_mapping)
data["Dietary Habits"] = data["Dietary Habits"].map(repas_mapping)
data["Have you ever had suicidal thoughts ?"] = data["Have you ever had suicidal thoughts ?"].map(yes_no_mapping)
data["Family History of Mental Illness"] = data["Family History of Mental Illness"].map(yes_no_mapping)

# Création total Pressure et Total Satisfaction
required_cols = ["Academic Pressure", "Work Pressure", "Study Satisfaction", "Job Satisfaction"]
for col in required_cols:
    print(f"{col} — existe : {col in data.columns} — NaNs : {data[col].isnull().sum() if col in data.columns else 'absent'}")

data["Total Pressure"] = data[["Academic Pressure", "Work Pressure"]].sum(axis=1, skipna=True)
data.loc[data[["Academic Pressure", "Work Pressure"]].isnull().all(axis=1), "Total Pressure"] = np.nan

data["Total Satisfaction"] = data[["Study Satisfaction", "Job Satisfaction"]].sum(axis=1, skipna=True)
data.loc[data[["Study Satisfaction", "Job Satisfaction"]].isnull().all(axis=1), "Total Satisfaction"] = np.nan

data.drop(columns=["Academic Pressure", "Work Pressure", "Study Satisfaction", "Job Satisfaction"], inplace=True)


# Remplissage des valeurs manquantes avec la médiane
numeric_cols = ["CGPA", "Work/Study Hours", "Financial Stress", "Total Pressure", "Total Satisfaction"]
for col in numeric_cols:
    data[col] = data[col].fillna(data[col].median())

# variables binaires pour le genre et le statut
data = pd.get_dummies(data, columns=["Gender", "Working Professional or Student"], drop_first=True)


# etape 2 :minmaxscaler aux pour depression score
cols_score_features = ["Total Pressure", "Total Satisfaction", "Sleep Duration", "Dietary Habits", 
                  "Financial Stress", "Family History of Mental Illness","Have you ever had suicidal thoughts ?","Work/Study Hours"]

# Calcul du "Depression Score" avec minmax scaler
scaler = MinMaxScaler()
score_features = data[cols_score_features].copy()
score_features = score_features.fillna(score_features.mean())
score_var = scaler.fit_transform(score_features)

weights = [0.1, -0.15, -0.1, -0.05, 0.2, 0.15, 0.3, 0.1]

data["Depression Score"] = score_var @ weights  # Produit matriciel

threshold = data["Depression Score"].quantile(0.75) # Classification en fonction du seuil de dépression
data["Depressed"] = (data["Depression Score"] > threshold).astype(int)

correlation = data["Depression Score"].corr(data["Depressed"])
print(f"Corrélation avec 'Depressed' : {correlation:.2f}")  # Idéalement > 0.5


# Étape 3 : Appliquer StandardScaler pour tracer le graphe
graph_features = [ "Total Pressure", "Total Satisfaction", 
                  "Sleep Duration", "Dietary Habits", "Financial Stress", 
                  "Family History of Mental Illness", "Work/Study Hours","Have you ever had suicidal thoughts ?"
                   ]
#le poids de ces colonne ??
scaler_std = StandardScaler()
final_graph_data = data[graph_features]
final_graph_data = final_graph_data.fillna(final_graph_data.mean())  # Par exemple, remplacer NaN par la moyenne de chaque colonne

normalized_data = scaler_std.fit_transform(final_graph_data)
final_graph_data["Depressed"] = data["Depressed"]
full_data = data.reset_index()

# utilisation de Gower pour la pondération
# Calcul de la matrice de similarité
distance_matrix = gower.gower_matrix(final_graph_data)  # final_graph_data doit être un DataFrame

# Définition dynamique du seuil
seuil_similarite = np.percentile(distance_matrix.flatten(), 5)  # 5% des distances les plus faibles

# Création directe du graphe à partir de la matrice
G = nx.from_numpy_array(distance_matrix < seuil_similarite)

# Détection des communautés avec Louvain
partition = community_louvain.best_partition(G)

# Supprimer communautés < 2 noeuds
community_counts = Counter(partition.values())
filtered_partition = {node: comm for node, comm in partition.items() if community_counts[comm] > 2}
G.remove_nodes_from([node for node in G.nodes() if node not in filtered_partition])

# Calculer les profils moyens de chaque communauté
# identifier les nœuds conservés après filtrage
kept_nodes = [node for node in G.nodes() if node in filtered_partition]

# Filtrer le DataFrame pour ne garder que ces nœuds
data = data.iloc[kept_nodes].copy()

# ajouter la colonne "Community"
data["Community"] = [filtered_partition[node] for node in kept_nodes]

# les statistiques des communautés
# Toutes les variables d'intérêt 
variables_a_explorer = [
    "Depression Score", "Total Pressure", "Total Satisfaction",
    "Sleep Duration", "Dietary Habits", "Financial Stress",
    "Family History of Mental Illness",
    "Have you ever had suicidal thoughts ?",
    "Work/Study Hours",
    
]

# Moyennes par communauté
profil_communautes = data.groupby("Community")[variables_a_explorer].mean().round(2)
print(profil_communautes)

plt.figure(figsize=(12, 6))
sns.heatmap(profil_communautes.T, cmap="coolwarm", annot=True, fmt=".2f")
plt.title("Profil moyen de chaque communauté")
plt.xlabel("Communauté")
plt.ylabel("Variable")
plt.tight_layout()
plt.show()

# colorier les noeuds du graphes en fonction de chaques facteurs
highlight_nodes = {
    "short Sleep": [],
    "long Sleep": [],
    "High Financial Stress": [],
    "Low Financial Stress": [],
    "suicidal thoughts": [],
    "Family History": [],
    "overworking": [],
    "free working": [],
    "pressure": [],
    "satisfaction": [],
    "male": [],
    "ménage": []
}


for node in G.nodes():
    person = data.loc[node]  # ici on utilise les données brutes, pas normalisées

    if person["Sleep Duration"] <= 1:
        highlight_nodes["short Sleep"].append(node)
    
    if person["Sleep Duration"] > 1:
        highlight_nodes["long Sleep"].append(node)

   
    if person["Financial Stress"] >= 4:  
        highlight_nodes["High Financial Stress"].append(node)

    if person["Financial Stress"] < 3:  
        highlight_nodes["Low Financial Stress"].append(node)
   

    if person["Have you ever had suicidal thoughts ?"] == 1:
        highlight_nodes["suicidal thoughts"].append(node)
   

    if person["Family History of Mental Illness"] == 1:
        highlight_nodes["Family History"].append(node)
   

    if person["Work/Study Hours"] >= 9:
        highlight_nodes["overworking"].append(node)

    if person["Work/Study Hours"] <= 5:
        highlight_nodes["free working"].append(node)
    

    if person["Total Pressure"] >3 :
        highlight_nodes["pressure"].append(node)
    
    if person["Total Satisfaction"] > 3:
        highlight_nodes["satisfaction"].append(node)

    if person["Gender_Male"] ==1:
        highlight_nodes["male"].append(node)

    if person["Working Professional or Student_Working Professional"] ==1:
        highlight_nodes["ménage"].append(node)


 
# Étape 8 : Visualisation du graphe avec code couleur pour les communautés
for node in G.nodes():
    G.nodes[node]["Depressed"] = data.loc[node, "Depressed"] 
pos = nx.spring_layout(G, seed=42, k=0.4,iterations=200)  

plt.figure(figsize=(14, 14))

# Nombre de communautés
num_communities = len(set(partition.values()))
colors_map = plt.cm.get_cmap("tab20", num_communities)
print("Nombre de communautés :", num_communities)

nodes_all = list(G.nodes())
sizes = [ 10+2 * G.degree(node) for node in nodes_all]
edge_colors = ["red" if G.nodes[node]["Depressed"] == 1 else 'black' for node in nodes_all]

########################################################################################
node_colors = [partition[node] for node in nodes_all]
#node_colors_tab = []

# for node in G.nodes():
#     if node in highlight_nodes["short Sleep"]:
#         node_colors_tab.append('blue')  
#     elif node in highlight_nodes["long Sleep"]:
#         node_colors_tab.append('green')  
#     else:
#         node_colors_tab.append('gray')  


# for node in G.nodes():
#     if node in highlight_nodes["High Financial Stress"]:
#         node_colors_tab.append('blue')  
#     elif node in highlight_nodes["Low Financial Stress"]:
#         node_colors_tab.append('green')  
#     else:
#         node_colors_tab.append('gray')  

# for node in G.nodes():
#     if node in highlight_nodes["overworking"]:
#         node_colors_tab.append('blue')  
#     elif node in highlight_nodes["free working"]:
#         node_colors_tab.append('green')  
#     else:
#         node_colors_tab.append('gray')  


#node_colors = ['blue' if node in highlight_nodes["pressure"] else 'gray' for node in G.nodes()]
#node_colors = ['blue' if node in highlight_nodes["satisfaction"] else 'gray' for node in G.nodes()]
#node_colors = ['blue' if node in highlight_nodes["male"] else 'gray' for node in G.nodes()]
#node_colors = ['purple' if node in highlight_nodes["suicidal thoughts"] else 'yellow' for node in G.nodes()]
#node_colors = ['blue' if node in highlight_nodes["Family History"] else 'gray' for node in G.nodes()]


##########################################################################################################


nx.draw_networkx_nodes(
    G, pos,
    nodelist=nodes_all,
    node_shape="o",
    node_size=sizes,
    node_color=node_colors,
    cmap=colors_map,
    edgecolors=edge_colors,
    linewidths=1.5,
    alpha=0.9,
    vmin=0,
    vmax=num_communities - 1


)

nx.draw_networkx_edges(
    G, pos,
    alpha=0.15,
    edge_color="grey",
    width=0.5
)

sampled_labels = {node: f"C{partition[node]}" for i, node in enumerate(G.nodes()) if i % 15 == 0}
nx.draw_networkx_labels(G, pos, labels=sampled_labels, font_size=9)


legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Non-dépressif',
           markerfacecolor='grey', markersize=10, markeredgecolor='black'),
    Line2D([0], [0], marker='o', color='w', label='Dépressif',
           markerfacecolor='grey', markersize=10, markeredgecolor='red')
]
plt.legend(handles=legend_elements, loc='upper right')


plt.axis('off')
plt.title("Visualisation des Communautés avec Détection de Dépression", fontsize=14)
plt.show()