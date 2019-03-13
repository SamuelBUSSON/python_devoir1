import pandas as pd
import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, metrics
from sklearn.cluster import DBSCAN
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

file = pd.read_csv("crimeMontreal.csv", encoding = "ISO-8859-1")
num_ligne = file.shape[0]
num_column = file.shape[1]


print("Le fichier possède", num_ligne, "observations et", file.shape[1], "propriétés")

for column in file:
    count_lines = file[column].isnull().sum()
    print("Sur les", num_ligne, "lignes, il y a", count_lines , "ligne(s) avec des données manquantes dans la colonne", column)



print("\nPBQ correspond au numéro du poste de police, aucun poste n'a le numéro -1, on remplace donc par ce dernier\n")
file = file.fillna(-1)
file = file[(file != 0).all(1)]
file['X'] = file['X'].apply(lambda x: x*0.0001)
file['Y'] = file['Y'].apply(lambda x: x*0.0001)
file = file.round(1)



print("Pour plus de clarté on remplace \"dans / sur\" par \"du contenu\"  \n")
file["CATEGORIE"] = file["CATEGORIE"].str.replace('dans / sur véhicule à moteur', 'du contenu d\'un véhicule')


labels = file["CATEGORIE"].unique()
sizes = []
totalValue = 0
for lab in labels:
    curVal = file[file["CATEGORIE"] == lab].shape[0]
    totalValue = totalValue + curVal
    sizes.append(curVal)

print("Sur les",num_ligne,"du fichier",totalValue,"lignes sont utilisée(s) pour le diagramme suivant")

# plt.pie(sizes, labels=labels,  autopct='%1.1f%%', shadow=True, startangle=90)
#
# plt.axis('equal')
#
# plt.savefig('PieChart.png')
# plt.show()





# #############################################################################
# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
labels_true = file["CATEGORIE"].unique();


# Now we can save it to a numpy array.
file = file[["CATEGORIE" , "QUART", "PDQ", "X", "Y"]]
# file = file[["CATEGORIE" , "QUART"]]

categorie = file["CATEGORIE"].unique()
categorieInt = file["CATEGORIE"].unique()


quart = file["QUART"].unique()
quartInt = file["QUART"].unique()

i = 0;
print("On remplace les catégorie par des numéros :")
for cat in categorie:
    print(cat, "aura pour valeur", i)
    categorieInt[i] = i
    file["CATEGORIE"] = file["CATEGORIE"].str.replace(cat, str(i))
    i = i + 1

print("\n===================================== \n\nOn remplace les jours, nuit,... par des numéros :")
i = 0;

for qua in quart:
    print(qua, "aura pour valeur", i)
    quartInt[i] = i
    file["QUART"] = file["QUART"].str.replace(qua, str(i))
    i = i + 1


file["QUART"] = file["QUART"].astype(int)
file["CATEGORIE"] = file["CATEGORIE"].astype(int)


#print(file)
X = file.tail(int(num_ligne/20))

df2 = file["CATEGORIE"].groupby(file["X"]).size()
print(df2)
df3 = file["Y"]
df1 = file["X"].unique()

X = np.vstack((df1, df2)).T

db = DBSCAN(eps=150, min_samples=2).fit(X)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)
    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1],  'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
