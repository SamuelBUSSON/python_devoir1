import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

import csv
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
#from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import csv
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
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

plt.pie(sizes, labels=labels,  autopct='%1.1f%%', shadow=True, startangle=90)

plt.axis('equal')

plt.savefig('PieChart.png')
plt.show()





# #############################################################################
# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
labels_true = file["CATEGORIE"].unique();


# Now we can save it to a numpy array.
file = file[["CATEGORIE" , "QUART", "PDQ"]]


categorie = file["CATEGORIE"].unique()
categorieInt = file["CATEGORIE"].unique()


quart = file["QUART"].unique()
quartInt = file["QUART"].unique()

i = 0;
for cat in categorie:
    print(cat, "aura pour valeur", i)
    categorieInt[i] = i
    file["CATEGORIE"] = file["CATEGORIE"].str.replace(cat, str(i))
    i = i + 1

print()
i = 0;
for qua in quart:
    print(qua, "aura pour valeur", i)
    quartInt[i] = i
    file["QUART"] = file["QUART"].str.replace(qua, str(i))
    i = i + 1


file["QUART"] = file["QUART"].astype(int)
file["CATEGORIE"] = file["CATEGORIE"].astype(int)


print(file)
#clustering = DBSCAN(eps=3, min_samples=2, metric='string').fit(X)

