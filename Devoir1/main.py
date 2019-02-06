import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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