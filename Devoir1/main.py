import numpy as np
import pandas as pd

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
