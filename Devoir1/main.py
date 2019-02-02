import numpy as np
import pandas as pd

file = pd.read_csv("crimeMontreal.csv", encoding = "ISO-8859-1")
num_ligne = file.shape[0]
num_column = file.shape[1]

incomplete_columun = []

print("Le fichier possède", num_ligne, "lignes et", file.shape[1], "colonnes")

for column in file:
    count_lines = file[column].isnull().sum()
    print("Sur les", num_ligne, "lignes, il y a", count_lines , "lignes avec des données manquantes dans la colonne", column)
    if count_lines > 0:
        incomplete_columun.append(column)

for column in incomplete_columun:
   file[column].dropna(inplace = True)


#Vérifier la valeur de retour


