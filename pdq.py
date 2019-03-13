import numpy as np
import pandas as pd

file = pd.read_csv("crimeMontreal.csv", encoding = "ISO-8859-1")

dataframe = pd.DataFrame(file) #Transformation en dataframe

ls_pdq = dataframe["PDQ"].unique() #Liste des PDQ
nb_pdq = len(ls_pdq)
max = 0
PDQ_max = 0
for x in range(0, nb_pdq):
    data = dataframe[(dataframe["PDQ"] == ls_pdq[x])]
    total = data["PDQ"].count()
    if (data["PDQ"].count() > max):
        max = total
        PDQ_max = ls_pdq[x]
    print("Pour le poste de police " + str(ls_pdq[x]) + ", il y a un total de : " + str(total) + " d'actes criminels enregistrés.")
print("Le poste de police ayant enregistré le plus d'actes criminels est le " + str(PDQ_max) + " avec un total de " + str(max) + " enregistrements.")