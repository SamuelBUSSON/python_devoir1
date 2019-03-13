import numpy as np
import pandas as pd

file = pd.read_csv("crimeMontreal.csv", encoding = "ISO-8859-1")

dataframe = pd.DataFrame(file) #Transformation en dataframe

ls_date = dataframe["DATE"].unique() #Liste des DATES
max = 0
date_max = 0
for x in ls_date:
    data = dataframe[(dataframe["DATE"] == x)]
    total = data["DATE"].count()
    if (data["DATE"].count() > max):
        max = total
        date_max = x
    print("Le " + str(x) + ", il y a un total de : " + str(total) + " d'actes criminels enregistrés.")
print("Le jour ou il y a eu le plus d'actes criminels est le " + str(date_max) + " avec un total de " + str(max) + " enregistrements.")