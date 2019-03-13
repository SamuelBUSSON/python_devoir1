import numpy as np
import pandas as pd
import matplotlib
file = pd.read_csv("crimeMontreal.csv", encoding = "ISO-8859-1")

dataframe = pd.DataFrame(file) #Transformation en dataframe

ls_cat = dataframe["CATEGORIE"].unique() #Liste des CATEGORIES
for x in ls_cat:
    data = dataframe[(dataframe["CATEGORIE"] == x)]
    print("Il y a un nombre total de " + str(data["CATEGORIE"].count()) + " actes criminels enregistrés dans la catégorie " + str(x))

#df = dataframe[(dataframe["QUART"] == "soir") & (dataframe["CATEGORIE"] == "Vols qualifiés")]
#print("Il y a un nombre total de " + str(len(df)) + " instances de vols qualifiés le soir.")

vols_qualifies = dataframe[(dataframe["CATEGORIE"] == "Vols qualifiés")]
print(vols_qualifies)
df = dataframe["CATEGORIE"].groupby(dataframe["PDQ"]).count()