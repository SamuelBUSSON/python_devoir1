import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

file = pd.read_csv("student-mat.csv", encoding = "ISO-8859-1", delimiter=";") #Ensemble mathématiques
file2 = pd.read_csv("student-por.csv", encoding = "ISO-8859-1", delimiter=";") #Ensemble portugais
file3 = pd.concat([file, file2]) #Concaténation des deux ensembles

dataframe1 = pd.DataFrame(file2) #Transformation en dataframe (protugais)
dataframe2 = pd.DataFrame(file) #Transformation en dataframe (protugais)
dataframe3 = pd.DataFrame(file3) #Transformation en dataframe (protugais)

ls_cat = dataframe1["school"].unique() #Liste des écoles
for x in ls_cat:
   data1 = dataframe1[(dataframe1["school"] == x)]
   data2 = dataframe2[(dataframe2["school"] == x)]
   data3 = dataframe3[(dataframe3["school"] == x)]
   print("--- Portugais ---")
   print("Il y a un nombre total de " + str(data1["school"].count()) + " étudiants dans l'école " + str(x))
   print("--- Mathématiques ---")
   print("Il y a un nombre total de " + str(data2["school"].count()) + " étudiants dans l'école " + str(x))
   print("--- Total ---")
   print("Il y a un nombre total de " + str(data3["school"].count()) + " étudiants dans l'école " + str(x))

labels = file3["studytime"].unique()
sizes = []
for lab in labels:
   curVal = file3[file3["studytime"] == lab].shape[0]
   sizes.append(curVal)
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']

plt.pie(sizes, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=90)
plt.show()