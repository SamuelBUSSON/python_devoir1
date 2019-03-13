import numpy as np
import pandas as pd

file = pd.read_csv("crimeMontreal.csv", encoding = "ISO-8859-1")
num_ligne = file.shape[0] #nombre de lignes
num_column = file.shape[1] #nombre de colonnes

dataframe = pd.DataFrame(file) #Transformation en dataframe
#print(dataframe)
# i = 4
#print(dataframe[0:i]["X"])
#print(dataframe['PDQ'].isnull().value_counts())
#print(dataframe['X'][i])
#print(dataframe[dataframe["PDQ"] == 21]) #les données ou PDQ = 21
#print(dataframe["QUART"].max()) #nous affiche le nombre d'instance les plus présentes

print(dataframe["X"].mean()) #nous calcule la moyenne des coordonnées X
print(dataframe["Y"].mean()) #nous calcule la moyenne des coordonnées Y


print("Please, input an X coordinate : ")
X = input()

print("Now, input and Y coordinate : ")
Y = input()

print("Now, input the range : ")
range = input()

#Conversion des floats en datatypes
dt1 = np.float32(X)
dt2 = np.float32(Y)
dt3 = np.float32(range)

#data = dataframe[(dataframe["LONGITUDE"] >= dt1) & (dataframe["LATITUDE"] >= dt2)]

data = dataframe[((dataframe["X"] <= (dt1 + dt3)) & (dataframe["X"] >= (dt1 -dt3))) & ((dataframe["Y"] <= (dt2 + dt3)) & (dataframe["Y"] >= (dt2 - dt3)))]

#data1 = dataframe[dataframe['X'] >= 297654]
#data2 = dataframe[dataframe["Y"] >= 5041877]
#data = np.concatenate((data1, data2), axis = 0) #On met l'axis sur 0 pour créer deux tableaux distinctement
print(data)
print("Le nombre d'instance est de " + str(data["X"].count()))
