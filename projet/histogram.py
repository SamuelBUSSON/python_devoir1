import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

file = pd.read_csv("student-mat.csv", encoding = "ISO-8859-1", delimiter=";")

num_ligne = file.shape[0]
num_column = file.shape[1]

for column in file:
    count_lines = file[column].isnull().sum()
    print("Sur les", num_ligne, "lignes, il y a", count_lines , "ligne(s) avec des données manquantes dans la colonne", column)

dataframe = pd.DataFrame(file) #Transformation en dataframe

df1 = dataframe["absences"].unique().tolist()
df2 = dataframe["age"].groupby(dataframe["absences"]).count().tolist()
print("La variance est : " + str(np.var(dataframe["absences"])))
print("La moyenne est : " + str(np.mean(dataframe["absences"])))
#par défaut, ils sont rangés dans le même ordre
print(df1, df2)
print(len(df1), len(df2))
plt.bar(df1, df2)
plt.suptitle('Nombres d\'absences par étudiants')
plt.ylabel('Nb Etudiants')
plt.xlabel('Absences')
plt.show()
