import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file = pd.read_csv("crimeMontreal.csv", encoding = "ISO-8859-1")

dataframe = pd.DataFrame(file) #Transformation en dataframe

df1 = dataframe["PDQ"].unique().tolist() #Liste des PDQ
df2 = dataframe["CATEGORIE"].groupby(dataframe["PDQ"]).count().tolist()
#par défaut, ils sont rangés dans le même ordre
print(df1, df2)
df1.pop(33)
print(len(df1), len(df2))
plt.plot(df1, df2)
#plt.scatter(df1, df2)
plt.suptitle('PDQ / Instances CATEGORIES')
plt.ylabel('Nb Instances')
plt.xlabel('PDQ')
plt.show()