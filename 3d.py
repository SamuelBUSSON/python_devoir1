import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)

file = pd.read_csv("crimeMontreal.csv", encoding = "ISO-8859-1")

dataframe = pd.DataFrame(file) #Transformation en dataframe

df1 = dataframe["PDQ"].unique().tolist() #Liste des PDQ
df2 = dataframe["CATEGORIE"].groupby(dataframe["PDQ"]).count().tolist()
df1.pop(33)

df1, df2 = np.meshgrid(df1, df2)
R = np.sqrt(df1**2 + df2**2)
Z = np.sin(R)
ax.plot_surface(df1, df2, Z, rstride=1, cstride=1, cmap='hot')
plt.show()