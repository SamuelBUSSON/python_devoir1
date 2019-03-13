import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

file = pd.read_csv("crimeMontreal.csv", encoding = "ISO-8859-1")

dataframe = pd.DataFrame(file) #Transformation en dataframe

df1 = dataframe["PDQ"].unique().tolist() #Liste des PDQ
df2 = dataframe["CATEGORIE"].groupby(dataframe["PDQ"]).count().tolist()
#par défaut, ils sont rangés dans le même ordre
df1.pop(33)
nullfmt = NullFormatter()

#definitions des axes :
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
bottom_h = left_h = left + width + 0.02
rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.2]
rect_histy = [left_h, bottom, 0.2, height]

plt.figure(1, figsize=(8, 8))
axScatter = plt.axes(rect_scatter)
axHistx = plt.axes(rect_histx)
axHisty = plt.axes(rect_histy)

# Pas de labels
axHistx.xaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)

# the scatter plot
axScatter.scatter(df1, df2)

# now determine nice limits by hands
bindwidth = 0.25
#Cette ligne de code ne fonctionne pas, c'est pourtant la même que le cours
#xymax = np.max( [ np.max(np.fabs(df1)), np.max(np.fabs(df2)) ] )
#print(xymax)
xymax = 3000 # On définit nos limites à 150
lim = (int(xymax/bindwidth) + 1) * bindwidth

# les limites
axScatter.set_xlim((-lim, lim))
axScatter.set_ylim((-lim, lim))
bins = np.arange(-lim, lim + bindwidth, bindwidth)
axHistx.hist(df1, bins=bins)
axHisty.hist(df2, bins=bins, orientation="horizontal")
axHistx.set_xlim(axScatter.get_xlim())
axHisty.set_ylim(axScatter.get_ylim())

#Tracement du graphe
plt.show()