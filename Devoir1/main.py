import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import plotly.graph_objs as go
import plotly.plotly as py

file = pd.read_csv("crimeMontreal.csv", encoding = "ISO-8859-1")
num_ligne = file.shape[0]
num_column = file.shape[1]

print("Le fichier possède", num_ligne, "observations et", file.shape[1], "propriétés")

for column in file:
    count_lines = file[column].isnull().sum()
    print("Sur les", num_ligne, "lignes, il y a", count_lines , "ligne(s) avec des données manquantes dans la colonne", column)



print("\nPBQ correspond au numéro du poste de police, aucun poste n'a le numéro -1, on remplace donc par ce dernier\n")
file = file.fillna(-1)

print("Pour plus de clarté on remplace \"dans / sur\" par \"du contenu\"  \n")
file["CATEGORIE"] = file["CATEGORIE"].str.replace('dans / sur véhicule à moteur', 'du contenu d\'un véhicule')


labels = file["CATEGORIE"].unique()
sizes = []
totalValue = 0
for lab in labels:
    curVal = file[file["CATEGORIE"] == lab].shape[0]
    totalValue = totalValue + curVal
    sizes.append(curVal)

print("Sur les",num_ligne,"du fichier",totalValue,"lignes sont utilisée(s) pour le diagramme suivant")

plt.pie(sizes, labels=labels,  autopct='%1.1f%%', shadow=True, startangle=90)

plt.axis('equal')

plt.savefig('PieChart.png')
plt.show()





# #############################################################################
# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
labels_true = file["CATEGORIE"].unique();


# Now we can save it to a numpy array.
file = file[["CATEGORIE" , "QUART", "PDQ"]]


categorie = file["CATEGORIE"].unique()
categorieInt = file["CATEGORIE"].unique()


quart = file["QUART"].unique()
quartInt = file["QUART"].unique()

i = 0;
print("On remplace les catégorie par des numéros :")
for cat in categorie:
    print(cat, "aura pour valeur", i)
    categorieInt[i] = i
    file["CATEGORIE"] = file["CATEGORIE"].str.replace(cat, str(i))
    i = i + 1

print("\n===================================== \n\nOn remplace les jours, nuit,... par des numéros :")
i = 0;
for qua in quart:
    print(qua, "aura pour valeur", i)
    quartInt[i] = i
    file["QUART"] = file["QUART"].str.replace(qua, str(i))
    i = i + 1


file["QUART"] = file["QUART"].astype(int)
file["CATEGORIE"] = file["CATEGORIE"].astype(int)


print(file)
X = file.tail(int(num_ligne/10))

# X = StandardScaler().fit_transform(X)
# 
#
# db = DBSCAN(eps=0.3, min_samples=10).fit(X)
# core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# core_samples_mask[db.core_sample_indices_] = True
# labels = db.labels_
#
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#
# print('Estimated number of clusters: %d' % n_clusters_)
#
#
# def matplotlib_to_plotly(cmap, pl_entries):
#     h = 1.0 / (pl_entries - 1)
#     pl_colorscale = []
#
#     for k in range(pl_entries):
#         C = map(np.uint8, np.array(cmap(k * h)[:3]) * 255)
#         pl_colorscale.append([k * h, 'rgb' + str((C[0], C[1], C[2]))])
#
#     return pl_colorscale
#
#
# unique_labels = set(labels)
#
# colors = matplotlib_to_plotly(plt.cm.Spectral, len(unique_labels))
# data = []
#
# for k, col in zip(unique_labels, colors):
#
#     if k == -1:
#         # Black used for noise.
#         col = 'black'
#     else:
#         col = col[1]
#
#     class_member_mask = (labels == k)
#
#     xy = X[class_member_mask & core_samples_mask]
#     trace1 = go.Scatter(x=xy[:, 0], y=xy[:, 1], mode='markers',
#                         marker=dict(color=col, size=14,
#                                     line=dict(color='black', width=1)))
#
#     xy = X[class_member_mask & ~core_samples_mask]
#     trace2 = go.Scatter(x=xy[:, 0], y=xy[:, 1], mode='markers',
#                         marker=dict(color=col, size=14,
#                                     line=dict(color='black', width=1)))
#     data.append(trace1)
#     data.append(trace2)
#
# layout = go.Layout(showlegend=False,
#                    title='Estimated number of clusters: %d' % n_clusters_,
#                    xaxis=dict(showgrid=False, zeroline=False),
#                    yaxis=dict(showgrid=False, zeroline=False))
# fig = go.Figure(data=data, layout=layout)
#
# py.iplot(fig)