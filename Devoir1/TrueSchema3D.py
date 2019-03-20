#Chargement des bibliothèques
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

# Importation du dataset
data = pd.read_csv('crimeMontreal.csv', encoding='ISO-8859-1')

#On nettoie les données
data = data[(data != 0).all(1)]
data['X'] = data['X'].apply(lambda x: x*0.001)
data['Y'] = data['Y'].apply(lambda x: x*0.001)

#On récupère le nombre de ligne
num_ligne = data.shape[0]
#On ne prend pas toutes les données pour éviter de surcharger la mémoire
data = data.tail(int(num_ligne/20))

# Paramètre de la figure de clusters, ici on change les categories en numéros
categorie = data["CATEGORIE"].unique()
categorieInt = data["CATEGORIE"].unique()
i = 0;
print("On remplace les catégorie par des numéros :")
for cat in categorie:
   print(cat, "aura pour valeur", i)
   categorieInt[i] = i
   data["CATEGORIE"] = data["CATEGORIE"].replace(cat, i)
   i = i + 1

#On choisit les différentes valeurs de code
f1 = data['X'].values
f2 = data['Y'].values
f3 = data['CATEGORIE'].values

#Les différentes couleurs
colormap=np.array(['red','green','blue','yellow'])
X = np.array(list(zip(f1, f2, f3)))
kmeans = KMeans(n_clusters=4)
kmeans = kmeans.fit(X)
labels = kmeans.predict(X)
C = kmeans.cluster_centers_
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(f1, f2, f3, c=colormap[kmeans.labels_])
print(C)
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c='#050505', s=1000)

#On définit les différents axes
ax.set_title('Catégorie en fonction des coordonnées ')
ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')


x = ['Méfait', 'Vol contenu ', 'Vol véhicule', 'Introduction', 'Vols', 'Crimes']
l = [0, 1, 2, 3, 4, 5]

ax.set_zticks(l)
ax.set_zticklabels(x)

plt.show()
