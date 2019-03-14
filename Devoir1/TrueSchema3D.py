#Chargement des bibliothèques
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

# Importation du dataset
data = pd.read_csv('crimeMontreal.csv', encoding='ISO-8859-1')

data = data[(data != 0).all(1)]
data['X'] = data['X'].apply(lambda x: x*0.001)
data['Y'] = data['Y'].apply(lambda x: x*0.001)

num_ligne = data.shape[0]
data = data.tail(int(num_ligne/20))

# Paramètre de la figure de clusters
# plt.rcParams['figure.figsize'] = (16, 9)
categorie = data["CATEGORIE"].unique()
categorieInt = data["CATEGORIE"].unique()
i = 0;
print("On remplace les catégorie par des numéros :")
for cat in categorie:
   print(cat, "aura pour valeur", i)
   categorieInt[i] = i
   data["CATEGORIE"] = data["CATEGORIE"].replace(cat, i)
   i = i + 1

f1 = data['X'].values
f2 = data['Y'].values
f3 = data['CATEGORIE'].values

colormap=np.array(['red','green','blue','yellow'])
# X, y = make_blobs(n_samples=800, n_features=3, centers=4)
X = np.array(list(zip(f1, f2, f3)))
print(f1, f3, f2)
kmeans = KMeans(n_clusters=4)
kmeans = kmeans.fit(X)
labels = kmeans.predict(X)
C = kmeans.cluster_centers_
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(f1, f2, f3, c=colormap[kmeans.labels_])
print(C)
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c='#050505', s=1000)

ax.set_title('Carégorie en fonction des coordonnées ')

ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_zlabel('Categorie')

plt.show()
