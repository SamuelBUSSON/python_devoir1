#Chargement des bibliothèques
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

# Importation du dataset
data = pd.read_csv('crimeMontreal.csv', encoding='ISO-8859-1')
num_ligne = data.shape[0]
data = data.tail(int(num_ligne/20))

# Paramètre de la figure de clusters
# plt.rcParams['figure.figsize'] = (16, 9)
######################################################################

######################################################################
categorie = data["CATEGORIE"].unique()
categorieInt = data["CATEGORIE"].unique()
i = 0;
print("On remplace les catégorie par des numéros :")
for cat in categorie:
    print(cat, "aura pour valeur", i)
    categorieInt[i] = i
    data["CATEGORIE"] = data["CATEGORIE"].replace(cat, i)
    i = i + 1
######################################################################

######################################################################

f1 = data['X'].values
f2 = data['Y'].values
f3 = data['CATEGORIE'].values
i = 0
for i in range(len(f1)-1):
    if f1[i] < 1 or f2[i] < 1 or f1[i] > 50000:
        f1b = np.delete(f1, i)
        f2b = np.delete(f2, i)
        f3b = np.delete(f3, i)

#######################################################################

#######################################################################
colormap=np.array(['red','green','blue','yellow'])
# X, y = make_blobs(n_samples=800, n_features=3, centers=4)
X = np.array(list(zip(f1b, f2b, f3b)))
print(f1b, f3b, f2b)
kmeans = KMeans(n_clusters=4)
kmeans = kmeans.fit(X)
labels = kmeans.predict(X)
C = kmeans.cluster_centers_
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(f1b, f2b, f3b, c=colormap[kmeans.labels_])
print(C)
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c='#050505', s=1000)
plt.show()