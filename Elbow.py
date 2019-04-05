import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

# Importation du dataset
data = pd.read_csv('crimeMontreal.csv', encoding='ISO-8859-1')
num_ligne = data.shape[0]
data = data.tail(int(num_ligne/20))

# Récupération des valeurs
f1 = data['LONGITUDE'].unique()
f2 = data['LATITUDE'].unique()
# Suppression des longitudes et lattitude non-placés
for i in range(len(f1)-1):
    if f1[i] == 1.0 or f2[i] == 1.0:
        f1 = np.delete(f1, i)
        f2 = np.delete(f2, i)
# Affichage des données
plt.plot()
plt.title('Dataset des crimes commis à Montréal')
plt.scatter(f1, f2)
plt.show()

# Création d'un nouveau graphique et données
plt.plot()
X = np.array(list(zip(f1, f2))).reshape(len(f1), 2)
colors = ['b', 'g', 'r']
markers = ['o', 'v', 's']

# K means détermine k
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# Graphique de la courbe d'Elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('Méthode d’Elbow, nombre de clusters optimaux')
plt.show()
