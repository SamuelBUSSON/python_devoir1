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

f1 = data['LONGITUDE'].unique()
f2 = data['LATITUDE'].unique()

for i in range(len(f1)-1):
    if f1[i] == 1.0 or f2[i] == 1.0:
        f1 = np.delete(f1, i)
        f2 = np.delete(f2, i)

X = np.array(list(zip(f1, f2)))
kmeans = KMeans(n_clusters=4)
kmeans = kmeans.fit(X)
labels = kmeans.predict(X)
C = kmeans.cluster_centers_

# Création du troisième attribut
f3 = np.copy(f2)
for i in NumClusters:
    dataInCluster = data[clusterLabels[cluster==i].rowNames,]
    distance = norm(dataInCluster-clusterCenter[i])