# coding: utf-8
#Chargement des bibliothèques
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Importation du dataset
data = pd.read_csv('crimeMontreal.csv', encoding='ISO-8859-1')
num_ligne = data.shape[0]
data = data.tail(int(num_ligne/20)) # On divise notre ensemble de données par 20, car il est vraiment grand
# Récupération de la longitude et de la lattitude
f1 = data['LONGITUDE'].unique()
f2 = data['LATITUDE'].unique()
# Suppression des longitudes et lattitudes non définis
for i in range(len(f1)-1):
    if f1[i] == 1.0 or f2[i] == 1.0:
        f1 = np.delete(f1, i)
        f2 = np.delete(f2, i)
# Création de l'ensemble de données
X = np.array(list(zip(f1, f2)))
#Création d'un objet K-Means avec un regroupement en 3 clusters (groupes)
model = KMeans(n_clusters=4)

#application du modèle sur notre jeu de données
model.fit(X)
colormap=np.array(['red','green','blue','yellow'])

#Visualisation des clusters formés par K-Means
plt.scatter(f1, f2, c=colormap[model.labels_],s=40)
plt.title('Classification K-means ')
plt.show()