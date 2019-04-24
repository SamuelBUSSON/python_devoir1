#Chargement des bibliothèques
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

### Intégration des datasets
file = pd.concat([pd.read_csv("student-mat.csv", encoding = "ISO-8859-1", delimiter=";"), pd.read_csv("student-por.csv", encoding = "ISO-8859-1", delimiter=";")])
dataframe = pd.DataFrame(file) #Transformation en dataframe
### Fin de l'intégration des datasets

f1 = dataframe['G1'].values
f2 = dataframe['G2'].values
f3 = dataframe['G3'].values
#######################################################################

#######################################################################
colormap=np.array(['red','green','blue','yellow'])
# X, y = make_blobs(n_samples=800, n_features=3, centers=4)
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
plt.show()