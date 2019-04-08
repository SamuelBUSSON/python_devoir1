import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
# Importation du dataset
file = pd.read_csv('student-mat.csv', encoding='ISO-8859-1', delimiter=";")
num_ligne = file.shape[0]
X = file.tail(int(num_ligne/1))
print(file)


romantic = file["sex"].unique()
romanticInt = file["sex"].unique()

i = 0;
for qua in romantic:
    print(qua, "aura pour valeur", i)
    romanticInt[i] = i
    file["sex"] = file["sex"].str.replace(qua, str(i))
    i = i + 1

df2 = file["studytime"]
df1 = file["sex"]
X = np.vstack((df1, df2)).T
file = pd.read_csv('student-mat.csv', encoding='ISO-8859-1')
num_ligne = file.shape[0]
db = DBSCAN(eps=150, min_samples=2).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

#Nombre de clusters dans les labels, en ignorant le bruit si présent
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
   if k == -1:
       # Utilisé pour le bruit.
       col = [0, 0, 0, 1]
   class_member_mask = (labels == k)
   xy = X[class_member_mask & core_samples_mask]
   plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14)
   xy = X[class_member_mask & ~core_samples_mask]
   plt.plot(xy[:, 0], xy[:, 1],  'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
