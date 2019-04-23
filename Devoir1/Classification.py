#Chargement des bibliothèques

from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support, pairwise_distances_argmin, \
    euclidean_distances
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from math import sin, cos, sqrt, atan2, radians

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn import tree
import graphviz
from sklearn.metrics import confusion_matrix, precision_score

# Importation du dataset
data = pd.read_csv('crimeMontreal.csv', encoding='ISO-8859-1')
num_ligne = data.shape[0]
data = data.tail(int(num_ligne/20))

data = data[(data != 0).all(1)]

f1 = data['LONGITUDE']
f2 = data['LATITUDE']

X = np.array(list(zip(f1, f2)))
kmeans = KMeans(n_clusters=4)
kmeans = kmeans.fit(X)
labels = kmeans.predict(X)
C = kmeans.cluster_centers_



# Création du troisième attribut
f3 = np.copy(f2)

# approximate radius of earth in km
R = 6373.0

minValue1 = 100.0
maxValue1 = 0.0

minValue2 = 100.0
maxValue2 = 0.0

minValue3 = 100.0
maxValue3 = 0.0

minValue4 = 100.0
maxValue4 = 0.0

i = 0
for clusterNbr in kmeans.labels_:
    clusterPoint = C[clusterNbr]
    currentPoint = X[i]

    lat1 = radians(clusterPoint[0])
    lon1 = radians(clusterPoint[1])

    lat2 = radians(currentPoint[0])
    lon2 = radians(currentPoint[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    print("Result:", distance, "km du centre de son cluster")

    if(clusterNbr == 0):
        if(distance < minValue1):
            minValue1 = distance
        if(distance > maxValue1):
            maxValue1 = distance

    if(clusterNbr == 1):
        if(distance < minValue2):
            minValue2 = distance
        if(distance > maxValue2):
            maxValue2 = distance

    if(clusterNbr == 2):
        if(distance < minValue3):
            minValue3 = distance
        if(distance > maxValue3):
            maxValue3 = distance

    if(clusterNbr == 3):
        if(distance < minValue4):
            minValue4 = distance
        if(distance > maxValue4):
            maxValue4 = distance
    i += 1

distanceMoyenneCluster1 = (maxValue1 + minValue1)/2
distanceMoyenneCluster2 = (maxValue2 + minValue2)/2
distanceMoyenneCluster3 = (maxValue3 + minValue3)/2
distanceMoyenneCluster4 = (maxValue4 + minValue4)/2

labellized_points = []



i = 0
value = ''
for clusterNbr in kmeans.labels_:
    clusterPoint = C[clusterNbr]
    currentPoint = X[i]

    lat1 = radians(clusterPoint[0])
    lon1 = radians(clusterPoint[1])

    lat2 = radians(currentPoint[0])
    lon2 = radians(currentPoint[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    if (clusterNbr == 0):
        if(distance > distanceMoyenneCluster1 ):
            value = ("Peu Dangereux")
        else:
            value = ("Dangereux")

    if (clusterNbr == 1):
        if(distance > distanceMoyenneCluster2 ):
            value = ("Peu Dangereux")
        else:
            value = ("Dangereux")

    if (clusterNbr == 2):
        if(distance > distanceMoyenneCluster3 ):
            value = ("Peu Dangereux")
        else:
            value = ("Dangereux")

    if (clusterNbr == 3):
        if(distance > distanceMoyenneCluster4 ):
            value = ("Peu Dangereux")
        else:
            value = ("Dangereux")

    labellized_points.append(value)
    i += 1



data = data.assign(Class=pd.Series(labellized_points).values)

print(data)

categorie = data["CATEGORIE"].unique()
quart = data["QUART"].unique()

date = data["DATE"].unique()
classValue = data["Class"].unique()


i = 0
print("On remplace les catégorie par des numéros :")
for cat in categorie:
    print(cat, "aura pour valeur", i)
    data["CATEGORIE"] = data["CATEGORIE"].str.replace(cat, str(i))
    i = i + 1

print("\n===================================== \n\nOn remplace les jours, nuit,... par des numéros :")
i = 0

for qua in quart:
    print(qua, "aura pour valeur", i)
    data["QUART"] = data["QUART"].str.replace(qua, str(i))
    i = i + 1

i = 0
for dat in date:
    print(dat, "aura pour valeur", i)
    data["DATE"] = data["DATE"].str.replace(dat, str(i))
    i = i + 1

i = 0
for clas in classValue:
    print(clas, "aura pour valeur", i)
    data["Class"] = data["Class"].str.replace(clas, str(i))
    i = i + 1


k_means_cluster_centers = kmeans.cluster_centers_
k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)


for i,j in enumerate(set(k_means_labels)):
    positions=X[np.where(k_means_labels == i)]
    output=sum(euclidean_distances(positions,k_means_cluster_centers[j].reshape(1,-1)))
    print('cluster {} has a  heterogeneity of {}'.format(i,output))




data = data[["Class",  "LATITUDE", "LONGITUDE"]]

#Pour garder la valeur des float après conversin en int
data['LATITUDE'] = data['LATITUDE'].apply(lambda x: x*10000)
data['LONGITUDE'] = data['LONGITUDE'].apply(lambda x: x*10000)

data["Class"] = data["Class"].astype(int)
data["LATITUDE"] = data["LATITUDE"].astype(int)
data["LONGITUDE"] = data["LONGITUDE"].astype(int)


#split dataset in features and target variable
feature_cols = ['LATITUDE', 'LONGITUDE', 'Class']
X = data[feature_cols] # Features
y = data.values# Target variable


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test


dt = DecisionTreeClassifier(min_samples_split=20, random_state=5)
dt.fit(X_train, y_train)


dot_data = tree.export_graphviz(dt, out_file=None, rounded=True, feature_names=feature_cols)

graph = graphviz.Source(dot_data)
graph.render("Dangerosité")

clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=5, max_depth=3)
clf_entropy.fit(X_train, y_train)

dot_data = tree.export_graphviz(clf_entropy, out_file=None, rounded=True, feature_names=feature_cols)

graph = graphviz.Source(dot_data)
graph.render("Dangerosité_entropy")


y_predicted = clf_entropy.predict(X_test.astype(int))
print("Precision score :", precision_score(y_test[:, 0], y_predicted[:, 0]))

