#Chargement des bibliothèques
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from math import sin, cos, sqrt, atan2, radians

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn import tree
import graphviz

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

# Rayon approximatif de la Terre
R = 6373.0
# Valeurs min et max pour chaque clusters (ici on en possède 4)
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

    print("Resultat :", distance, "km du centre de son cluster")

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

# Calcul de la moyenne des distances pour chaque cluster
distanceMoyenneCluster1 = (maxValue1 + minValue1)/2
distanceMoyenneCluster2 = (maxValue2 + minValue2)/2
distanceMoyenneCluster3 = (maxValue3 + minValue3)/2
distanceMoyenneCluster4 = (maxValue4 + minValue4)/2
#Liste des points labellisés
labellized_points = []
# Parcours complet des points en X et Y par clusters
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
    # Calcul de la distance
    distance = R * c
    # Assignation selon la distance par rapport au centroïde de chaque cluster
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
    # Ajout dans le tableau de variable de classe
    labellized_points.append(value)
    i += 1
# Assignation de la liste de variables labellisées dans le data
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

######################################################################################################
                            # Fin de la labélisation #
######################################################################################################

data = data[["Class",  "LATITUDE", "LONGITUDE"]]

# On souhaite avoir une précision des coordonnées jusqu'au 5ème chiffre après la virgule
data['LATITUDE'] = data['LATITUDE'].apply(lambda x: x*10000)
data['LONGITUDE'] = data['LONGITUDE'].apply(lambda x: x*10000)
#Pour garder la valeur des float après conversion en int
data["Class"] = data["Class"].astype(int)
data["LATITUDE"] = data["LATITUDE"].astype(int)
data["LONGITUDE"] = data["LONGITUDE"].astype(int)


feature_cols = ['LATITUDE', 'LONGITUDE']
class_cols = ['Class']
X = data[feature_cols] # Attributs
y = data.values # Variable de classes à prédire

# Découpage du dataset en dataset d'entrainement et de tests
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
dt.fit(X_train, y_train)
dot_data = tree.export_graphviz(dt, out_file=None, rounded=True, feature_names=feature_cols, class_names=class_cols, filled=True)
graph = graphviz.Source(dot_data)
graph.render("Dangerosite")

from sklearn.metrics import precision_score

clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100,
                                       max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)
dot_data = tree.export_graphviz(dt, out_file=None, rounded=True, feature_names=feature_cols)
graph = graphviz.Source(dot_data)
graph.render("Dangerosité_entropy")


y_predicted = clf_entropy.predict(X_test.astype(int))
print(precision_score(y_test[:, 0], y_predicted[:, 0]))