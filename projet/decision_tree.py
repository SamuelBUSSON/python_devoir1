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
data = pd.read_csv('student-mat.csv', encoding='ISO-8859-1')
num_ligne = data.shape[0]
data = data.tail(int(num_ligne/20))
def marks(total_grades):
    if(total_grades < 5): return("Very Low")
    if(total_grades < 12 and total_grades >=5): return("Low")
    if(total_grades >= 12 and total_grades < 16): return("Average")
    if(total_grades >= 16): return("High")
    else : return("Average")

def mergeMarksData(csv_name):
    all_students = pd.read_csv(csv_name, sep=";")
    all_students["total_grades"] = (all_students["G1"] + all_students["G2"] + all_students["G3"]) / 3
    student_Without_All_Grad = all_students
    max = student_Without_All_Grad["total_grades"].max()
    min = student_Without_All_Grad["total_grades"].min()
    student_Without_All_Grad["grades"] = student_Without_All_Grad["total_grades"].apply(marks)
    return student_Without_All_Grad

all_studentsMat = mergeMarksData("student-mat.csv")
all_studentsPor = mergeMarksData("student-por.csv")

data = data[["total_grades",  "absences", ""]]

#Pour garder la valeur des float après conversion en int
data["total_grades"] = data["total_grades"].astype(int)
data["absences"] = data["absences"].astype(int)
data["frreetime"] = data["frreetime"].astype(int)

feature_cols = ['absences', 'frreetime']
class_cols = ['total_grades']
X = data[feature_cols] # Attributs
y = data.values # Variable de classes à prédire

# Découpage du dataset en dataset d'entrainement et de tests
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
dt.fit(X_train, y_train)
dot_data = tree.export_graphviz(dt, out_file=None, rounded=True, feature_names=feature_cols, class_names=class_cols, filled=True)
graph = graphviz.Source(dot_data)
graph.render("Résultat en fonction des sorties")

from sklearn.metrics import precision_score

clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100,
                                       max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)
dot_data = tree.export_graphviz(dt, out_file=None, rounded=True, feature_names=feature_cols)
graph = graphviz.Source(dot_data)
graph.render("Entropie")


y_predicted = clf_entropy.predict(X_test.astype(int))
print(precision_score(y_test[:, 0], y_predicted[:, 0]))