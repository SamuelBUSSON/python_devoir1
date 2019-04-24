import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

plt.rc("font", size=14)

from sklearn.model_selection import train_test_split
import seaborn as sns

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

### Intégration des datasets
file = pd.read_csv("student-mat.csv", encoding = "ISO-8859-1", delimiter=";") #Ensemble mathématiques
file2 = pd.read_csv("student-por.csv", encoding = "ISO-8859-1", delimiter=";") #Ensemble portugais
file3 = pd.concat([file, file2]) #Concaténation des deux ensembles
dataframe1 = pd.DataFrame(file2) #Transformation en dataframe (protugais)
dataframe2 = pd.DataFrame(file) #Transformation en dataframe (portugais)
dataframe3 = pd.DataFrame(file3) #Transformation en dataframe
### Fin de l'intégration des datasets

# Variable de prédiction : bon élève (grade bon ou pas)
sns.countplot(x="sex", data=dataframe3, palette="hls")
plt.show()
count_no_sub = len(dataframe3[dataframe3['sex']=="F"])
count_sub = len(dataframe3[dataframe3['sex']=="M"])
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("Percentage of female is", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("Percentage of male", pct_of_sub*100)
print(dataframe3.groupby("sex").mean())

def marks(total_grades):
    if(total_grades < 5): return("Low")
    if(total_grades < 12 and total_grades >=5): return("Average")
    if(total_grades >= 12): return("High")
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
print(all_studentsMat.columns.values)
print(all_studentsPor.columns.values)

#Convert all string as number
for col in all_studentsMat.columns:  # Iterate over chosen columns
    all_studentsMat[col] = pd.Categorical(all_studentsMat[col])
    all_studentsMat[col] = all_studentsMat[col].cat.codes

targetColumn = 'grades'
feature_cols = ["failures", "studytime","famsize","Pstatus","Medu","Fedu","absences","freetime", 'schoolsup',
                'activities', 'higher', 'Walc', 'romantic']
X_AllColumn = all_studentsMat[feature_cols]
Y_FinalGrade = all_studentsMat[targetColumn]
X_train, X_test, y_train, Y_test = train_test_split(X_AllColumn, Y_FinalGrade, test_size=0.3, random_state=0)

model = linear_model.LogisticRegression(C=100000.0, class_weight=None, dual=False,
    fit_intercept=True, intercept_scaling=1, max_iter=100,
    multi_class='multinomial', n_jobs=None, penalty='l2', random_state=None,
    solver='lbfgs', tol=0.0001, verbose=0, warm_start=False)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# The coefficients
print('Coefficients : \n', model.coef_)
# Explained variance score: 1 is perfect prediction
print("Logistic Regression --> model score =",model.score(X=X_test, y=Y_test))

##################### NOW WE USE NEURAL NETWORK #########################################

#Dans le code qui suit nous utilisont StandartScaler pour normaliser les données de test et d'entrainement
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Importation des librairies Keras et du modèle Séquentiel
from keras.models import Sequential
from keras.layers import Dense
#Initialisation du réseau neuronal
classifier = Sequential()
# Ajout de la couche d'entrée et de la première couche cachée
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 13))
# Ajout de la seconde couche
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
# Ajout de la couche de sortie
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
# Compilation du réseau de neurones
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Application sur notre modèle
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
# Variable de prédiction
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
# Création de la matrice de confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)