
import matplotlib.pyplot as plt
import seaborn as sns
from keras.losses import mean_squared_error
from scipy import stats
import pandas as pd
from sklearn import tree, linear_model, preprocessing, model_selection, metrics
import graphviz
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, classification_report, r2_score

#Class grade in 3 parts, low, average and High
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



def marks(total_grades):
    if(total_grades < 5):
        return("Very Low")
    if(total_grades < 12 and total_grades >=5):
        return("Low")
    if(total_grades >= 12 and total_grades < 16):
        return("Average")
    if(total_grades >= 16):
        return("High")


def showAverageMarks(all_students, title):
    # Visualising grade
    plt.figure(figsize=(8, 6))
    # Use sns to show number of students in all grade category
    sns.countplot(all_students["grades"], order=["Low", "Average", "High"])
    plt.title('Final Grade/Number of Students in ' + title, fontsize=20)
    plt.xlabel('Final Grade', fontsize=16)
    plt.ylabel('Number of Student', fontsize=16)



def mergeMarksData(csv_name):
    all_students = pd.read_csv(csv_name, sep=";")
    all_students["total_grades"] = (all_students["G1"] + all_students["G2"] + all_students["G3"]) / 3

    #                              #
    #  Merge grade colum into one  #
    #                              #

    # remove the grading
    # student_Without_All_Grad = all_students.drop(["G1", "G2", "G3"], axis=1)
    student_Without_All_Grad = all_students
    max = student_Without_All_Grad["total_grades"].max()
    min = student_Without_All_Grad["total_grades"].min()

    print("Max value is : " + str(max))
    print("Min grade is : " + str(min))

    # Create last colum as the class column
    student_Without_All_Grad["grades"] = student_Without_All_Grad["total_grades"].apply(marks)

    return student_Without_All_Grad


def getPValueScore(student_data, analyzeData, title):

    # p-value between total grade and attribute
    student_data["grades"] = pd.Categorical(student_data["grades"])
    student_data["grades"] = student_data["grades"].cat.codes

    pearson_coef, p_value = stats.pearsonr(student_data[analyzeData], student_data["grades"])
    print("The Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value, "for students in", title)


all_studentsMat = mergeMarksData("student-mat.csv")
all_studentsPor = mergeMarksData("student-por.csv")

showAverageMarks(all_studentsMat, "Math")
showAverageMarks(all_studentsPor, "Portuguese")

plt.show()

getPValueScore(all_studentsMat, "absences", "Math")
getPValueScore(all_studentsPor, "absences", "Portuguese")


#split dataset in features and target variable

#Convert all string as number
for col in all_studentsMat.columns:  # Iterate over chosen columns
    all_studentsMat[col] = pd.Categorical(all_studentsMat[col])
    all_studentsMat[col] = all_studentsMat[col].cat.codes


targetColumn = 'total_grades'
feature_cols = ["failures", "studytime","famsize","Pstatus","Medu","Fedu","absences","freetime", 'schoolsup', 'activities', 'higher', 'Walc', 'romantic', 'G1', 'G2', 'G3' ]


X_AllColumn = all_studentsMat[feature_cols]
Y_FinalGrade = all_studentsMat[targetColumn]

X_train, X_test, y_train, Y_test = train_test_split(X_AllColumn, Y_FinalGrade, test_size=0.3, random_state=0)

#Let's try different prediction model

#First
print("#--------------#")
print("LinearRegression")
print("#--------------#")
print()

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# The coefficients
print('Coefficients : \n', model.coef_)
# Explained variance score: 1 is perfect prediction
print("LinearRegression --> model score =",model.score(X=X_test, y=Y_test))


#Second
print()
print("#--------------------#")
print("RandomForestClassifier")
print("#--------------------#")
print()

model = RandomForestClassifier(n_estimators=80,criterion="entropy",random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#Get importance of features in classification
importances = model.feature_importances_

#Get indice of each importances to show them on plot
indices = np.argsort(importances)

#Get columns name from indices
names = [X_train.columns[i] for i in indices]

#Prepare the diagram from X_Train
plt.figure(figsize=(20,20))
plt.bar(range(X_train.shape[1]), importances[indices],width=0.5)
plt.xticks(range(X_train.shape[1]),names, rotation=60, fontsize = 12)
# Create plot title
plt.title("Feature Importance RandomForest")
# Show plot
plt.show()

print("RandomForestClassifier --> model score =",model.score(X=X_test, y=Y_test))

#Third
print()
print("#----------------------#")
print(" DecisionTreeClassifier ")
print("#----------------------#")
print()

model = DecisionTreeClassifier(min_samples_split=20, random_state=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

dot_data = tree.export_graphviz(model, out_file=None, rounded=True, feature_names=feature_cols)

graph = graphviz.Source(dot_data)
graph.render("Student")

print("DecisionTreeClassifier --> model score =",model.score(X=X_test, y=Y_test))

#Third
print()
print("#----------------------------#")
print("DecisionTreeClassifier Entropy")
print("#----------------------------#")
print()

model = DecisionTreeClassifier(criterion="entropy", random_state=5, max_depth=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

dot_data = tree.export_graphviz(model, out_file=None, rounded=True, feature_names=feature_cols)

graph = graphviz.Source(dot_data)
graph.render("Student_entropy")

print("DecisionTreeClassifier Entropy --> model score =",model.score(X=X_test, y=Y_test))
