import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from sklearn import tree
import graphviz
from sklearn.metrics import confusion_matrix, precision_score




#Class grade in 3 parts, low, average and High
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def marks(total_grades):
    if(total_grades < 7):
        return("Low")
    if(total_grades >= 7 and total_grades < 14):
        return("Average")
    if(total_grades >= 14):
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
    student_Without_All_Grad = all_students.drop(["G1", "G2", "G3"], axis=1)
    max = student_Without_All_Grad["total_grades"].max()
    min = student_Without_All_Grad["total_grades"].min()

    print("Max value is : " + str(max))
    print("Min grade is : " + str(min))

    # Create last colum as the class column
    student_Without_All_Grad["grades"] = student_Without_All_Grad["total_grades"].apply(marks)

    return student_Without_All_Grad


def getPValueScore(student_data, analyzeData, title):


    # print(student_data[analyzeData].dtype)
    #
    # type = student_data[analyzeData].unique()
    # typeInt = student_data[analyzeData].unique()
    #
    # i = 0;
    # print("We replace " + analyzeData + " by numbers")
    # for cat in type:
    #     print(cat, " got value ", i)
    #     typeInt[i] = i
    #     student_data[analyzeData] = student_data[analyzeData].str.replace(cat, str(i))
    #     i = i + 1
    # analyzeData = str(analyzeData)

    # p-value between total grade and attribute
    pearson_coef, p_value = stats.pearsonr(student_data[analyzeData], student_data["total_grades"])
    print("The Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value, "for students in", title)

all_studentsMat = mergeMarksData("student-mat.csv")
all_studentsPor = mergeMarksData("student-por.csv")

showAverageMarks(all_studentsMat, "Math")
showAverageMarks(all_studentsPor, "Portuguese")

plt.show()


getPValueScore(all_studentsMat, "absences", "Math")
getPValueScore(all_studentsPor, "absences", "Portuguese")


#split dataset in features and target variable
feature_cols = ['studytime', 'absences', 'freetime', 'goout', 'total_grades']
X = all_studentsMat[feature_cols] # Features
y = all_studentsMat.values# Target variable

y = all_studentsMat.values# Target variable

X_train, X_test, y_train, y_test = train_test_split(all_studentsMat, y, test_size = 0.2, random_state = 0)

dt = DecisionTreeClassifier(min_samples_split=20, random_state=5)
dt.fit(X_train, y_train)


dot_data = tree.export_graphviz(dt, out_file=None, rounded=True, feature_names=feature_cols)

graph = graphviz.Source(dot_data)
graph.render("Classification")
#
# clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=5, max_depth=3)
# clf_entropy.fit(X_train, y_train)
#ValueError: could not convert string to float: 'GP'

# dot_data = tree.export_graphviz(clf_entropy, out_file=None, rounded=True, feature_names=feature_cols)
#
# graph = graphviz.Source(dot_data)
# graph.render("Classification_entropy")
#
#
# y_predicted = clf_entropy.predict(X_test.astype(int))
# print("Precision score :", precision_score(y_test[:, 0], y_predicted[:, 0]))