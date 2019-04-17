import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

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
dataframe2 = pd.DataFrame(file) #Transformation en dataframe (protugais)
dataframe3 = pd.DataFrame(file3) #Transformation en dataframe (protugais)
### Fin de l'intégration des datasets

# Variable de prédiction : bon élève (grade bon ou pas)
sns.countplot(x="sex", data=dataframe3, palette="hls")
plt.show()

count_no_sub = len(dataframe3[dataframe3['sex']=="F"])
count_sub = len(dataframe3[dataframe3['sex']=="M"])
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("percentage of female is", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("percentage of male", pct_of_sub*100)

print(dataframe3.groupby("sex").mean())
# Ajout de la variable de prédiction :
cat_vars=['bon_eleve']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(dataframe3[var], prefix=var)
    data1=dataframe3.join(cat_list)
    dataframe3=data1
cat_vars=['bon_eleve']
data_vars=dataframe3.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]
data_final=dataframe3[to_keep]
print(data_final.columns.values)

X = data_final.loc[:, data_final.columns != 'y']
y = data_final.loc[:, data_final.columns == 'y']
from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns
os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])
# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['y']==0]))
print("Number of subscription",len(os_data_y[os_data_y['y']==1]))
print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==1])/len(os_data_X))