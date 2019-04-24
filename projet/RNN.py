# Importing the libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
# Importing the dataset
### Intégration des datasets
file = pd.read_csv("student-mat.csv", encoding="ISO-8859-1", delimiter=";")  # Ensemble mathématiques
file2 = pd.read_csv("student-por.csv", encoding="ISO-8859-1", delimiter=";")  # Ensemble portugais
file3 = pd.concat([file, file2])  # Concaténation des deux ensembles
dataframe1 = pd.DataFrame(file2)  # Transformation en dataframe (protugais)
dataframe2 = pd.DataFrame(file)  # Transformation en dataframe (portugais)
dataframe3 = pd.DataFrame(file3)  # Transformation en dataframe

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values
