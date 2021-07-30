### Gaussian Process ###
# Import der Module
import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import KFold
import class_functions as fun


#Laden des Datensatzes
dataset = pd.read_csv("/Users/Marcel Herold/Desktop/DataAnalytics/DataAnalyticsDatensatz.csv")

#Aufteilen des Datensatzes in abhängige und unabhängige Variablen
X = dataset[['Preisspanne', 'Kilometerstand', 'Leistung', 'Kraftstoff', 'Tueren', 'Halter' ]] 
y = dataset['FairJaNein'] # Vorhersage Variable

#umwandeln der Vorherzusagenden Variable in Integer (0 und 1): 0 = fairer Preis, 1 = unfairer Preis
y = dataset['FairJaNein'].replace(to_replace = ['ja', 'nein'], value=[0, 1])



## Klassifizierung 
name = "Gaussian Process"


cv = KFold(n_splits=3, random_state=1, shuffle=True)
# Initiieren des Klassifizierungsmodells
clas = GaussianProcessClassifier()

# Modell anpassen
clas.fit(X, y)


##Gütemaße ermitteln
fun.validitation(clas, X, y, cv, name)


##Confusion Matrix
fun.cofuMatrix(clas, X, y, cv, name)