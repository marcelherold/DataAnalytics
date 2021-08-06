### Support Vector Machines ###
# Import der Module
import pandas as pd
from sklearn import svm
from sklearn.model_selection import KFold
import class_functions as fun

#Laden der Datensätze
datasetModified = pd.read_csv("./Datensätze/DatasetModified.csv")

##Vorbereitung Daten
#Aufteilen des Datensatzes in abhängige und unabhängige Variablen
X = datasetModified[['Preisspanne', 'Kilometerspanne', 'Leistung', 'Erstzulassung', 'Benzin', 'Diesel', 'Elektro', 'Manuell', 'Automatik', 'Tueren', 'Halter']]
y = datasetModified['FairJaNein'] #Vorhersage Variable
#umwandeln der Vorherzusagenden Variable in Integer (0 und 1): 0 = fairer Preis, 1 = unfairer Preis
y = datasetModified['FairJaNein'].replace(to_replace = ['ja', 'nein'], value=[0, 1])

##Klassifizierung
name = "Support Vector Machines"
#n-splits=k   k-1 trainingsdata ein fold zur validierung
cv = KFold(n_splits=10, random_state=1, shuffle=True)
#Initiieren des Klassifizierungsmodells
clas = svm.NuSVC()
#Modell anpassen
clas.fit(X, y)


##Gütemaße ermitteln
fun.validitation(clas, X, y, cv, name)

##Kosten Nutzen Analyse
fun.knAnalysis(clas, X, y, cv, name)

##Confusion Matrix
fun.cofuMatrix(clas, X, y, cv, name)
