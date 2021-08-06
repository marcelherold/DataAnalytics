### Check, ob die Accuracy beim modifizieren Datensatz besser ist als beim Original Datensatz

# Import der Module
import pandas as pd
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import KFold
import class_functions as fun

#Laden der Datensätze
datasetOriginal = pd.read_csv("./Datensätze/DatasetOriginal.csv")
datasetModified = pd.read_csv("./Datensätze/DatasetModified.csv")


##Decition Tree
### ORIGINAL DATASET 10-FOLD ###

##Vorbereitung Daten
#Aufteilen des Datensatzes in abhängige und unabhängige Variablen
X = datasetOriginal[['Preis', 'Kilometerstand', 'Leistung', 'Erstzulassung', 'Benzin', 'Diesel', 'Elektro', 'Manuell', 'Automatik', 'Tueren', 'Halter']]
y = datasetOriginal['FairJaNein'] #Vorhersage Variable
#umwandeln der Vorherzusagenden Variable in Integer (0 und 1): 0 = fairer Preis, 1 = unfairer Preis
y = datasetOriginal['FairJaNein'].replace(to_replace = ['ja', 'nein'], value=[0, 1])

##Klassifizierung
name = "Decision Tree Original Data"
#n-splits=k   k-1 trainingsdata ein fold zur validierung
cv = KFold(n_splits=10, random_state=1, shuffle=True)
#Initiieren des Klassifizierungsmodells
clas = DTC(random_state=1)
#Modell anpassen
clas.fit(X, y)

##Gütemaße ermitteln
fun.accuracy(clas, X, y, cv, name)

### MODIFIED DATASET 10-FOLD ###

##Vorbereitung Daten
#Aufteilen des Datensatzes in abhängige und unabhängige Variablen
X = datasetModified[['Preisspanne', 'Kilometerspanne', 'Leistung', 'Erstzulassung', 'Benzin', 'Diesel', 'Elektro', 'Manuell', 'Automatik', 'Tueren', 'Halter']]
y = datasetModified['FairJaNein'] #Vorhersage Variable
#umwandeln der Vorherzusagenden Variable in Integer (0 und 1): 0 = fairer Preis, 1 = unfairer Preis
y = datasetModified['FairJaNein'].replace(to_replace = ['ja', 'nein'], value=[0, 1])

##Klassifizierung
name = "Decision Tree Modified Data"
#n-splits=k   k-1 trainingsdata ein fold zur validierung
cv = KFold(n_splits=10, random_state=1, shuffle=True)
#Initiieren des Klassifizierungsmodells
clas = DTC(random_state=1)
#Modell anpassen
clas.fit(X, y)

##Gütemaße ermitteln
fun.accuracy(clas, X, y, cv, name)

################################

##Gaussian Process
### ORIGINAL DATASET 10-FOLD ###

##Vorbereitung Daten
#Aufteilen des Datensatzes in abhängige und unabhängige Variablen
X = datasetOriginal[['Preis', 'Kilometerstand', 'Leistung', 'Erstzulassung', 'Benzin', 'Diesel', 'Elektro', 'Manuell', 'Automatik', 'Tueren', 'Halter']]
y = datasetOriginal['FairJaNein'] #Vorhersage Variable
#umwandeln der Vorherzusagenden Variable in Integer (0 und 1): 0 = fairer Preis, 1 = unfairer Preis
y = datasetOriginal['FairJaNein'].replace(to_replace = ['ja', 'nein'], value=[0, 1])

##Klassifizierung
name = "Gaussian Process Original Data"
#n-splits=k   k-1 trainingsdata ein fold zur validierung
cv = KFold(n_splits=10, random_state=1, shuffle=True)
#Initiieren des Klassifizierungsmodells
clas = GaussianProcessClassifier()
#Modell anpassen
clas.fit(X, y)

##Gütemaße ermitteln
fun.accuracy(clas, X, y, cv, name)

### MODIFIED DATASET 10-FOLD ###

##Vorbereitung Daten
#Aufteilen des Datensatzes in abhängige und unabhängige Variablen
X = datasetModified[['Preisspanne', 'Kilometerspanne', 'Leistung', 'Erstzulassung', 'Benzin', 'Diesel', 'Elektro', 'Manuell', 'Automatik', 'Tueren', 'Halter']]
y = datasetModified['FairJaNein'] #Vorhersage Variable
#umwandeln der Vorherzusagenden Variable in Integer (0 und 1): 0 = fairer Preis, 1 = unfairer Preis
y = datasetModified['FairJaNein'].replace(to_replace = ['ja', 'nein'], value=[0, 1])

##Klassifizierung
name = "Gaussian Process Modified Data"
#n-splits=k   k-1 trainingsdata ein fold zur validierung
cv = KFold(n_splits=10, random_state=1, shuffle=True)
#Initiieren des Klassifizierungsmodells
clas = GaussianProcessClassifier()
#Modell anpassen
clas.fit(X, y)

##Gütemaße ermitteln
fun.accuracy(clas, X, y, cv, name)

################################

##Logistic Regression
### ORIGINAL DATASET 10-FOLD ###

##Vorbereitung Daten
#Aufteilen des Datensatzes in abhängige und unabhängige Variablen
X = datasetOriginal[['Preis', 'Kilometerstand', 'Leistung', 'Erstzulassung', 'Benzin', 'Diesel', 'Elektro', 'Manuell', 'Automatik', 'Tueren', 'Halter']]
y = datasetOriginal['FairJaNein'] #Vorhersage Variable
#umwandeln der Vorherzusagenden Variable in Integer (0 und 1): 0 = fairer Preis, 1 = unfairer Preis
y = datasetOriginal['FairJaNein'].replace(to_replace = ['ja', 'nein'], value=[0, 1])

##Klassifizierung
name = "Logistic Regression Original Data"
#n-splits=k   k-1 trainingsdata ein fold zur validierung
cv = KFold(n_splits=10, random_state=1, shuffle=True)
#Initiieren des Klassifizierungsmodells
clas = LogisticRegression()
#Modell anpassen
clas.fit(X, y)

##Gütemaße ermitteln
fun.accuracy(clas, X, y, cv, name)

### MODIFIED DATASET 10-FOLD ###

##Vorbereitung Daten
#Aufteilen des Datensatzes in abhängige und unabhängige Variablen
X = datasetModified[['Preisspanne', 'Kilometerspanne', 'Leistung', 'Erstzulassung', 'Benzin', 'Diesel', 'Elektro', 'Manuell', 'Automatik', 'Tueren', 'Halter']]
y = datasetModified['FairJaNein'] #Vorhersage Variable
#umwandeln der Vorherzusagenden Variable in Integer (0 und 1): 0 = fairer Preis, 1 = unfairer Preis
y = datasetModified['FairJaNein'].replace(to_replace = ['ja', 'nein'], value=[0, 1])

##Klassifizierung
name = "Logistic Regression Modified Data"
#n-splits=k   k-1 trainingsdata ein fold zur validierung
cv = KFold(n_splits=10, random_state=1, shuffle=True)
#Initiieren des Klassifizierungsmodells
clas = LogisticRegression()
#Modell anpassen
clas.fit(X, y)

##Gütemaße ermitteln
fun.accuracy(clas, X, y, cv, name)

################################

##Naive Bayes
### ORIGINAL DATASET 10-FOLD ###

##Vorbereitung Daten
#Aufteilen des Datensatzes in abhängige und unabhängige Variablen
X = datasetOriginal[['Preis', 'Kilometerstand', 'Leistung', 'Erstzulassung', 'Benzin', 'Diesel', 'Elektro', 'Manuell', 'Automatik', 'Tueren', 'Halter']]
y = datasetOriginal['FairJaNein'] #Vorhersage Variable
#umwandeln der Vorherzusagenden Variable in Integer (0 und 1): 0 = fairer Preis, 1 = unfairer Preis
y = datasetOriginal['FairJaNein'].replace(to_replace = ['ja', 'nein'], value=[0, 1])

##Klassifizierung
name = "Naive Bayes Original Data"
#n-splits=k   k-1 trainingsdata ein fold zur validierung
cv = KFold(n_splits=10, random_state=1, shuffle=True)
#Initiieren des Klassifizierungsmodells
clas = GaussianNB()
#Modell anpassen
clas.fit(X, y)

##Gütemaße ermitteln
fun.accuracy(clas, X, y, cv, name)

### MODIFIED DATASET 10-FOLD ###

##Vorbereitung Daten
#Aufteilen des Datensatzes in abhängige und unabhängige Variablen
X = datasetModified[['Preisspanne', 'Kilometerspanne', 'Leistung', 'Erstzulassung', 'Benzin', 'Diesel', 'Elektro', 'Manuell', 'Automatik', 'Tueren', 'Halter']]
y = datasetModified['FairJaNein'] #Vorhersage Variable
#umwandeln der Vorherzusagenden Variable in Integer (0 und 1): 0 = fairer Preis, 1 = unfairer Preis
y = datasetModified['FairJaNein'].replace(to_replace = ['ja', 'nein'], value=[0, 1])

##Klassifizierung
name = "Naive Bayes Modified Data"
#n-splits=k   k-1 trainingsdata ein fold zur validierung
cv = KFold(n_splits=10, random_state=1, shuffle=True)
#Initiieren des Klassifizierungsmodells
clas = GaussianNB()
#Modell anpassen
clas.fit(X, y)

##Gütemaße ermitteln
fun.accuracy(clas, X, y, cv, name)

################################

##Nearest Neighbours
### ORIGINAL DATASET 10-FOLD ###

##Vorbereitung Daten
#Aufteilen des Datensatzes in abhängige und unabhängige Variablen
X = datasetOriginal[['Preis', 'Kilometerstand', 'Leistung', 'Erstzulassung', 'Benzin', 'Diesel', 'Elektro', 'Manuell', 'Automatik', 'Tueren', 'Halter']]
y = datasetOriginal['FairJaNein'] #Vorhersage Variable
#umwandeln der Vorherzusagenden Variable in Integer (0 und 1): 0 = fairer Preis, 1 = unfairer Preis
y = datasetOriginal['FairJaNein'].replace(to_replace = ['ja', 'nein'], value=[0, 1])

##Klassifizierung
name = "Nearest Neighbours Original Data"
#n-splits=k   k-1 trainingsdata ein fold zur validierung
cv = KFold(n_splits=10, random_state=1, shuffle=True)
#Initiieren des Klassifizierungsmodells
clas = KNeighborsClassifier(n_neighbors=5)
#Modell anpassen
clas.fit(X, y)

##Gütemaße ermitteln
fun.accuracy(clas, X, y, cv, name)

### MODIFIED DATASET 10-FOLD ###

##Vorbereitung Daten
#Aufteilen des Datensatzes in abhängige und unabhängige Variablen
X = datasetModified[['Preisspanne', 'Kilometerspanne', 'Leistung', 'Erstzulassung', 'Benzin', 'Diesel', 'Elektro', 'Manuell', 'Automatik', 'Tueren', 'Halter']]
y = datasetModified['FairJaNein'] #Vorhersage Variable
#umwandeln der Vorherzusagenden Variable in Integer (0 und 1): 0 = fairer Preis, 1 = unfairer Preis
y = datasetModified['FairJaNein'].replace(to_replace = ['ja', 'nein'], value=[0, 1])

##Klassifizierung
name = "Nearest Neighbours Modified Data"
#n-splits=k   k-1 trainingsdata ein fold zur validierung
cv = KFold(n_splits=10, random_state=1, shuffle=True)
#Initiieren des Klassifizierungsmodells
clas = KNeighborsClassifier(n_neighbors=5)
#Modell anpassen
clas.fit(X, y)

##Gütemaße ermitteln
fun.accuracy(clas, X, y, cv, name)

################################

##Support Vector Machines
### ORIGINAL DATASET 10-FOLD ###

##Vorbereitung Daten
#Aufteilen des Datensatzes in abhängige und unabhängige Variablen
X = datasetOriginal[['Preis', 'Kilometerstand', 'Leistung', 'Erstzulassung', 'Benzin', 'Diesel', 'Elektro', 'Manuell', 'Automatik', 'Tueren', 'Halter']]
y = datasetOriginal['FairJaNein'] #Vorhersage Variable
#umwandeln der Vorherzusagenden Variable in Integer (0 und 1): 0 = fairer Preis, 1 = unfairer Preis
y = datasetOriginal['FairJaNein'].replace(to_replace = ['ja', 'nein'], value=[0, 1])

##Klassifizierung
name = "Support Vector Machines Original Data"
#n-splits=k   k-1 trainingsdata ein fold zur validierung
cv = KFold(n_splits=10, random_state=1, shuffle=True)
#Initiieren des Klassifizierungsmodells
clas = svm.NuSVC()
#Modell anpassen
clas.fit(X, y)

##Gütemaße ermitteln
fun.accuracy(clas, X, y, cv, name)

### MODIFIED DATASET 10-FOLD ###

##Vorbereitung Daten
#Aufteilen des Datensatzes in abhängige und unabhängige Variablen
X = datasetModified[['Preisspanne', 'Kilometerspanne', 'Leistung', 'Erstzulassung', 'Benzin', 'Diesel', 'Elektro', 'Manuell', 'Automatik', 'Tueren', 'Halter']]
y = datasetModified['FairJaNein'] #Vorhersage Variable
#umwandeln der Vorherzusagenden Variable in Integer (0 und 1): 0 = fairer Preis, 1 = unfairer Preis
y = datasetModified['FairJaNein'].replace(to_replace = ['ja', 'nein'], value=[0, 1])

##Klassifizierung
name = "Support Vector Machines Modified Data"
#n-splits=k   k-1 trainingsdata ein fold zur validierung
cv = KFold(n_splits=10, random_state=1, shuffle=True)
#Initiieren des Klassifizierungsmodells
clas = svm.NuSVC()
#Modell anpassen
clas.fit(X, y)

##Gütemaße ermitteln
fun.accuracy(clas, X, y, cv, name)