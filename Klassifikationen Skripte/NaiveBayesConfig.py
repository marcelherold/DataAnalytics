###Naive Bayes Ökonomische Konfiguration
# Import der Module
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
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
name = "Naive Bayes Modified Data"
#n-splits=k   k-1 trainingsdata ein fold zur validierung
cv = KFold(n_splits=10, random_state=1, shuffle=True)
#Initiieren des Klassifizierungsmodells
clas = GaussianNB()
#Modell anpassen
clas.fit(X, y)
#Klassenvorhersage für den Testdatensatz
y_pred_class = cross_val_predict(clas, X, y, cv=cv, method='predict_proba')
##Kosten Nutzen Analyse ohne Konfiguration
fun.knAnalysis(clas, X, y, cv, name)


####Bester Schwellwert basierend auf Gewinn/Verlust
fpr, tpr, thresholds = metrics.roc_curve(y, y_pred_class[::,1], drop_intermediate=False)
#Zu besseren Handhabung Umwandlung in DataFrame und Ausgabe
df = pd.DataFrame([thresholds, fpr, tpr])
#Zeilen und Spalten werden zu besseren Übersichtlichkeit getauscht
df = df.T
df.columns=["Threshold","FPR", "TPR"]

#Gewinn und Verlust
gewinn = 2000
verlust = -1000
#Berechnung des Gewinns/Verlustes
fairerPreis = len(X[y==0])
unfairerPreis = len(X[y==1])
df["profit"] = (df["TPR"] * gewinn * fairerPreis) + (df["FPR"] * verlust * unfairerPreis)

#Gesamte Tabelle ausgeben
pd.set_option("display.max_rows", None, "display.max_columns", None)
print(df)
df.to_csv("./TabelleOekonomischeKonfig/NaiveBayesConfig.csv", ";",  index= False)

# Ausgabe der bestmöglichen Kombination
rownumber = df.profit == max(df.profit)
print("Optimaler Threshold: ", df["Threshold"][rownumber])
print("Errechneter Gewinn:", df["profit"][rownumber])
profit = df["profit"][rownumber].to_string(index=False)

##############################################

###ROC-Curve und Area under ROC
#Beste Kombination aus TPR und FPR in Variable einfügen
best_tpr = df['TPR'][rownumber]
best_fpr = df['FPR'][rownumber]
auc = cross_val_score(clas, X, y, cv=cv, scoring='roc_auc').mean()

#Fenster initiieren
fig, ax = plt.subplots(figsize=(11, 6))
# ROC Kurve Plotten und höchsten Profit-Threshold markieren
plt.title('ROC Curve – Naive Bayes')
plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
plt.plot(fpr, tpr, marker='.', color='green', label='Naive Bayes')
plt.plot(fpr,tpr,label="AUC=" + str(auc),color='green')
plt.scatter(best_fpr, best_tpr, marker='o', color='black', label="Höchster Gewinn:" + str(profit) + ' €')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

##############################################

##Klassifizierung mit angepasstem Threshold
name = "Naive Bayes Konfiguriert"
#n-splits=k   k-1 trainingsdata ein fold zur validierung
cv = KFold(n_splits=10, random_state=1, shuffle=True)
#Initiieren des Klassifizierungsmodells
clas = GaussianNB(priors= (0.1044,1-0.1044))
#clas = GaussianNB(priors= (df["Threshold"][rownumber],1-df["Threshold"][rownumber]))
#Modell anpassen
clas.fit(X, y)

##Gütemaße ermitteln
fun.validitation(clas, X, y, cv, name)

##Kosten Nutzen Analyse
fun.knAnalysis(clas, X, y, cv, name)

##Confusion Matrix
fun.cofuMatrix(clas, X, y, cv, name)