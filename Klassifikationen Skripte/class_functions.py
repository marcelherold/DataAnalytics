# Import der Module
from sklearn import metrics
import seaborn as sn
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn import tree

font = {'weight' : 'bold',
        'size'   : 15}

matplotlib.rc('font', **font)



##Confusion Matrix
def cofuMatrix(clas, X, y, cv, name):
    y_pred_class = cross_val_predict(clas, X, y, cv=cv)
    confusion_matrix = metrics.confusion_matrix(y, y_pred_class, labels=[1,0])
    fig, ax = plt.subplots(figsize=(9, 7))
    x_axis_labels = ['Fairer Preis', 'Unfairer Preis']
    y_axis_labels = ['Fairer Preis', 'Unfairer Preis']
    plt.title("Confusion Matrix - " + str(name), font = {'size'   : 14})
    plt.xlabel('Vorhergesagte Werte')
    plt.ylabel('Tatsächliche Werte')
    sn.heatmap(confusion_matrix, xticklabels=x_axis_labels, yticklabels=y_axis_labels, annot=True, cmap="YlGn")
    plt.show()
    
    


##Kosten Nutzen Analyse
def knAnalysis(clas, X, y, cv, name):
    #Confusion Matrix
    y_pred_class = cross_val_predict(clas, X, y, cv=cv)
    confusion_matrix = metrics.confusion_matrix(y, y_pred_class, labels=[1,0])
    #Gewinn/Verlust verkauf der Autos
    gewinn = 2000
    verlust = -1000  
    #Kostenanalyse, wenn ohne Verfahren alle Fahrzeuge gekauft werden 
    gesamtNutzen = sum(confusion_matrix[0]) * gewinn + sum(confusion_matrix[1]) * verlust
    #Kostenanalyse, wenn nur faire Autos des Verfahrens gekauft werden
    clasNutzen= confusion_matrix[0,0] * gewinn + confusion_matrix[1,0] * verlust
    bewertung = clasNutzen - gesamtNutzen
    #Ausgaben der Kosten Nutzen Analyse
    print("Kosten-Nutzen Analyse der " + str(name) + " Klassifikation:")
    print("Der erwartete Gewinn, wenn einfach alle Autos gekauft werden: " + str(gesamtNutzen) + "€")
    print("Der erwartet Gewinn mit dem Verfahren liegt bei: " + str(clasNutzen) + "€")
    print("Der ökonomische Nutzen des Verfahrens für den Händler liegt also bei: " + str(bewertung) + "€")
    


##Gütemaße  
def validitation(clas, X, y, cv, name):
    print("Gütemaße für die " + str(name) + " Klassifikation:")
    #Accuracy
    accuracy = cross_val_score(clas, X, y, cv=cv, scoring='accuracy')
    print("Korrekt klassifiziert (Accuracy):", round(accuracy.mean(), 2))
    print("Falsch klassifiziert:", round(1 - accuracy.mean(), 2))
    #Precision
    precision = cross_val_score(clas, X, y, cv=cv, scoring='precision')
    print("Precision:", round(precision.mean(), 2))
    #Recall
    recall = cross_val_score(clas, X, y, cv=cv, scoring='recall')
    print("Recall:", round(recall.mean(), 2))
    #F1-Score
    f1 = cross_val_score(clas, X, y, cv=cv, scoring='f1')
    print("F-Measure:", round(f1.mean(), 2))
    #Matthew coefficient
    mcc = cross_val_predict(clas, X, y, cv=cv)
    print('MCC:', round(matthews_corrcoef(y, mcc), 2))
    #AUC
    auc = cross_val_score(clas, X, y, cv=cv, scoring='roc_auc')
    print("ROC-AUC:", round(auc.mean(), 2))



def accuracy(clas, X, y, cv, name):
    #Accuracy
    accuracy = cross_val_score(clas, X, y, cv=cv, scoring='accuracy')
    print("Korrekt klassifiziert (Accuracy) bei " + str(name) + ":", round(accuracy.mean(), 2))
 

def plotTree(clas, X, y, cv, name):
    plt.figure(figsize=(30,10))  # set plot size (denoted in inches)
    tree.plot_tree(clas, fontsize=10)
    plt.show()
