# Import der Module
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score




##Confusion Matrix
def cofuMatrix(clas, X, y, cv, name):
    y_pred_class = cross_val_predict(clas, X, y, cv=cv)
    confusion_matrix = metrics.confusion_matrix(y, y_pred_class, labels=[1,0])
    fig, ax = plt.subplots(figsize=(9, 7))
    x_axis_labels = ['Fairer Preis', 'Unfairer Preis']
    y_axis_labels = ['Fairer Preis', 'Unfairer Preis']
    plt.title("Fairer Preis | Unfairer Preis – Confusion Matrix - " + str(name))
    plt.xlabel('Vorhergesagte Werte')
    plt.ylabel('Tatsächliche Werte')
    sn.heatmap(confusion_matrix, xticklabels=x_axis_labels, yticklabels=y_axis_labels, annot=True, cmap="YlGn")
    plt.show()
    nutzen = confusion_matrix[0,0] * 500 + confusion_matrix[1,0] * -500
    print("Der ökonomische Nutzen für den Händler liegt also bei: " + str(nutzen) + "€")
    


##Gütemaße  
def validitation(clas, X, y, cv, name):
    print("Gütemaße für die " + str(name) + " Klassifikation:")
    #Accuracy
    accuracy = cross_val_score(clas, X, y, cv=cv, scoring='accuracy')
    print("Korrekt klassifiziert:", round(accuracy.mean(), 2))
    print("Falsch klassifiziert:", round(1 - accuracy.mean(), 2))
    #Precision
    precision = cross_val_score(clas, X, y, cv=cv, scoring='precision')
    print("Precision:", round(precision.mean(), 2))
    #Recall
    recall = cross_val_score(clas, X, y, cv=cv, scoring='recall')
    print("Recall:", round(recall.mean(), 2))
    #F1-Score
    f1 = cross_val_score(clas, X, y, cv=cv, scoring='f1')
    print("F1-Score:", round(f1.mean(), 2))
    #Matthew coefficient
    mcc = cross_val_predict(clas, X, y, cv=cv)
    print('MCC:', round(matthews_corrcoef(y, mcc), 2))
    #AUC
    auc = cross_val_score(clas, X, y, cv=cv, scoring='roc_auc')
    print("AUC:", round(auc.mean(), 2))
    


