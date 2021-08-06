import numpy as np
import pandas as pd





##Data preperation Original Data
datasetOriginal = pd.read_csv("./Datensätze/DataOriginalUnedit.csv")

#Getriebe in GetriebeNum
datasetOriginal["Manuell"] = np.nan
datasetOriginal["Automatik"] = np.nan

datasetOriginal.loc[datasetOriginal['Getriebe'] == 'Manuell', 'Manuell'] = 1
datasetOriginal.loc[datasetOriginal['Getriebe'] == 'Automatik', 'Manuell'] = 0

datasetOriginal.loc[datasetOriginal['Getriebe'] == 'Manuell', 'Automatik'] = 0
datasetOriginal.loc[datasetOriginal['Getriebe'] == 'Automatik', 'Automatik'] = 1


#Kraftstof in KraftstofNum
datasetOriginal["Benzin"] = np.nan
datasetOriginal["Diesel"] = np.nan
datasetOriginal["Elektro"] = np.nan

datasetOriginal.loc[datasetOriginal['Kraftstoff'] == 'Benzin', 'Benzin'] = 1
datasetOriginal.loc[datasetOriginal['Kraftstoff'] == 'Diesel', 'Benzin'] = 0
datasetOriginal.loc[datasetOriginal['Kraftstoff'] == 'Elektro', 'Benzin'] = 0

datasetOriginal.loc[datasetOriginal['Kraftstoff'] == 'Benzin', 'Diesel'] = 0
datasetOriginal.loc[datasetOriginal['Kraftstoff'] == 'Diesel', 'Diesel'] = 1
datasetOriginal.loc[datasetOriginal['Kraftstoff'] == 'Elektro', 'Diesel'] = 0

datasetOriginal.loc[datasetOriginal['Kraftstoff'] == 'Benzin', 'Elektro'] = 0
datasetOriginal.loc[datasetOriginal['Kraftstoff'] == 'Diesel', 'Elektro'] = 0
datasetOriginal.loc[datasetOriginal['Kraftstoff'] == 'Elektro', 'Elektro'] = 1

#Null Halter löschen
datasetOriginal = datasetOriginal[datasetOriginal.Halter != 0]

print(datasetOriginal)

datasetOriginal.to_csv("./Datensätze/DatasetOriginal.csv", index=False)





##Data preperation Modified Data
datasetModified = pd.read_csv("./Datensätze/DataModifiedUnedit.csv")

#Kilometerspanne
datasetModified["Kilometerspanne"] = round(datasetModified["Kilometerstand"]/10000)*10000
#Preisspanne
datasetModified["Preisspanne"] = round(datasetModified["Preis"]/1000)*1000

#Getriebe in GetriebeNum
datasetModified["Manuell"] = np.nan
datasetModified["Automatik"] = np.nan

datasetModified.loc[datasetModified['Getriebe'] == 'Manuell', 'Manuell'] = 1
datasetModified.loc[datasetModified['Getriebe'] == 'Automatik', 'Manuell'] = 0

datasetModified.loc[datasetModified['Getriebe'] == 'Manuell', 'Automatik'] = 0
datasetModified.loc[datasetModified['Getriebe'] == 'Automatik', 'Automatik'] = 1


#Kraftstof in KraftstofNum
datasetModified["Benzin"] = np.nan
datasetModified["Diesel"] = np.nan
datasetModified["Elektro"] = np.nan

datasetModified.loc[datasetModified['Kraftstoff'] == 'Benzin', 'Benzin'] = 1
datasetModified.loc[datasetModified['Kraftstoff'] == 'Diesel', 'Benzin'] = 0
datasetModified.loc[datasetModified['Kraftstoff'] == 'Elektro', 'Benzin'] = 0

datasetModified.loc[datasetModified['Kraftstoff'] == 'Benzin', 'Diesel'] = 0
datasetModified.loc[datasetModified['Kraftstoff'] == 'Diesel', 'Diesel'] = 1
datasetModified.loc[datasetModified['Kraftstoff'] == 'Elektro', 'Diesel'] = 0

datasetModified.loc[datasetModified['Kraftstoff'] == 'Benzin', 'Elektro'] = 0
datasetModified.loc[datasetModified['Kraftstoff'] == 'Diesel', 'Elektro'] = 0
datasetModified.loc[datasetModified['Kraftstoff'] == 'Elektro', 'Elektro'] = 1

#Null Halter löschen
datasetModified = datasetModified[datasetModified.Halter != 0]

print(datasetModified)


datasetModified.to_csv("./Datensätze/DatasetModified.csv", index=False)
