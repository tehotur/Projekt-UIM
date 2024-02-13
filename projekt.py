import os
import pickle
import numpy as np
import csv
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from statistics import mean
import scipy.stats as stats
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def DataPreprocessing(data):
    """
    Funkce slouží pro předzpracování dat. V datech byly NaN hodnoty, které byly zpracovány a doplněny 0 hodnotami.

    :parameter inputData:
        Vstupni data, ktera se budou predzpracovavat.
    :return preprocessedData:
        Predzpracovana data na vystup
    """
    # Odstranění nepotřebných hodnot (Demografické charakteristiky)
    data = data.drop(columns=['Age',
                      'Gender',
                      'Unit1',
                      'Unit2',
                      'HospAdmTime',
                      'ICULOS'], axis=1)

    # Rozdělení na septické a neseptické
    non_sepsis = data.loc[data['isSepsis'] == 0]
    is_sepsis = data.loc[data['isSepsis'] == 1]

    # Průměrný počet NaN hodnot v řádku je 14.7
    mean_non = mean(non_sepsis.isnull().sum(axis=1))

    # Vybrání řádků s méně než 5 hodnotami NaN
    non_sepsis_nan = non_sepsis[non_sepsis.isnull().sum(axis=1) < mean_non - 5]

    # Sloučení septických a upravených neseptických dat
    new_data = pd.concat([is_sepsis, non_sepsis_nan])

    # DataFrame
    df = pd.DataFrame(new_data)

    # Doplnění chybějících hodnot 0
    df = df.fillna(0)

    # Promíchání řádků
    df = df.sample(frac=1)

    # Škalování dat - Z-skor
    zscores = stats.zscore(df)
    return zscores


def MyModel(inputData):
    """
    Funkce pro algoritmy strojového učení, který založen na tom, že vlastnosti datového bodu lze předpovědět
    na základě vlastností jeho sousedů - k-Nearest Neighbors (kNN).

    :param inputData:
        Vstupni data; vzdy jde o jeden objekt pro vyhodnoceni (1 pacient)
    :return outputClass:
        Vystupni trida objektu
    """

    # Rozdělení na klasifikační skupiny a data
    y = inputData['isSepsis']
    X = inputData.drop(['isSepsis'], axis=1)


    # Encoder - převedení hodnot do 0 a 1
    lab_enc = preprocessing.LabelEncoder()
    y_encoded = lab_enc.fit_transform(y)

    # Rozdělení na testovací a trénovací množinu
    # 75% dat pro trénování a 25% dat pro testování test_size=0,25
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded,
                                                        test_size=0.25, random_state=42)

    # PCA
    # Záchování 80% dat
    pca = PCA(n_components=0.80)

    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # kNN klasifikace
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=7)
    classifier.fit(X_train, y_train)

    # Uložení modelu
    file_name = "knn"
    pickle.dump(classifier, open(file_name, 'wb'))

    # Načtení modelu
    outputClass = pickle.load(open('knn', 'rb'))
    y_pred = outputClass.predict(X_test)
    return y_pred, y_test

def GetScoreSepsis(y_pred, y_test):
    """
    Funkce pro vyhodnocování úspěšnosti strojového učení.

    :param confMatrix: Vstupni matice zamen
    :return:
        se:     Senzitivita modelu
        sp:     Specificita modelu
        acc:    Presnost modelu (acccuracy)
        fScore: F1 skore modelu
        ppv:    Pozitivni prediktivni hodnota
    """

    # Přidání názvů tříd pro lepší interpretaci
    classes_names = ['Positive', 'Negative']
    confMatrix = pd.DataFrame(confusion_matrix(y_test, y_pred),
                      columns=classes_names, index=classes_names)

    # Seabornova heatmap pro lepší vizualizaci matice záměny
    sns.heatmap(confMatrix, annot=True, fmt='d')
    print(classification_report(y_test, y_pred))

    return confMatrix

def main():
    filePath = os.path.abspath("dataSepsis.csv")
    data = pd.read_csv(filePath, delimiter=';')
    predspracovane_data = DataPreprocessing(data)
    y_pred, y_test = MyModel(predspracovane_data)
    matrix = GetScoreSepsis(y_pred, y_test)
    return matrix


if __name__ == "__main__":
        main()