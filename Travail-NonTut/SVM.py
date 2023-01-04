import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import column_or_1d
from sklearn import metrics
from sklearn.svm import SVC

def SVM():
    np.random.seed(7)

    # Récupération des données d'entraînement
    donnees_data_frame = pd.read_csv ("train.txt" , delimiter=" ")
    donnees_data_frame_eval = pd.read_csv('test_EVALUATION.txt', delimiter=" ")  # lecture du fichier de validation
    donnees_ensemble_total = donnees_data_frame.values

    donnees_ensemble_total_eval = donnees_data_frame_eval.values
    x_evaluation = donnees_ensemble_total_eval[:, :donnees_ensemble_total.shape[1] - 1]


    # On mélange ces données aléatoirement
    np.random.shuffle(donnees_ensemble_total)

    # On sépare les données de train (3/4) et de test (1/4)
    nombre_lignes = donnees_ensemble_total.shape[0]

    x_train = donnees_ensemble_total[0:round(nombre_lignes * 3 / 4), :donnees_ensemble_total.shape[1] - 1]
    y_train = donnees_ensemble_total[0:round(nombre_lignes * 3 / 4), donnees_ensemble_total.shape[1] - 1:]
    y_train = column_or_1d(y_train, warn=False)

    x_test = donnees_ensemble_total[round(nombre_lignes * 3 / 4) + 1:, :donnees_ensemble_total.shape[1] - 1]
    y_test = donnees_ensemble_total[round(nombre_lignes * 3 / 4) + 1:, donnees_ensemble_total.shape[1] - 1:]
    y_test = column_or_1d(y_test, warn=False)

    # On normalise ces données pour être contenue entre 0 et 1
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    x_evaluation = scaler.transform(x_evaluation)

    #importation de la classe calcul
    svm = SVC(gamma='auto')

    print ("\n\n===================\n")

    # Apprentissage - construction du modèle prédictif
    svm.fit(x_train, y_train)

    # On détermine les valeurs de test prédites par le SVM
    y_predit_test = svm.predict(x_test)

    # Evaluation du taux d'erreur (résultats prédits vs résultats attendus)
    err = (1.0 -metrics.accuracy_score (y_test ,y_predit_test ))*100
    print ("Erreur = ", round (err,2), "%" )
    print ("\n\n===================")

    y_predit_evaluation = svm.predict(x_evaluation)

    return y_predit_evaluation

if __name__ == '__main__':
    print(SVM())