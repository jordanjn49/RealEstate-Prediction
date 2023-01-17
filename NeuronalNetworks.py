import numpy as np  # pour utiliser des matrices
import matplotlib.pyplot as plt  # pour afficher des courbes
import pandas as pd
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils import column_or_1d


def early_stop(test_erreurs):
    minimum = max(test_erreurs)
    minimumIndex = 0

    for i in range(len(test_erreurs)):
        if test_erreurs[i] < minimum :
           minimum = test_erreurs[i]
           minimumIndex = i

    for i in range(minimumIndex, len(test_erreurs)):
        if test_erreurs[i] >= minimum*1.25:
            return True

    return False

def neuronalNetworks():
    np.random.seed(7)  # pour la réproductibilité des simulations

    donnees_data_frame = pd.read_csv('DATASET-Final.csv', delimiter=",")

    selected_vars = [4, 7, 8, 9, 13, 14, 15, 16]
    selected_vars = [k - 1 for k in selected_vars]
    donnees_data_frame = donnees_data_frame[donnees_data_frame.columns[selected_vars]]

    donnees_ensemble_total = donnees_data_frame.values

    # mélange du tableau numpy  (mélange des lignes)
    np.random.shuffle(donnees_ensemble_total)
    print(donnees_ensemble_total)

    nombre_lignes_base = donnees_ensemble_total.shape[0]
    nombre_colonnes_base = donnees_ensemble_total.shape[1]

    x_train = donnees_ensemble_total[0:round(nombre_lignes_base * 2 / 3 * 2 / 3), :donnees_ensemble_total.shape[1] - 1]
    y_train = donnees_ensemble_total[0:round(nombre_lignes_base * 2 / 3 * 2 / 3), donnees_ensemble_total.shape[1] - 1:]
    y_train = column_or_1d(y_train, warn=False)
    tableau_erreurs_train = np.shape(0)

    x_validation = donnees_ensemble_total[
                   round(nombre_lignes_base * 2 / 3 * 2 / 3) + 1:round(nombre_lignes_base * 2 / 3),
                   :donnees_ensemble_total.shape[1] - 1]
    y_validation = donnees_ensemble_total[
                   round(nombre_lignes_base * 2 / 3 * 2 / 3) + 1:round(nombre_lignes_base * 2 / 3),
                   donnees_ensemble_total.shape[1] - 1:]
    y_validation = column_or_1d(y_validation, warn=False)
    tableau_erreurs_validation = np.shape(0)

    x_test = donnees_ensemble_total[round(nombre_lignes_base * 2 / 3) + 1:, :donnees_ensemble_total.shape[1] - 1]
    y_test = donnees_ensemble_total[round(nombre_lignes_base * 2 / 3) + 1:, donnees_ensemble_total.shape[1] - 1:]
    y_test = column_or_1d(y_test, warn=False)
    tableau_erreurs_test = np.shape(0)

    # instanciation
    scaler = StandardScaler(with_mean=True, with_std=True)

    # calcul des paramètres de centrage-réduction :
    scaler.fit(x_train)

    # application à cet ensemble d'apprentissage
    x_train = scaler.transform(x_train)
    x_validation = scaler.transform(x_validation)
    x_test = scaler.transform(x_test)

    nbr_neurones = 8

    erreur_validation_la_plus_basse = 100
    early_stopping = False

    pause = "n"
    nbr_iterations = 40
    i = 0
    while (i < 50) and (not early_stopping):
        model = MLPRegressor(hidden_layer_sizes=[nbr_neurones], activation='tanh', solver='lbfgs', random_state=7,
                             max_iter=nbr_iterations, tol=1e-6)
        model.fit(x_train, y_train)

        y_predit_train = model.predict(x_train)
        y_predit_validation = model.predict(x_validation)
        y_predit_test = model.predict(x_test)

        print("Taux de reconnaissance en test :")
        print("----------------------")
        print("Taux d'erreur en test : " + str((100 - (metrics.r2_score(y_test, y_predit_test) * 100))) + "% \n")

        if i == 0:
            tableau_erreurs_train = np.array(100 - metrics.r2_score(y_train, y_predit_train) * 100)
            tableau_erreurs_validation = np.array(100 - metrics.r2_score(y_validation, y_predit_validation) * 100)
            tableau_erreurs_test = np.array(100 - metrics.r2_score(y_test, y_predit_test) * 100)
        else:
            tableau_erreurs_train = np.append(tableau_erreurs_train,
                                              100 - metrics.r2_score(y_train, y_predit_train) * 100)
            tableau_erreurs_validation = np.append(tableau_erreurs_validation,
                                                   100 - metrics.r2_score(y_validation, y_predit_validation) * 100)
            tableau_erreurs_test = np.append(tableau_erreurs_test, 100 - metrics.r2_score(y_test, y_predit_test) * 100)

        i = i + 1
        nbr_iterations = nbr_iterations + 40

        plt.plot(tableau_erreurs_train, label="train")
        plt.plot(tableau_erreurs_validation, label="validation")
        plt.plot(tableau_erreurs_test, label="test")
        plt.legend()
        plt.title("Erreurs avec " + str(nbr_neurones) + " neurones et " + str(nbr_iterations) + " itérations.")
        plt.show()

        ###############################################
        # gestion de l'early-stopping :
        ##############################################
        if i > 5:  # on voudrait pas que ça ne stoppe trop vite...
            erreur_validation_courante = tableau_erreurs_validation[tableau_erreurs_validation.size - 1]
            erreur_validation_la_plus_basse = min(tableau_erreurs_validation)

            print("erreur_validation_courante =" + str(erreur_validation_courante))
            print("erreur_validation_la_plus_basse =" + str(min(tableau_erreurs_validation)))

            if erreur_validation_courante >= 1.20 * erreur_validation_la_plus_basse:  # i>5 pour laisser le réseau s'élancer
                early_stopping = True
                print("Stop ça remonte !!!")

if __name__ == '__main__':
    print(neuronalNetworks())