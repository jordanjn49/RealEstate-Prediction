import numpy as np  # pour utiliser des matrices
import matplotlib.pyplot as plt  # pour afficher des courbes
import pandas as pd
from sklearn.utils.validation import column_or_1d
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

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


def reseauNeuronne():
    np.random.seed(7)  # pour la réproductibilité des simulations

    donnees_data_frame = pd.read_csv('train.txt', delimiter=" ") # lecture du fichier
    donnees_data_frame_eval = pd.read_csv('test_EVALUATION.txt', delimiter=" ") # lecture du fichier de validation

    donnees_ensemble_total = donnees_data_frame.values
    np.random.shuffle(donnees_ensemble_total)

    nombre_lignes_base = donnees_ensemble_total.shape[0]

    x_train = donnees_ensemble_total[0:round(nombre_lignes_base * 3 / 4 * 3 / 4), :donnees_ensemble_total.shape[1] - 1]
    y_train = donnees_ensemble_total[0:round(nombre_lignes_base * 3 / 4 * 3 / 4), donnees_ensemble_total.shape[1] - 1:]
    y_train = column_or_1d(y_train, warn=False)
    tableau_erreurs_train = np.shape(0)

    x_validation = donnees_ensemble_total[round(nombre_lignes_base * 3 / 4 * 3 / 4) + 1:round(nombre_lignes_base * 3 / 4), :donnees_ensemble_total.shape[1] - 1]
    y_validation = donnees_ensemble_total[round(nombre_lignes_base * 3 / 4 * 3 / 4) + 1:round(nombre_lignes_base * 3 / 4), donnees_ensemble_total.shape[1] - 1:]
    y_validation = column_or_1d(y_validation, warn=False)
    tableau_erreurs_validation = np.shape(0)

    x_test = donnees_ensemble_total[round(nombre_lignes_base * 3 / 4) + 1:, :donnees_ensemble_total.shape[1] - 1]
    y_test = donnees_ensemble_total[round(nombre_lignes_base * 3 / 4) + 1:, donnees_ensemble_total.shape[1] - 1:]
    y_test = column_or_1d(y_test, warn=False)
    tableau_erreurs_test = np.shape(0)

    donnees_ensemble_total_eval = donnees_data_frame_eval.values
    x_evaluation = donnees_ensemble_total_eval[:, :donnees_ensemble_total.shape[1] - 1]

    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_validation = scaler.transform(x_validation)
    x_test = scaler.transform(x_test)
    x_evaluation = scaler.transform(x_evaluation)

    erreur_validation_la_plus_basse = 100
    early_stopping = False

    nbr_iterations = 40
    i = 0
    nbr_neurones = 6
    while (i < 50) and (not early_stopping):
        model = MLPClassifier(hidden_layer_sizes=[nbr_neurones], random_state=7, max_iter=nbr_iterations)
        model.fit(x_train, y_train)

        y_predit_train = model.predict(x_train)
        y_predit_validation = model.predict(x_validation)
        y_predit_test = model.predict(x_test)

        print("Taux d'erreur en test : " + str((100 - (metrics.accuracy_score(y_test, y_predit_test) * 100))) + "% \n")

        if i == 0:
            tableau_erreurs_train = np.array(100 - metrics.accuracy_score(y_train, y_predit_train) * 100)
            tableau_erreurs_validation = np.array(100 - metrics.accuracy_score(y_validation, y_predit_validation) * 100)
            tableau_erreurs_test = np.array(100 - metrics.accuracy_score(y_test, y_predit_test) * 100)
        else:
            tableau_erreurs_train = np.append(tableau_erreurs_train, 100 - metrics.accuracy_score(y_train, y_predit_train) * 100)
            tableau_erreurs_validation = np.append(tableau_erreurs_validation, 100 - metrics.accuracy_score(y_validation, y_predit_validation) * 100)
            tableau_erreurs_test = np.append(tableau_erreurs_test, 100 - metrics.accuracy_score(y_test, y_predit_test) * 100)



        if i > 10:  # on laisse s'amorcer
            early_stopping = early_stop(tableau_erreurs_validation)

        i = i + 1
        nbr_iterations = nbr_iterations + 40

    plt.plot(tableau_erreurs_train, label="Train")
    plt.plot(tableau_erreurs_validation, label="Validation")
    plt.plot(tableau_erreurs_test, label="Test")
    plt.legend()
    plt.title("Erreurs avec " + str(nbr_neurones) + " neurones et " + str(nbr_iterations) + " itérations.")
    plt.show()

    y_predit_evalutation = model.predict(x_evaluation)
    return y_predit_evalutation

if __name__ == '__main__':
    print("La prédiction en sortie est :")
    print(reseauNeuronne())
