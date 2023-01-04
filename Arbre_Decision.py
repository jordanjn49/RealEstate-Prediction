import numpy as np  # pour utiliser des matrices
import matplotlib.pyplot as plt  # pour afficher des courbes
import pandas as pd
from sklearn import metrics


def erreur_commise(valeurs_reelles, valeurs_predites):
    erreur = (1.0 - metrics.accuracy_score(valeurs_reelles, valeurs_predites)) * 100

    return erreur


def arbreDecision():
    np.random.seed(7)  # pour la réproductibilité des simulations

    # lecture du fichier
    donnees_data_frame = pd.read_csv('DATASET-Final.csv', delimiter=",")

    selected_vars = [3, 5, 6, 7, 8, 9, 10, 12, 13, 14]
    selected_vars = [k - 1 for k in selected_vars]
    donnees_data_frame = donnees_data_frame[donnees_data_frame.columns[selected_vars]]

    print(donnees_data_frame)

    donnees_ensemble_total = donnees_data_frame.values

    # mélange des lignes du tableau
    np.random.shuffle(donnees_ensemble_total)

    nombre_lignes_base = donnees_ensemble_total.shape[0]
    nombre_colonnes_base = donnees_ensemble_total.shape[1]

    x_train = donnees_ensemble_total[0:round(nombre_lignes_base * 2 / 3 * 2 / 3), :donnees_ensemble_total.shape[1] - 1]
    y_train = donnees_ensemble_total[0:round(nombre_lignes_base * 2 / 3 * 2 / 3), donnees_ensemble_total.shape[1] - 1:]

    x_validation = donnees_ensemble_total[
                   round(nombre_lignes_base * 2 / 3 * 2 / 3) + 1:round(nombre_lignes_base * 2 / 3),
                   :donnees_ensemble_total.shape[1] - 1]
    y_validation = donnees_ensemble_total[
                   round(nombre_lignes_base * 2 / 3 * 2 / 3) + 1:round(nombre_lignes_base * 2 / 3),
                   donnees_ensemble_total.shape[1] - 1:]

    x_test = donnees_ensemble_total[round(nombre_lignes_base * 2 / 3) + 1:, :donnees_ensemble_total.shape[1] - 1]
    y_test = donnees_ensemble_total[round(nombre_lignes_base * 2 / 3) + 1:, donnees_ensemble_total.shape[1] - 1:]

    y_predit_test = np.shape(0)
    # arbre de décision - importation de la classe
    from sklearn.tree import DecisionTreeClassifier

    erreur_validation_la_plus_basse = 100
    early_stopping = False;

    max_depth_courante = 1

    tableau_erreurs_train = np.empty(0)
    tableau_erreurs_validation = np.empty(0)
    tableau_erreurs_test = np.empty(0)

    while (max_depth_courante < 101) and (not early_stopping):
        mon_arbre = DecisionTreeClassifier(max_depth=max_depth_courante)

        mon_arbre.fit(x_train, y_train)

        y_predit_train = mon_arbre.predict(x_train)
        y_predit_validation = mon_arbre.predict(x_validation)
        y_predit_test = mon_arbre.predict(x_test)

        if max_depth_courante == 1:  # premier point à ajouter :
            tableau_erreurs_train = erreur_commise(y_train, y_predit_train)
            tableau_erreurs_validation = erreur_commise(y_validation, y_predit_validation)
            tableau_erreurs_test = erreur_commise(y_test, y_predit_test)
        else:
            tableau_erreurs_train = np.append(tableau_erreurs_train, erreur_commise(y_train, y_predit_train))
            tableau_erreurs_validation = np.append(tableau_erreurs_validation,
                                                   erreur_commise(y_validation, y_predit_validation))
            tableau_erreurs_test = np.append(tableau_erreurs_test, erreur_commise(y_test, y_predit_test))

        max_depth_courante = max_depth_courante + 1

        # gestion de l'early-stopping :
        erreur_validation_courante = erreur_commise(y_validation, y_predit_validation)
        print("erreur_validation_courante =" + str(erreur_validation_courante))
        print("erreur_validation_la_plus_basse =" + str(erreur_validation_la_plus_basse))
        print("erreur_validation_la_plus_basse *1.05 =" + str(1.05 * erreur_validation_la_plus_basse))

        if (erreur_validation_courante >= 1.05 * erreur_validation_la_plus_basse) and (
                max_depth_courante > 100):  # i>4 pour laisser le réseau s'élancer
            early_stopping = True
            print("Stop ça remonte !!!")

        if erreur_validation_courante < erreur_validation_la_plus_basse:
            erreur_validation_la_plus_basse = erreur_validation_courante

    plt.plot(tableau_erreurs_train, label="train")
    plt.plot(tableau_erreurs_validation, label="validation")
    plt.plot(tableau_erreurs_test, label="test")
    plt.legend()
    titre = "Restent : " + str(100 - max_depth_courante) + " coups à appuyer. \"s\" pour passer à la suite..."
    titre = titre + "Avec hauteurs de 1 à 100"
    plt.show()


if __name__ == '__main__':
    arbreDecision()
