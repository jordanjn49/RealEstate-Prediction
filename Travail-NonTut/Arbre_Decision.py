import numpy as np  # pour utiliser des matrices
import matplotlib.pyplot as plt  # pour afficher des courbes
import pandas as pd

def arbreDecision():
    np.random.seed(7)  # pour la réproductibilité des simulations

    # lecture du fichier
    donnees_data_frame = pd.read_csv('train.txt', delimiter=" ")
    donnees_data_frame_reel = pd.read_csv('test_EVALUATION.txt', delimiter=" ")

    donnees_ensemble_total = donnees_data_frame.values
    donnees_ensemble_total_reel = donnees_data_frame_reel.values

    # mélange des lignes du tableau
    np.random.shuffle(donnees_ensemble_total)

    nombre_lignes_base = donnees_ensemble_total.shape[0]

    x_train = donnees_ensemble_total[0:round(nombre_lignes_base * 3 / 4 * 3 / 4), :donnees_ensemble_total.shape[1] - 1]
    y_train = donnees_ensemble_total[0:round(nombre_lignes_base * 3 / 4 * 3 / 4), donnees_ensemble_total.shape[1] - 1:]

    x_test = donnees_ensemble_total[round(nombre_lignes_base * 2 / 3) + 1:, :donnees_ensemble_total.shape[1] - 1]
    y_test = donnees_ensemble_total[round(nombre_lignes_base * 2 / 3) + 1:, donnees_ensemble_total.shape[1] - 1:]

    x_evaluation = donnees_ensemble_total_reel[:, :donnees_ensemble_total.shape[1] - 1]

    y_predit_test = np.shape(0)
    #arbre de décision - importation de la classe
    from sklearn.tree import DecisionTreeClassifier

    for i in range(13):
        mon_arbre = DecisionTreeClassifier(max_depth=i+1)

        mon_arbre.fit(x_train,y_train)

        from sklearn import tree

        from matplotlib.pyplot import figure
        figure(figsize=(10,8))

        tree.plot_tree (mon_arbre, filled=True,  impurity=False, proportion=True, rounded=False,
        feature_names=['var1','var2','var3','var4','var5','var6','var7','var8','var9','var10','var11','var12','var13'])
        plt.show()

        y_predit_test = mon_arbre.predict (x_test)
        from sklearn import metrics
        err = (1.0 -metrics.accuracy_score (y_test ,y_predit_test ))*100
        print("Erreur pour "+str(i+1)+" branches = ", round(err, 2), "%")
    y_predit_evaluation = mon_arbre.predict(x_evaluation)
    return y_predit_evaluation

if __name__ == '__main__':
    arbreDecision()