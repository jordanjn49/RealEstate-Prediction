import numpy as np  # pour utiliser des matrices
import matplotlib.pyplot as plt  # pour afficher des courbes
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import set_config
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error


def RandomForestRegressor():
    np.random.seed(7)  # pour la réproductibilité des simulations

    # lecture du fichier
    donnees_data_frame = pd.read_csv('DATASET-Final.csv', delimiter=",")

    selected_vars = [6, 7, 8, 9, 10, 11, 13, 14, 15, 16]
    selected_vars = [k - 1 for k in selected_vars]
    donnees_data_frame = donnees_data_frame[donnees_data_frame.columns[selected_vars]]
    print(donnees_data_frame)

    donnees_ensemble_total = donnees_data_frame.values

    # mélange des lignes du tableau
    np.random.shuffle(donnees_ensemble_total)

    nombre_lignes_base = donnees_ensemble_total.shape[0]

    x_train = donnees_ensemble_total[0:round(nombre_lignes_base * 2 / 3), :donnees_ensemble_total.shape[1] - 1]
    y_train = donnees_ensemble_total[0:round(nombre_lignes_base * 2 / 3), donnees_ensemble_total.shape[1] - 1:]

    x_test = donnees_ensemble_total[round(nombre_lignes_base * 2 / 3) + 1:, :donnees_ensemble_total.shape[1] - 1]
    y_test = donnees_ensemble_total[round(nombre_lignes_base * 2 / 3) + 1:, donnees_ensemble_total.shape[1] - 1:]

    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # arbre de décision - importation de la classe
    from sklearn.ensemble import RandomForestRegressor

    set_config(print_changed_only=False)

    rfr = RandomForestRegressor()

    RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                          max_depth=None, max_features='auto', max_leaf_nodes=None,
                          max_samples=None, min_impurity_decrease=0.0,
                          min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0,
                          n_estimators=40, n_jobs=None, oob_score=False,
                          random_state=None, verbose=0, warm_start=False)

    rfr.fit(x_train, y_train)

    score = rfr.score(x_train, y_train)
    print("R-squared:", score)

    y_validation = rfr.predict(x_test)

    print('MAPE: ', mean_absolute_percentage_error(y_test, y_validation))
    print('MSE: ', mean_squared_error(y_test, y_validation))
    print('RMSE: ', np.sqrt(mean_squared_error(y_test, y_validation)))

    x_ax = range(len(y_test))
    plt.plot(x_ax, y_test, linewidth=1, label="Test")
    plt.plot(x_ax, y_validation, linewidth=1.1, label="Prédiction")
    plt.title("Comparaison entre les données de test et celles prédites")
    plt.xlabel('Nombre de ligne testées')
    plt.ylabel('Prix du m²')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    RandomForestRegressor()
