import math

import numpy as np
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

pd.set_option('display.max_columns', None)


def preprocessing(filename):
    dataset = pd.read_csv(filename, delimiter="|")

    # We choose the different columns with interests and we reindex them (-1)
    selected_vars = [9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 35, 37, 39, 40, 43]
    selected_vars = [k - 1 for k in selected_vars]
    dataset = dataset[dataset.columns[selected_vars]]

    # We take only mutation as sell or sell in achievement because others aren't valuable
    dataset = dataset[
        (dataset['Nature mutation'] == 'Vente') | (dataset['Nature mutation'] == "Vente en l'état futur d'achèvement")]

    # We want the number or lot to be equals to 0 or 1 because each lot is priced equally
    # dataset = dataset[(dataset['Nombre de lots'] == 0) | (dataset['Nombre de lots'] == 1)]

    # We eliminate surface that are null or NaN
    dataset = dataset[dataset['Surface reelle bati'] != 0]
    dataset = dataset[dataset['Surface reelle bati'].notna()]
    dataset = dataset[dataset['Surface terrain'].notna()]

    # We eliminate land values that are NaN
    dataset = dataset[dataset['Valeur fonciere'].notna()]

    # We select only rows in Maine-et-Loire (49)
    dataset = dataset[(dataset['Code departement'] == 49)]

    # We replace all value to work with GeoEncoding, and we merge into a new column "Address"
    dataset['Code postal'] = dataset['Code postal'].fillna('0').astype(int)
    dataset['Commune'] = dataset['Commune'].fillna(' ')
    dataset['No voie'] = dataset['No voie'].fillna('0').astype(int)
    dataset['Type de voie'] = dataset['Type de voie'].fillna(' ')
    dataset['Voie'] = dataset['Voie'].fillna(' ')
    dataset['Adresse'] = dataset['No voie'].astype(str) + ' ' + dataset['Type de voie'] + ' ' + dataset['Voie'] + ' ' + \
                         dataset['Commune'] + ' ' + dataset['Code postal'].astype(str) + ' ' + 'France'

    # On parse la valeur foncière (string) en float pour pouvoir en manipuler la valeur
    floatValeursFoncieres = np.empty(0)
    for valeurFonciere in dataset['Valeur fonciere']:
        valeurFonciere = float(valeurFonciere.replace(',', '.'))
        floatValeursFoncieres = np.append(floatValeursFoncieres, valeurFonciere)

    dataset['Valeur fonciere'] = floatValeursFoncieres

    dataset = dataset.drop(
        columns=['Code postal', 'Commune', 'No voie', 'Type de voie', 'Voie', 'Code voie', 'Code departement'])
    dataset.reset_index(drop=True, inplace=True)
    dataset.to_csv("DATASET-Preprocessed.csv")
    return dataset


def feature_engineering(filename):
    dataset_geocoded = pd.read_csv(filename, delimiter=",")

    #######################################################
    ###########        Ajout des distances      ###########
    #######################################################

    tree = BallTree(dataset_geocoded[['latitude', 'longitude']].values, leaf_size=2, metric='haversine')
    nbVoisins = 10
    distances, indices = tree.query(dataset_geocoded[['latitude', 'longitude']].values, k=nbVoisins,
                                    return_distance=True)

    listMoyennesVf = np.empty(0)
    for index in range(len(indices)):
        listSommesVF = np.empty(0)
        sommeVFNeighbors = 0
        for neighborIndex in range(nbVoisins):
            sommeVFNeighbors += dataset_geocoded['Valeur fonciere'][index]
            listSommesVF = np.append(listSommesVF, sommeVFNeighbors)
        moyenneVF = listSommesVF.sum() / nbVoisins
        listMoyennesVf = np.append(listMoyennesVf, moyenneVF)

    listDistancesMoyennes = np.empty(0)
    for index in range(len(distances)):
        distanceMoyenne = distances[index].sum() / nbVoisins * 6371  # We want to have it in kilometers
        listDistancesMoyennes = np.append(listDistancesMoyennes, distanceMoyenne)

    dataset_geocoded['moy_vf_' + str(nbVoisins) + '_plus_proches'] = listMoyennesVf
    dataset_geocoded['moy_dist_' + str(nbVoisins) + '_plus_proches'] = listDistancesMoyennes

    #######################################################
    ###########        Ajout des densités       ###########
    #######################################################

    density = pd.read_csv('DATASET-DEN2018.csv', delimiter=",")

    dataset_geocoded['densite_pop'] = ''
    for i in range(len(dataset_geocoded)):
        for j in range(len(density)):
            if dataset_geocoded['result_postcode'][i] == density['ZIPCode'][j]:
                dataset_geocoded['densite_pop'][i] = density['Densite'][j]
                i += 1
        dataset_geocoded['densite_pop'][i] = 0

    #######################################################
    ###########         Ajout du €/m2           ###########
    #######################################################

    dataset_geocoded['prix_m2'] = ''
    for i in range(len(dataset_geocoded)):
        price = dataset_geocoded['Valeur fonciere'][i] / dataset_geocoded['Surface terrain'][i]
        if price <= 5000:
            dataset_geocoded['prix_m2'][i] = price
            i += 1
        dataset_geocoded['prix_m2'][i] = 0

    ##################################################
    #######          Tri    ##########
    ##################################################

    # creating a dict file
    type_local = {'Maison': 1, 'Appartement': 2, 'Local industriel. commercial ou assimilé': 3}
    dataset_geocoded['Type local'] = [type_local[item] for item in dataset_geocoded['Type local']]

    dataset_geocoded = dataset_geocoded[dataset_geocoded['latitude'].notna()]
    dataset_geocoded = dataset_geocoded[dataset_geocoded['longitude'].notna()]
    dataset_geocoded = dataset_geocoded[dataset_geocoded['densite_pop'] != 0]
    dataset_geocoded = dataset_geocoded[dataset_geocoded['prix_m2'] != 0]
    dataset_geocoded = dataset_geocoded[dataset_geocoded['moy_dist_' + str(nbVoisins) + '_plus_proches'] != 0]

    selected_vars = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 28, 29, 30, 31]
    selected_vars = [k - 1 for k in selected_vars]
    dataset_geocoded = dataset_geocoded[dataset_geocoded.columns[selected_vars]]

    dataset_geocoded.reset_index(drop=True, inplace=True)
    dataset_geocoded.to_csv("DATASET-Final.csv")
