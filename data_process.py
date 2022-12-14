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


def add_coordinates(filename):
    dataset_geocoded = pd.read_csv(filename, delimiter=",")

    # We choose the different columns with interests and we reindex them (-1)
    selected_vars = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14]
    selected_vars = [k - 1 for k in selected_vars]
    dataset_geocoded = dataset_geocoded[dataset_geocoded.columns[selected_vars]]

    dataset_geocoded.reset_index(drop=True, inplace=True)
    dataset_geocoded.to_csv("DATASET-Preprocessed&Geocoded&Filtered.csv")
    return dataset_geocoded


def add_neighborhood(dataframe, distances, indices):
    for index in range(len(indices)):
        sommesVF = np.empty(0)
        sommeVFNeighbors = 0
        for neighborIndex in range(len(indices[index])):
            sommeVFNeighbors += float(dataframe['Valeur fonciere'][index])
        sommesVF = np.append(sommesVF, sommeVFNeighbors)
        
        
def determine_neighborhood(dataframe):
    df = dataframe.copy()
    tree = BallTree(df[['latitude', 'longitude']].values, leaf_size=2, metric='haversine')

    distances, indices = tree.query(df[['latitude', 'longitude']].values, k=10, return_distance=True)

    add_neighborhood(df, distances, indices)
