import pandas as pd
pd.set_option('display.max_columns', None)


def preprocessing(filename):
    dataset = pd.read_csv(filename, delimiter="|")

    # We choose the different columns with interests and we reindex them (-1)
    selected_vars = [9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 35, 37, 39, 43]
    selected_vars = [k - 1 for k in selected_vars]
    dataset = dataset[dataset.columns[selected_vars]]

    # We take only mutation as sell or sell in achievement because others aren't valuable
    dataset = dataset[
        (dataset['Nature mutation'] == 'Vente') | (dataset['Nature mutation'] == "Vente en l'état futur d'achèvement")]

    # We want the number or lot to be equals to 0 or 1 because each lot is priced equally
    #dataset = dataset[(dataset['Nombre de lots'] == 0) | (dataset['Nombre de lots'] == 1)]

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

    dataset = dataset.drop(columns=['Code postal','Commune','No voie','Type de voie','Voie','Code voie','Code departement'])
    dataset.reset_index(drop=True, inplace=True)
    dataset.to_csv("DATASET-Preprocessed.csv")
    return dataset


def add_coordinates(dataframe):
    # TODO
    df = dataframe.copy()
    return df
