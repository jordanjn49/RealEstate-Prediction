########################################################################################################################
# The purpose of this file is to plot the evolution of the value per square meter within a year to see if there is a
# big evolution
########################################################################################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset = pd.read_csv('DATASET-Final.csv', delimiter=",")

valeursFoncieres = dataset['Valeur fonciere']
surfacesTerrains = dataset['Surface terrain']
dates = dataset['Date mutation']

for i in range(len(dates)):
    date = dates[i].split('/')[1]
    dates[i] = date

prixMetreCarre = valeursFoncieres/surfacesTerrains

prixMetreCarreCumules = np.empty(6)
nbVentesParMois = np.empty(6)

for i in range(len(dates)):
    prixMetreCarreCumules[int(dates[i])-1] += prixMetreCarre[i]
    nbVentesParMois[int(dates[i])-1] += 1


# prixMedian = np.empty(6)
#
# for i in range(1,7):
#     prixParMois = np.empty(0)
#     for j in range(len(dates)):
#         if int(dates[j]) == i:
#             prixParMois = np.append(prixParMois, prixMetreCarre[j])
#     prixMedian = np.append(prixMedian, prixParMois)


prixMoyenMetreCarreParMois = prixMetreCarreCumules / nbVentesParMois

mois = ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin']

# fig, (moyen, median) = plt.subplots(1,2)

plt.plot(mois, prixMoyenMetreCarreParMois)
plt.title('Prix moyen du mètre carré par mois')

# median.plot(mois, prixMedianMetreCarreParMois)
# median.title('Prix médian du mètre carré par mois')

plt.show()