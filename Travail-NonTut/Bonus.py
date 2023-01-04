from Reseau_Neurones import reseauNeuronne
from RandomForestRegressor import RandomForestRegressor
from SVM import SVM
import numpy as np  # pour utiliser des matrices

reseauNeurone = reseauNeuronne()
arbreDecision = RandomForestRegressor()
svm = SVM()

# Tableau totalisant les résultats des 3 algos
total = reseauNeurone + arbreDecision + svm
resultat = np.empty(0)

for i in range(len(total)):
    if (total[i] > 1):
        resultat = np.append(resultat, 1)
    else: resultat = np.append(resultat, 0)


print("Le résultat obtenu par vote majoritaire des 3 algos est : ")
print(resultat)