predictions_65_patients = [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1,
                           1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                           1, 0, 0]

nombre_0 = 0
nombre_1 = 0
for i in range(0, 65):
    if abs(predictions_65_patients[i]) < 0.0000001:
        nombre_0 = nombre_0 + 1

    if abs(predictions_65_patients[i] - 1) < 0.0000001:
        nombre_1 = nombre_1 + 1

print("Nombre de 0 :")
print(nombre_0)

print("Nombre de 1 :")
print(nombre_1)

if (nombre_0 + nombre_1 != 65):
    print("!!!!!!!!!!!!!!!!!")
    print("Attention, la somme du nombre de 0 et du nombre de 1 ne fait pas 65 ! mais", nombre_0 + nombre_1)
