import numpy as np
import pandas as pd
from  skimage.io import imread, imshow
from knn import KNN
import pywt
import pywt.data
import os
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


def Get_Feacture(picture, cortes):
  LL = picture
  for i in range(cortes):
     LL, (LH, HL, HH) = pywt.dwt2(LL, 'haar')
  return LL.flatten().tolist()

def cargar_dataset():

    carpeta_imagenes = './imagenes_1/'

    # Se ordena los archivos segun el nombre asignado
    archivos = os.listdir(carpeta_imagenes)

    clases = []
    vectores_caracteristicos = []


    for archivo in archivos:
        # Verifica si el archivo es una imagen (puedes agregar más extensiones si es necesario)
        if archivo.endswith(('.png')):

            ruta_completa = os.path.join(carpeta_imagenes, archivo)
            imagen = imread(ruta_completa)

            # Se añade la clase correspondiente al vector clases 
            if int(archivo[1]) < 1:
                clases.append(int(archivo[2]))
            else:
                clases.append(int(archivo[1:3]))

            # Se añade el vector caracteristico de cada imagen
            vectores_caracteristicos.append(Get_Feacture(imagen, 2))

    return vectores_caracteristicos,clases

def acuary(y_prueba,y_correct):
    correctos= np.sum(y_prueba == y_correct)
    return (correctos/len(y_correct))
  

# Definimos el dataset

x,y= cargar_dataset()

x=np.array(x)
y=np.array(y)

# Mezclamos los datos aleatoriamente
indices = np.arange(len(x))
np.random.shuffle(indices)
X = x[indices]
Y = y[indices]


# Definimos los tamaños de los conjuntos (70% entrenamiento, 15% validación, 15% prueba)
total_samples = len(X)
train = int(0.7 * total_samples)
validation = int(0.15 * total_samples)

# Dividimoss los datos en conjuntos de entrenamiento, validación y prueba
X_train = X[:train]
y_train = Y[:train]

X_val = X[train:train + validation]
y_val = Y[train:train + validation]

X_test = X[train + validation:]
y_test = Y[train + validation:]


#Entrenamos el modelo
clasificador = KNN(k=5)
clasificador.aprendizaje(X_train.T,y_train)

# Evalúa el modelo en el conjunto de validacion
y_val_pred=clasificador.clasificacion(X_val.T)
aciertos_val=acuary(y_val_pred,y_val)

print(f"Precisión en el conjunto de validación: {aciertos_val:.2f}")


# Evalúa el modelo en el conjunto de prueba
y_test_pred=clasificador.clasificacion(X_test.T)
aciertos_prue=acuary(y_test_pred,y_test)

print(f"Precisión en el conjunto de prueba: {aciertos_prue:.2f}")





