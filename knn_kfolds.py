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
            # Construye la ruta completa del archivo
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
    return (correctos/len(y_correct))*100      

  

# Definimos el dataset

x,y= cargar_dataset()

x=np.array(x)
y=np.array(y)

# Definimos el número de folds
k = 5

# Mezclamos los datos aleatoriamente
indices = np.arange(len(x))
np.random.shuffle(indices)
X = x[indices]
Y = y[indices]


# Divide los datos en k grupos
X_folds = np.array_split(X, k)
Y_folds = np.array_split(Y, k)



# Lista para almacenar los porcentajes de rendimiento
accuracies=[]

# Realizamos k iteraciones
for i in range(k):
    # Selecciona el conjunto de prueba y entrenamiento actual
    X_test = X_folds[i]
    Y_test = Y_folds[i]
    
    X_train = np.concatenate([X_folds[j] for j in range(k) if j != i])
    Y_train = np.concatenate([Y_folds[j] for j in range(k) if j != i])
    
    #Creamos y entrenamos el modelo.

    # Iniciar KNN
    clasificador = KNN(k=5)
    clasificador.aprendizaje(X_train.T,Y_train)
    
    # Evalúa el modelo en el conjunto de prueba
    y_pred=clasificador.clasificacion(X_test.T)
    porcentaje_aciertos=acuary(y_pred,Y_test)    
    
    # Almacena la precisión en la lista de accuracies
    accuracies.append(porcentaje_aciertos)


# Calcula la precisión promedio de precion en cada K-fold
average_accuracy = np.mean(accuracies)
print(f'Precisión promedio: {average_accuracy:.2f}%')


classification_rep = classification_report(Y_test, y_pred)
print("Informe de Clasificación:")
print(classification_rep)