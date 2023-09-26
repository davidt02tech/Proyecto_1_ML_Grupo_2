
import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def aprendizaje(self,X,Y):
        self.x=X # matriz de vectores caracteristicos
        self.y=Y # clases asociadas a cada vector x(n)
        self.muestras=len(X[0]) # cantidad de muestras
    
    def clasificacion(self,X_train):
        clases=[]
        for i in range(len(X_train[0])): 
            distancias=np.zeros(self.muestras)
            for j in range(self.muestras): 
                # por cada vector x(n) de caracteristicas
                distancias[j]=dist_euclidiana(self.x[:,j],X_train[:,i])
            
            # Ordenamos las distancias mas cercanas segun la posicion en la cual se encuentren
            k_distancias=np.argsort(distancias)

            # Identificamos las k distancias con respecto a las clases
            k_etiqueta=self.y[k_distancias[:self.k]]

            clases_prioridad = Counter(k_etiqueta).most_common(1)

            # Almacenamos la clase con mayor catidad de votos a la lista de predicciones
            clases.append(clases_prioridad[0][0]) 
        return clases
            

def dist_euclidiana(x,y):
    #Aplicamos distancia euclidiana con respecto a cada punto
    return np.sqrt(np.sum((x-y)**2)) 