import numpy as np
import pandas as pd


class TreeNode:
    def __init__(self, X, Y, index):
        self.X = X  # Features
        self.Y = Y  # Label
        self.index = index  # Índice
        self.label = None  # Etiqueta si es terminal
        self.feature_index = None  # Índice de la división
        self.threshold = None  # Umbral de la división
        self.left = None  # Nodo hijo izquierdo
        self.right = None  # Nodo hijo derecho

    def IsTerminal(self): # Función para saber si un nodo es terminal
        return len(np.unique(self.Y)) == 1 # Si solo hay una etiqueta retorna true

    def entropy(self, Y): # Función para medir la incertidumbre en la clasificación
        _, counts = np.unique(Y, return_counts=True)
        probabilities = counts / len(Y) # Obtenemos las probabilidades de los valores únicos sobre el total de etiquetas
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10)) # Aplicamos la ecuación
        return entropy

    def information_gain(self, left_group, right_group): # Función para calcular la ganancia de información
        parent_entropy = self.entropy(self.Y) # Obtenemos la entropía del padre
        left_entropy = self.entropy(self.Y[left_group])# Obtenemos la entropía del hijo izquierdo
        right_entropy = self.entropy(self.Y[right_group])# Obtenemos la entropía del hijo derecho
        total_labels = len(self.Y)
        # Aplicamos la fórmula
        info_gain = parent_entropy - (
            (len(self.Y[left_group]) / total_labels) * left_entropy +
            (len(self.Y[right_group]) / total_labels) * right_entropy
        )

        return info_gain

    def BestSplit(self):
      # Inicializamos los valores a retornar
      best_feature = None
      best_threshold = None
      best_information_gain = -10000 # Ganancia de información negativa al inicio

      for index in range(self.X.shape[1]):
          
          sorted_values = np.sort(np.unique(self.X[:, index])) # Ordenamos los valores únicos de la feature actual

          for j in range(1, len(sorted_values)):
              threshold = (sorted_values[j - 1] + sorted_values[j]) / 2 # Obtenemos el umbral
              left_group = self.X[:, index] <= threshold # Creamos el grupo izquierdo con los valores menores e iguales al umbral
              right_group = ~left_group # Creamos el grupo derecho con los valores restantes
              information_gain = self.information_gain(left_group, right_group) # Obtenemos la ganancia de información con estos grupos
              
              if information_gain > best_information_gain: # Verificamos si esta ganancia es mejor que la actual
                  # De ser así actualizamos todos los valores
                  best_information_gain = information_gain
                  best_feature = index
                  best_threshold = threshold

      return best_feature, best_threshold

    def create_subtree(self):
        if self.IsTerminal(): # Si es un nodo terminal
            self.label = np.unique(self.Y)[0] # Asignamos la etiqueta correspondiente
        else: # Si no es un nodo terminal
            feature, threshold = self.BestSplit() # Encontramos el mejor split
            self.feature_index = feature
            self.threshold = threshold

            # Separamos la data conforma al umbral respectivamente
            X_left = self.X[self.X[:, feature] <= threshold]
            Y_left = self.Y[self.X[:, feature] <= threshold]
            X_right = self.X[self.X[:, feature] > threshold]
            Y_right = self.Y[self.X[:, feature] > threshold]

            # Lista con valores de la feature menores o iguales al umbral
            left_values = [self.index[i] for i in range(len(self.index)) if self.X[i, feature] <= threshold]
            # Lista con valores de la feature mayores al umbral
            right_values = [self.index[i] for i in range(len(self.index)) if self.X[i, feature] > threshold]

            # Si hay elementos se crea un nodo hijo izquierdo y recursivamente crea su subárbol
            if len(X_left) > 0:
                self.left = TreeNode(X_left, Y_left, left_values)
                self.left.create_subtree()
            
           # Si hay elementos se crea un nodo hijo derecho y recursivamente crea su subárbol
            if len(X_right) > 0:
                self.right = TreeNode(X_right, Y_right, right_values)
                self.right.create_subtree()

class DT:
    def __init__(self, X, Y, index): # Inicializamos el árbol
        self.X = X
        self.Y = Y
        self.index = index
        self.m_Root = None
        self.create_DT(self.X, self.Y, [index] * len(Y))

    def create_DT(self, X, Y, index): # Creamos el árbol
        self.m_Root = TreeNode(X, Y, index) # Creamos el nodo inicial
        self.m_Root.create_subtree() # Comenzamos la recursividad

    def fit(self, X): #Función para realizar las predicciones
        predictions = []
        for feature in X: # Iteramos por cada feature
            label = self.predict(feature, self.m_Root)
            predictions.append(label) # Agregamos la predicción
        return predictions

    def predict(self, sample, node): # Función para predecir de manera recursiva
        feature_index, threshold = node.feature_index, node.threshold
        if threshold: # Si tiene umbral
            if sample[feature_index] <= threshold:
                return self.predict(sample, node.left)
            else:
                return self.predict(sample, node.right)
        return node.label # Si no tiene umbral retornamos su etiqueta