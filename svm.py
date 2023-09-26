import numpy as np

def loss(y, x, w, bias, c):
  hinge_loss = c * np.sum(np.maximum(0, 1 - y * (np.dot(x, w) + bias)))
  regularization_term = np.dot(w, w)/2
  return hinge_loss + regularization_term

def grad(y, x, w, bias, c):
  grad=[]
  # write your code here
  if(y*(np.dot(x,w)+bias)>=1):
    grad.append(w)
    grad.append(0)
  else:
    grad.append(w - c*x*y)
    grad.append(-c*y)
  return grad

def update(w, b, grad, alpha):
    w -= alpha * grad[0]
    b -= alpha * grad[1]
    return w, b

class SVM:
    def __init__(self,C,epocas,alpha):
       self.C=C
       self.epocas=epocas
       self.alpha=alpha

    def fit(self,x,y):
        self.clases = np.unique(y)
        self.clasicados = {}

        for class_label in self.clases:
            y_binario = np.where(y == class_label, 1, -1)
            clasificacion = self.train(x, y_binario)
            self.clasicados[class_label] = clasificacion

    def train(self,x,y):
        numero_ejemplos, num_caracteristicas = x.shape
        w = np.zeros(num_caracteristicas)
        bias = 0

        for e in range(self.epocas):
            for i in range(numero_ejemplos):
                condition = y[i] * (np.dot(x[i], w) - bias) >= 1
                if condition:
                    w -= self.alpha * (2 * self.C * w)
                else:
                    w -= self.alpha * (2 * self.C * w - np.dot(x[i], y[i]))
                    bias -= self.alpha * y[i]

        return (w, bias)
               
    def prediccion(self, x):
        predicciones = []

        for i in range(x.shape[0]):
            scores = {}
            for clase, classifier in self.clasicados.items():
                weights, bias = classifier
                score = np.dot(x[i], weights) - bias
                scores[clase] = score

            predicted_class = max(scores, key=scores.get)
            predicciones.append(predicted_class)

        return np.array(predicciones)   