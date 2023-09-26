import numpy as np

def Hiperplano(x,w):
  # write your code here
  return np.dot(x,w.T)

def S(x,w):
  return 1 / (1 + np.exp(-Hiperplano(x,w)))

def Loss_function(x,y,w):
  # write your code here
  f1=y*np.log(S(x,w))
  f2=(1-y)*np.log(1-S(x,w))
  f3= np.sum(f1+f2)
  return -1/len(y)*(f3)

def Derivatives(x,y,w):
  # write your code here
  return np.matmul((y - S(x,w)),-x)/len(y)

def change_parameters(w, derivatives, alpha):
  # write your code here
  return w - alpha * derivatives  


class Regresion:
    def __init__(self, epocas,alpha):
        self.epocas=epocas
        self.alpha=alpha
        self.numero_clases=None
        self.w=None

    def train(self,x,y):
        num_samples, num_caracteristicas = x.shape
        self.numero_clases = len(np.unique(y))  
        self.w = np.zeros((self.numero_clases, num_caracteristicas))

        for i in range(self.epocas):
           for c in range(self.numero_clases):
              y_class = (y == c).astype(int)

              s=S(x,self.w[c])
              loss=s-y_class
              gradient = np.dot(x.T, loss) / num_samples
              self.w[c] -= self.alpha * gradient

    def predict(self,x):
       num_samples= x.shape[0]
       y_pred = np.zeros(num_samples, dtype=int)

       for i in range(num_samples):
            class_probs = [ S(x[i], self.w[c]) for c in range(self.numero_clases)]
            y_pred[i] = np.argmax(class_probs)

       return y_pred    


              

