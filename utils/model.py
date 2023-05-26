import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as plt

class Perceptron:
    def __init__(self,eta,epochs): #eta= learning rate
        self.weights = np.random.randn(3)*1e-4
        self.eta = eta
        self.epochs = epochs
    def activationFunction(self,inputs,weights):
        z= np.dot(inputs, weights)
        return np.where(z>0,1,0)
    
    def fit(self,X,y):
        
        self.X = X #we use self.X so that it can be used anywhere in the code
        self.y = y
        X_with_bias = np.c_[self.X, -np.ones((len(self.X),1))]
        print(X_with_bias)
        
        for epoch in range(self.epochs):
            j=0
            print("--"*10)
            print(f"for epoch: {epoch}")
            print("--"*10)
            y_hat = self.activationFunction(X_with_bias, self.weights) #forward prop
            print(f"predicted value: \n {y_hat}")
            self.error = self.y-y_hat
            print(f"error {self.error}")
                
            self.weights = self.weights + self.eta*np.dot(X_with_bias.T, self.error)#backward prop
            print(f"updated weights {self.weights}")
        
    def predict(self,X):
        X_with_bias = np.c_[X, -np.ones((len(X),1))]
        return self.activationFunction(X_with_bias, self.weights)
        
    def total_loss(self):
        total_loss = np.sum(self.error)
        print(f"total loss:{total_loss}")
        return total_loss
    