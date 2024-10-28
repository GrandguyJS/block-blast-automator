import numpy as np
import random
import pickle
import math
import copy

def lerp(A, B, t):
    return A+(B-A)*t

def difference(A, B):
    best = [float("inf"), None]
    for i,a in enumerate(A):
        difference = abs(np.mean(a-B))
        if difference < best[0]:
            best[0] = difference
            best[1] = i
    return best

class NeuralNetwork():
    def __init__(self, layers, amount):
        self.layers = layers
        self.amount = amount
        self.generation = 1
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def load(self, path=None):
        if path is not None:
            with open(path, 'rb') as file: 
                parameters = pickle.load(file)
                self.weights = [copy.deepcopy(parameters[0]) for _ in range(self.amount)]
                self.biases = [copy.deepcopy(parameters[1]) for _ in range(self.amount)]
                self.mutate()
                self.mutate()
            return
        self.weights = []
        self.biases = []
        for _ in range(self.amount):
            weights = []
            biases = []
            for i,v in enumerate(self.layers[:-1]):
                weights.append(np.random.randn(self.layers[i+1], v) * 0.1)
                biases.append(np.zeros((self.layers[i+1], 1)))
            self.weights.append(weights)
            self.biases.append(biases)
        return
    def save(self, path):
        with open(path, 'wb') as file: 
            pickle.dump([self.weights[0], self.biases[0]], file)
        return
    def mutate(self):
        weights = self.weights
        biases = self.biases

        mutation_rate = min(1, 1 / (math.log(self.generation, 2) + 1e-15))

        for x in range(1, self.amount): # Leave out first
            for i in range(0, len(weights[x])):
                random_weights = np.random.uniform(-1, 1, weights[x][i].shape)
                weights[x][i] = lerp(weights[x][i], random_weights, mutation_rate)
                
            for i in range(0, len(biases[x])):
                random_biases = np.random.uniform(-1, 1, biases[x][i].shape)
                biases[x][i] = lerp(biases[x][i], random_biases, mutation_rate)
        return

    def forward(self, X):
        weights, biases = self.weights, self.biases
        self.inputValue = X

        cache = [[X.T] for _ in range(self.amount)]

        for i in range(self.amount):
            for j in range(0, len(weights[i])):
                cache[i].append(np.dot(weights[i][j], cache[i][-1]) + biases[i][j])
                cache[i].append(self.sigmoid(cache[i][-1]))
        
        return [x[-1].T for x in cache]
    
    def selection(self, index):
        self.weights = [copy.deepcopy(self.weights[index]) for _ in range(self.amount)]
        self.biases = [copy.deepcopy(self.biases[index]) for _ in range(self.amount)]
        self.mutate()
        self.generation+=1
        return

"""X = [[0,0], [0,1], [1, 0], [1,1]]
Y = [[0], [1], [1], [0]]
    

path = "model/parameters.pkl"

ai = NeuralNetwork([2, 3, 1], 3)
ai.load(path)

res = ai.forward(np.array(X))

for i in range(0, 10000):
    diff = difference(res, Y)
    print("Difference: ", diff[0])
    ai.selection(diff[1])
    res = ai.forward(np.array(X))   

ai.save(path)"""
