import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

class Regressor:
    def __init__(self, data, x, y, n = 1):
        self.data = data
        self.X = self.data[x]
        self.Y = self.data[y]
        self.size = len(self.Y)
        self.n = n
        
        avg = self.Y.mean()
        # f = weights0 + weights1.x + ... + weightsn.x^n
        self.weights = np.random.uniform(0, 1, self.n + 1)

        self.alpha = 1e-4
        self.mu = 1

    def apply(self, f, x):
        total = 0
        for i in range(len(f)):
            total += f[i] * (x ** i)
        return total

    # compute partial derivative of Loss with respect to ith weight
    def weightiPartialLoss(self, weightIndex):
        total = 0
        
        for i in range(self.size):
            x = self.X[i]
            y = self.Y[i]

            temp = 2 * (x ** weightIndex) * (self.apply(self.weights, x) - y)
            if weightIndex > 0:
                temp += 2 * self.mu * self.weights[weightIndex]

            total += temp

        return total / self.size

    def updateWeights(self):
        new_f = []
        for i in range(self.n + 1):
            new_wi = self.weights[i] - (self.alpha * self.weightiPartialLoss(i))
            new_f.append(new_wi)

        muChange = 0
        for j in range(self.n + 1):
            muChange += self.weights[j] ** 2
        self.mu = self.mu - (self.alpha * muChange)

        self.weights = new_f

        
f = [0, 1, 5, 4]

if __name__ == "__main__":
    data = pd.read_csv("Nat_Gas.csv")
    height, width = data.shape
    data['Dates'] = pd.to_datetime(data['Dates'], dayfirst = False)
    first = data['Dates'][0]
    
    data['Dates'] = data['Dates'].apply(lambda x: (x - first).days)# // 30)
    print (data.head())
    
    plt.plot(data['Dates'], data['Prices'])
    r = Regressor(data, 'Dates', 'Prices', 1)
    
    xnums = np.linspace(0, data['Dates'].iloc[-1])

    for i in range(1, 2001):
        r.updateWeights()
        if i % 400 == 0:
            print (r.weights)
            ynums = np.array(list(map(lambda x: r.apply(r.weights, x), xnums)))
            plt.plot(xnums, ynums, 'g')
            
    ynums2 = np.array(list(map(lambda x: r.apply(r.weights, x), xnums)))
    plt.plot(xnums, ynums2, 'y')
    plt.show()