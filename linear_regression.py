import numpy as np
import matplotlib.pyplot as plt

def getMeanSquaredError(data, m, b):
    sum = 0
    n = float(len(data))
    for row in data:
        x = row[0]  # hours studied
        y = row[1]  # test score
        sum += (y - (m*x + b))**2
    mse = sum/n
    return mse
        
def stepGradient(data, learningRate, m, b):
    mGradient = 0
    bGradient = 0
    n = float(len(data))
    for row in data:
        x = row[0] # hours studied
        y = row[1] # test scores
        mGradient += -(2/n) * x * (y-(m*x+b))
        bGradient += -(2/n) * (y - (m*x+b))
    m -= mGradient * learningRate
    b -= bGradient * learningRate
    return [m, b]

def runGradientDescent(data, iterations, learningRate):
    m = 0
    b = 0
    for i in range(0, iterations):
        [m, b] = stepGradient(data, learningRate, m, b)
        mse = getMeanSquaredError(data, m, b)
        # print mse
    return [m, b]

def predict(x, m, b):
    return m*x + b

def showGraph(data, m, b):
    x = data[:,0]
    y = data[:,1]
    plt.scatter(x, y)
    plt.plot(x, m*x+b)
    plt.show()
    
def run():
    data = np.genfromtxt('data.csv', delimiter=",")
    iterations = 1000
    learningRate = 0.0001
    [m, b] = runGradientDescent(data, iterations, learningRate)
    showGraph(data, m, b)
    
run()