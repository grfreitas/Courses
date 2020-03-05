import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv',header=None)
df.columns = ['x1', 'x2', 'y']

def stepFunction(t):
    if t >= 0:
        return 1
    return 0

def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])

def perceptronStep(X, y, W, b, learn_rate = 0.01):
    for i in range(len(X)):
        y_hat = prediction(X[i],W,b)
        if y[i]-y_hat == 1:
            W[0] += X[i][0]*learn_rate
            W[1] += X[i][1]*learn_rate
            b += learn_rate
        elif y[i]-y_hat == -1:
            W[0] -= X[i][0]*learn_rate
            W[1] -= X[i][1]*learn_rate
            b -= learn_rate
    return W, b

def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max

    boundary_lines = []
    for i in range(num_epochs):
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines

X = np.array(df[['x1', 'x2']])
y = np.array(df['y'])

learn_rate = .01
num_epochs = 50
np.random.seed(42)
boundary_lines = trainPerceptronAlgorithm(X, y, learn_rate=learn_rate, num_epochs=num_epochs)

# ---------------------------------------------------------

plt.figure(figsize=(10,10))
plt.xlim(-.5,1.5)
plt.ylim(-.5,1.5)

x = np.linspace(0,1,100)
colors = ['green']*(len(boundary_lines)-1)+['black']
styles = ['dashed']*(len(boundary_lines)-1)+[None]
alphas = np.linspace(0,1,len(boundary_lines))

plt.pause(5)

plt.scatter(df['x1'], df['x2'], c=np.where(np.array(df['y']==1),'red','blue'))

for i in range(len(boundary_lines)):
    a, b = boundary_lines[i]
    y = a*x + b
    plt.title('Num Epochs: {} | Epoch: {} | Learning Rate: {}'.format(num_epochs, i+1, learn_rate))
    plt.plot(x, y, c=colors[i], alpha=alphas[i], linestyle=styles[i])
    plt.pause(.3)

plt.pause(5)