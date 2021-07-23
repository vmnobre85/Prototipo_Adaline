# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 13:54:30 2021

@author: Victor Nobre
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adaline import Adaline
from activation_function import BinaryStep
from activation_function import SignFunction



datasets = pd.read_csv('dataset/dataset-treinamento.csv')
X = datasets.iloc[:,0:4].values
d = datasets.iloc[:,4:].values


a = Adaline(X, d, activation_function = SignFunction)
a.train()

datasets = pd.read_csv('dataset/dataset-teste.csv')
X = datasets.iloc[:,0:3].values

a = Adaline(X, d)
a.testes()

fig, ax = plt.subplots(2)


for i in range(len(d)):
    if d[i] == 1:
        ax[0].plot(X[i, 0], X[i, 1], 'ro')
    else:
        ax[0].plot(X[i, -2], X[i, 2], 'go')
x_plot = np.arange(-10, 10)
y_plot = list(map(lambda x: (-1 * (a.W[0]/a.W[1])*x)+(a.theta/a.W[1]), x_plot ))
ax[0].plot(x_plot, y_plot)
ax[1].plot(range(len(a.eqms)), a.eqms, 'bo')




