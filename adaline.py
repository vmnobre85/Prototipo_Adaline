import numpy as np
from activation_function import BinaryStep
from activation_function import SignFunction

class Adaline:
    def __init__(self, input_values, output_values, precision = 1e-6, learning_rate=0.0025, activation_function=SignFunction):
        self.input_values = input_values
        self.output_values = output_values
        self.precision = precision
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.W = np.random.rand(len(input_values[0]))
        self.theta = np.random.rand(1)[0]
        self.epochs = 0
        self.eqms =[]
    
    def eqm(self):
        eqm = 0
        for x, d in zip(self.input_values, self.output_values):
            u = np.dot(np.transpose(x), self.W) - self.theta
            eqm += (d-u)**2
        return eqm/len(self.output_values)
    
        
    def train(self):
        last_eqm = 0
        actual_eqm = 0
        while True:
            self.epochs += 1
            print(f'Épocas {self.epochs}')
            last_eqm = self.eqm()
            print(f'Valores pré ajuste')
            print(f'\t W: {self.W}')
            print(f'\t theta: {self.theta}')
            print('')
            
            for x, d in zip(self.input_values, self.output_values):
                u = np.dot(np.transpose(x), self.W) - self.theta
                self.theta = self.theta + self.learning_rate * (d - u) * -1
                self.W = self.W + self.learning_rate * (d - u) * x
                
            actual_eqm = self.eqm()
            self.eqms.append(actual_eqm)
            print(f'Valor da precisão é {self.precision} valor encontrado {u} ')
            print(f'Valores pós ajuste')
            print(f'\t W: {self.W}')
            print(f'\t theta: {self.theta}')
            print('')
                    
            if abs(actual_eqm - last_eqm) <= self.precision:
                break
        print(f'Fim do treinamento')
        print(f'\t W: {self.W}')
        print(f'\t theta: {self.theta}')
        print('')
        
    def testes(self):
        
        while self.epochs <= 5:
            self.epochs += 1
            print('Epoca {}'.format(self.epochs))
            for x, d in zip(self.input_values, self.output_values):
                u = np.dot(np.transpose(x), self.W) - self.theta
                y = self.activation_function.g(u)

                print('O valor encontrado foi {}'.format(y))
                               
    def evaluate(self, input_values):
        u = np.dot(np.transpose(input_values), self.W) - self.theta
        return self.activation_function.g(u)
    