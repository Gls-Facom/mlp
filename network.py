#!/usr/bin/python

import numpy as np
import random
from layer import Layer
from dataHandler import DataHandler

#como inserir a funcao de ativacao?

class Network(object):

    def __init__(self,sizes, dataHandler):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.dataHandler = dataHandler
        # fazer inicializacao de layers

    def feedforward(self, a, keep_z=False):
        self.layer[0].activation = a
        for layer in self.layer[1:]:
            layer.update_layer(self.layer[i-1].activation, keep_z)
        return layer.activation

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases] # nablas terão formato de acordo com seus respectivos layers
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch: # para cada exemplo de treino da mini batch, calcula o ajuste necessário
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb+dnb for nb,dnb in zip(nabla_b, delta_nabla_b)] # dC/db
            nabla_w = [nw+dnw for nw,dnw in zip(nabla_w, delta_nabla_w)] # dC/dw
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)] # atualiza pesos e biases
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        # feedforward, passa pela rede indo em direção a ultima camada, calculando os zs e as ativações
        self.feedforward(x, keep_z=True)
        activation = self.layers[-1].activation

        # output error (calcula a última camada "na mão")
        delta = self.cost_derivative(activations[-1], y) * self.layers[-1].activation_function(z, prime=True) # (BP1)
        nabla_b[-1] = delta # (BP3)
        nabla_w[-1] = np.dot(delta, self.layers[-2].activation.tranpose()) # (BP4)

        # backpropagate the error, l é usado de forma crescente, mas como acessar posições
        # negativas significa acessar de trás pra frente, o erro é propagado do fim ao começo da rede
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            afp = self.layers[-l].activation_function(z, prime=True)
            delta = np.dot(self.layers[-l+1].weights.transpose(), delta) * afp # (BP2)
            nabla_b[-l] = delta # (BP3)
            nabla_w[-l] = np.dot(delta, self.layers[-l-1].activation.transpose()) # (BP4)
        return (nabla_b, nabla_w)


    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

    def evaluate(self, test_data):
        # guarda resultados passando o conjunto de teste pela rede
        # e assume o maior resultado como resposta da rede
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x,y) in test_data]
        n = len(test_data)
        hit = sum(int(x==y) for (x,y) in test_results)
        # retorna taxa de acerto
        return (hit/n)

    def SGD(self, training_data, epochs, mini_batch_size, eta, val_data=None):
        n = len(training_data)
        # para cada epoch, embaralha o conjunto de treino, faz mini batches de tamanho definido, recalcula pesos e biases
        for j in xrange(epochs):
            random.shuffle(training_data)
            # adaptar com get_mini_batches
            mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
            # se houver conjunto de teste, usa a rede atual para ver o hit rate
            if val_data:
                print "Epoch {0} - hit rate: {1}".format(j, evaluate(val_data))
            # senão, a epoch acabou e vamos para a próxima
            else:
                print "Epoch {0} complete.".format(j)
