#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import random
import layer
from json_handler import JsonHandler
from plot import plot


class Network(object):

    def __init__(self, sizes, save_rate, checkpoints_dir, activation_function):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.layers = layer.create_layers(self.sizes, activation_function)
        self.save_rate = save_rate
        self.checkpoints_dir = checkpoints_dir
        self.json_handler = JsonHandler()

    def feedforward(self, a, keep_z=False):
        self.layers[0].activation = a.reshape(a.shape[0],1)
        i = 1
        for layer in self.layers[1:]:
            layer.update_layer(self.layers[i-1].activation, keep_z)
            i +=1
        return layer.activation

    def update_mini_batch(self, mini_batch, eta):
        # nablas terão formato de acordo com seus respectivos layers
        nabla_b = [np.zeros(layer.bias.shape) for layer in self.layers[1:]]
        nabla_w = [np.zeros(layer.weight.shape) for layer in self.layers[1:]]
        mini_batch_length = len(mini_batch)

        for x,y in mini_batch: # para cada exemplo de treino da mini batch, calcula o ajuste necessário
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb+dnb for nb,dnb in zip(nabla_b, delta_nabla_b)] # dC/db
            nabla_w = [nw+dnw for nw,dnw in zip(nabla_w, delta_nabla_w)] # dC/dw

        for i, (nw, nb) in enumerate(zip(nabla_w, nabla_b)):
            self.layers[i+1].weight -= (eta/mini_batch_length)*nw # update weight
            self.layers[i+1].bias   -= (eta/mini_batch_length)*nb # update bias

    def backprop(self, x, y):
        # feedforward, passa pela rede indo em direção a ultima camada, calculando os zs e as ativações
        self.feedforward(x, keep_z=True)
        activation = self.layers[-1].activation

        # output error (calcula a última camada "na mão")
        nabla_b = []
        nabla_w = []
        for l in self.layers[1:]:
            nabla_b.append(np.zeros(l.bias.shape))
            nabla_w.append(np.zeros(l.weight.shape))
        delta = self.cost_derivative(activation, y) * self.layers[-1].activation_function(self.layers[-1].z, prime=True) # (BP1)
        nabla_b[-1] = delta # (BP3)
        nabla_w[-1] = np.dot(delta, self.layers[-2].activation.transpose()) # (BP4)

        # backpropagate the error, l é usado de forma crescente, mas como acessar posições
        # negativas significa acessar de trás pra frente, o erro é propagado do fim ao começo da rede
        for l in xrange(2, self.num_layers):
            z = self.layers[-l].z
            afp = self.layers[-l].activation_function(z, prime=True)
            delta = np.dot(self.layers[-l+1].weight.transpose(), delta) * afp # (BP2)
            nabla_b[-l] = delta # (BP3)
            nabla_w[-l] = np.dot(delta, self.layers[-l-1].activation.transpose()) # (BP4)
        return (nabla_b, nabla_w)


    def cost_derivative(self, output_activations, y):
        res = np.subtract(output_activations, y)
        return res

    def evaluate(self, test_data):
        # guarda resultados passando o conjunto de teste pela rede
        # e assume o maior resultado como resposta da rede
        test_results = []
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        n = len(test_data)
        hit = sum(int(x==y) for (x,y) in test_results)
        # retorna taxa de acerto
        return (float(hit)/float(n))

    def SGD(self, training_data, epochs, mini_batch_size, eta, val_data=None):
        x = []
        y = []
        # para cada epoch, embaralha o conjunto de treino, faz mini batches de tamanho definido, recalcula pesos e biases
        for j in xrange(epochs):
            # each mini_batch contains a list of indexes, each index corresponds
            # to an example
            # mini_batches = self.dataHandler.get_mini_batches(minBatch_size=mini_batch_size)
            n = len(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]
            i = 1
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
                i+=1

            if j > 0 and j % self.save_rate == 0:
                self.save_learning(self.checkpoints_dir + 'epoch' + str(j) + '.json')

            # se houver conjunto de teste, usa a rede atual para ver o hit rate
            if val_data != None:
                print "Epoch {0} - hit rate: {1}".format(j, self.evaluate(val_data))
                x.append(j)
                y.append(self.evaluate(val_data))

            # senão, a epoch acabou e vamos para a próxima
            else:
                print "Epoch {0} complete.".format(j)

        if val_data:
            plot(x,y)

    def save_learning(self, ckpt_name):
        # Dict to save the params from the checkpoint
        ckpt = {}
        weights = {}
        biases = {}

        for i,layer in enumerate(self.layers):
            weights['l'+str(i)] = layer.weight.tolist();
            biases['l'+str(i)] = layer.bias.tolist();


        ckpt = {"weights": weights, "biases": biases}

        self.json_handler.write(ckpt, ckpt_name)

    def load_learning(self, ckpt_file):
        # Loads the parameters from the checkpoint file
        params = self.json_handler.read(ckpt_file)        
        weights = params['weights']
        biases = params['biases']

        # Fills the layers with the weights and biases
        for i in xrange(len(self.layers)):
            self.layers[i].weight = np.asarray(weights['l'+str(i)]).astype("float64")
            self.layers[i].bias = np.asarray(biases['l'+str(i)]).astype("float64")

    def predict(self, x):
        # x must have the same size of the input layer
        self.feedforward(x)
        return np.argmax(self.layers[-1].activation)

    def weights_for_humans(self, img_dir):
        # This function saves the weights in image format
        # So humans can try to see the magic better
        # But for now, it is only possible if the number of neurons in a layer
        # is a square number
        for i,layer in enumerate(self.layers[1:]):
            k = len(layer.weight[0])
            sqrt_k = int(math.sqrt(k))
            # checks if it has a square size
            if sqrt_k * sqrt_k == k:
                img = np.zeros((k, 3), dtype="uint8")
                for j,weight in enumerate(layer.weight):
                    p = np.argwhere(weight >= 0)
                    n = np.argwhere(weight < 0)

                    img[p[:,0], :] = np.absolute(weight[p]) * np.array([0, 255, 0]) # green
                    img[n[:,0], :] = np.absolute(weight[n]) * np.array([0, 0, 255]) # red

                    if img_dir[-1] == '/':
                        img_name = img_dir+"l"+str(i+1)+"w"+str(j)+".jpg"
                    else:
                        img_name = img_dir+'/'+"l"+str(i+1)+"w"+str(j)+".jpg"

                    cv2.imwrite(img_name, img.reshape(sqrt_k, sqrt_k, 3) )
