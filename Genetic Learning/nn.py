import numpy as np
from matplotlib import pyplot as plt

'''
Simple neural network class
Meant for genetic algorithm, so doesn't support backpropogation, but rather mutations and crossovers
This also means forward propogation only supports vectors of one observation, not matrices of multiple observations
'''

#np.random.seed(13)
# creating a vectorized function of the derivative of the relu function
d_relu = np.vectorize(lambda x: 1 if x > 0 else 0)
# used to make sure zeroes aren't being used in places like log(x) or 1/x
nonzero = np.vectorize(lambda x: 0.1**10 if x == 0 else x)

class NeuralNetwork(object):
    def __init__(self,layers,activations,cost_func,imr=0.05,amr=0.01,weights=None,biases=None):
        if weights is None:
            weights = []
            biases = []
            for l in range(len(layers[:-1])):
                w = np.random.randn(layers[l+1],layers[l]) # rows are receiving neurons, columns are previous neurons
                b = np.random.randn(layers[l+1])
                weights.append(w)
                biases.append(b)
        self.weights = weights
        self.biases = biases

        self.activations = activations

        self.imr = imr # incremental mutation rate (adding/subtracting small numbers)

        self.amr = amr # assignment mutation rate (completely reinitializing value)

        self.cost_func = cost_func

        self.costs = []

        self.layers = layers





    # cost function (thing the NN is trying to minimize)
    def cost(self,y_hat,y):
        if self.cost_func == 'SSE': # sum of squares error
            return np.sum(0.5*(y_hat-y)**2)

        elif self.cost_func == 'CE': # cross entropy
            return -np.sum(np.log(nonzero(y_hat))*y)/y.size

    # derivative of the cost function
    def d_cost(self,y_hat,y):
        if self.cost_func == 'SSE':
            return (y_hat-y)

        elif self.cost_func == 'CE':

            return -(y*(1.0/nonzero(y_hat)) + (1.0-y)*(1.0/nonzero(1-y_hat)))

    # activation function of each neuron
    def activ(self,z,n):
        if self.activations[n] == 'sigmoid':
            return 1.0/(1.0+np.exp(-z))

        elif self.activations[n] == 'relu':
            return np.maximum(np.zeros(z.shape),z)

        elif self.activations[n] == 'linear':
            return z

    # derivative of the activation function
    def d_activ(self,z,n):
        if self.activations[n] == 'sigmoid':
            return self.activ(z,n)*(1-self.activ(z,n))

        elif self.activations[n] == 'relu':
            return d_relu(z)

        elif self.activations[n] == 'linear':
            return np.full(z.shape,1)


    # feed the input values all the way through the network
    def forward_propogation(self,x):
        activs = [x.T]
        zs = []
        for l,weights,biases in zip(range(len(self.layers[1:])),self.weights,self.biases):
            z = np.dot(weights,activs[l]) + biases # multiply previous layer values by the weights, add biases (1-D vector length of the next layer)
            zs.append(z)
            a = self.activ(z,l) # put the input values through the activation function
            activs.append(a)

        return activs,zs

    # change the weights and biases of the neural network
    def mutate(self):
        for n in range(len(self.weights)):
            # weights
            im_mask = np.random.rand(self.weights[n].shape[0],self.weights[n].shape[1]) < self.imr
            im_vals = np.random.normal(size=self.weights[n].shape)/5

            am_mask = np.random.rand(self.weights[n].shape[0],self.weights[n].shape[1]) < self.amr
            am_vals = np.random.rand(self.weights[n].shape[0],self.weights[n].shape[1])

            self.weights[n][im_mask] += im_vals[im_mask]
            self.weights[n][am_mask] = am_vals[am_mask]

            # biases
            im_mask = np.random.rand(self.biases[n].shape[0]) < self.imr
            im_vals = np.random.normal(size=self.biases[n].shape)/5

            am_mask = np.random.rand(self.biases[n].shape[0]) < self.amr
            am_vals = np.random.rand(self.biases[n].shape[0])

            self.biases[n][im_mask] += im_vals[im_mask]
            self.biases[n][am_mask] = am_vals[am_mask]

    # create a child network by randomly selecting weights and biases from two parent networks
    def crossover(self,other):
        weights = []
        biases = []
        for n in range(len(self.weights)):
            w = np.zeros(self.weights[n].shape)
            mask = np.random.rand(self.weights[n].shape[0],self.weights[n].shape[1])<0.5 # random boolean mask, should contain roughly equal amounts of True and False
            w[mask] = self.weights[n][mask]
            w[np.logical_not(mask)] = other.weights[n][np.logical_not(mask)]

            b = np.zeros(self.biases[n].shape)
            mask = np.random.rand(self.biases[n].shape[0])<0.5
            b[mask] = self.biases[n][mask]
            b[np.logical_not(mask)] = other.biases[n][np.logical_not(mask)]

            weights.append(w)
            biases.append(b)


        child_nn = NeuralNetwork(self.layers,self.activations,self.cost_func,
                                imr=self.imr,amr=self.amr,weights=weights, biases=biases)

        return child_nn

    def clone(self):
        weights = [w.copy() for w in self.weights]
        biases = [b.copy() for b in self.biases]

        return NeuralNetwork(self.layers,self.activations,self.cost_func,imr=self.imr,amr=self.amr,weights=self.weights,biases=self.biases)

if __name__ == '__main__':
    test = NeuralNetwork([10,5,5,1],['relu','relu','sigmoid'],'SSE')
    output = test.forward_propogation(np.random.rand(10))
    print(output[0][-1])

    test2 = NeuralNetwork([10,5,5,1],['relu','relu','sigmoid'],'SSE')
    child = test.crossover(test2)

    data = np.random.rand(10)
    output = child.forward_propogation(data)
    print(output[0][-1])

    child.mutate()
    output = child.forward_propogation(data)
    print(output[0][-1])

    twin = child.clone()
    output = twin.forward_propogation(data)
    print(output[0][-1])
