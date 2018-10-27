import numpy as np
from matplotlib import pyplot as plt

'''
This script contains a neural network class. The class supports several options for building a nerual network, including:
- the number of layers and number of neurons for each layer
- the activation function for each later
- the learning rate (alpha) for backpropogation
- the cost function to minimize

This neural network is build to take a single vector of input at a time, and so is better suited for temporal-based reinforcement learning than supervised learning
'''

#np.random.seed(13)
# creating a vectorized function of the derivative of the relu function
d_relu = np.vectorize(lambda x: 1 if x > 0 else 0)
# used to make sure zeroes aren't being used in places like log(x) or 1/x
nonzero = np.vectorize(lambda x: 0.1**10 if x == 0 else x)

def NeuralNetwork(layers,activations,alpha,cost_func,rem_costs=False,save_mem=False,kind='single',weights=None,biases=None):
    if kind == 'single':
        return NeuralNetwork_Single(layers,activations,alpha,cost_func,rem_costs=rem_costs,save_mem=save_mem,weights=weights,biases=biases)
    elif kind == 'batch':
        return NeuralNetwork_Batch(layers,activations,alpha,cost_func,rem_costs=rem_costs,save_mem=save_mem,weights=weights,biases=biases)


class NeuralNetwork_Single(object):
    def __init__(self,layers,activations,alpha,cost_func,rem_costs=False,save_mem=False,weights=None,biases=None):
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

        self.alpha = alpha

        self.cost_func = cost_func

        self.costs = []

        self.layers = layers

        self.rem_costs = rem_costs

        self.save_mem = save_mem

    # cost function (thing the NN is trying to minimize)
    def cost(self,y_hat,y):
        if self.cost_func == 'SSE': # sum of squares error
            return np.sum(0.5*(y_hat-y)**2)/y.shape[0]

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

    # use gradient descent to adjust weights and biases
    def backpropogation(self,x,y):
        activs,zs = self.forward_propogation(x)
        y_hat = activs[-1]
        self.y_hat = y_hat

        delta_b = [] # list of gradient adjustments (change in cost with respect to biases)
        delta_w = [] # list of gradient adjustments (change in cost with respect to weights)

        Cost = self.cost(y_hat,y) # cost of the iteration
        if self.rem_costs: self.costs.append(Cost)

        dCda_dadz = self.d_cost(y_hat,y)*(self.d_activ(zs[-1],-1)) # vector of length number of neurons
        #dC/da * da/dz (derivative of cost with respect to activation * derivative of activation with respect to input)

        db = dCda_dadz # change in cost with respect to bias = dC/da * da/dz * 1

        #dw = np.outer(dCda_dadz,activs[-2])
        dw = np.dot(dCda_dadz[:,np.newaxis],activs[-2][np.newaxis])
        # change in cost with respect to weight = dC/da * da/dz * a(l-1)

        delta_b.append(db)
        delta_w.append(dw)

        for l in range(2,len(self.layers)):
            weights = self.weights[-l+1]
            dCda_dadz = np.dot(weights.T,dCda_dadz)*self.d_activ(zs[-l],-l)
            #dC/da * da/dz (derivative of cost with respect to activation * derivative of activation with respect to input)

            db = dCda_dadz # change in cost with respect to bias = dC/da * da/dz * 1

            dw = np.dot(dCda_dadz[:,np.newaxis],activs[-l-1][np.newaxis])
            #dw = np.outer(dCda_dadz,activs[-l-1])
            # change in cost with respect to weight = dC/da * da/dz * a(l-1)

            delta_b.append(db)
            delta_w.append(dw)

        delta_b = delta_b[::-1]
        delta_w = delta_w[::-1]

        if self.save_mem:
            del activs,zs,y_hat,dCda_dadz,db,dw,weights
            self.weights = [w.astype(np.float32) for w in self.weights]
            self.biases = [b.astype(np.float32) for b in self.biases]
            delta_w = [dw.astype(np.float32) for dw in delta_w]
            delta_b = [db.astype(np.float32) for db in delta_b]


        # adjust the weights and biases based on the calculated gradients
        self.weights = [w - self.alpha*dw for w,dw in zip(self.weights,delta_w)]
        self.biases = [b - self.alpha*db for b,db in zip(self.biases,delta_b)]



        return Cost

class NeuralNetwork_Batch(object):
    def __init__(self,layers,activations,alpha,cost_func,rem_costs=False,save_mem=False,weights=None,biases=None):
        if weights is None:
            weights = []
            biases = []
            for l in range(len(layers[:-1])):
                #w = np.random.randn(layers[l+1],layers[l]))*0.01 # rows are receiving neurons, columns are previous neurons
                w = np.random.randn(layers[l+1],layers[l])*np.sqrt(2/layers[l])
                b = np.random.randn(layers[l+1],1)
                weights.append(w)
                biases.append(b)
        self.weights = weights
        self.biases = biases

        self.activations = activations

        self.alpha = alpha

        self.cost_func = cost_func

        self.costs = []

        self.layers = layers

        self.rem_costs = rem_costs

        self.save_mem = save_mem

    # cost function (thing the NN is trying to minimize)
    def cost(self,y,y_hat):
        if self.cost_func == 'SSE': # sum of squares error
            #print('y,y_hat',y,y_hat)
            return np.mean(0.5*(y-y_hat)**2)

        elif self.cost_func == 'CE': # cross entropy
            return -np.sum(np.log(nonzero(y_hat))*y)/y.size

    # derivative of the cost function
    def d_cost(self,y,y_hat):
        if self.cost_func == 'SSE':
            return (y-y_hat)

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
        activs = [x]
        zs = []
        for l,weights,biases in zip(range(len(self.layers[1:])),self.weights,self.biases):
            z = np.matmul(weights,activs[l]) + biases # multiply previous layer values by the weights, add biases (1-D vector length of the next layer)
            zs.append(z)
            a = self.activ(z,l) # put the input values through the activation function
            activs.append(a)

        return activs,zs

    # use gradient descent to adjust weights and biases
    def backpropogation(self,x,y):
        activs,zs = self.forward_propogation(x)

        y_hat = activs[-1]
        self.y_hat = y_hat

        delta_b = [] # list of gradient adjustments (change in cost with respect to biases)
        delta_w = [] # list of gradient adjustments (change in cost with respect to weights)

        Cost = self.cost(y,y_hat) # cost of the iteration

        if self.rem_costs: self.costs.append(Cost)

        dCda_dadz = self.d_cost(y,y_hat)*(self.d_activ(zs[-1],-1)) # vector of length number of neurons
        #dC/da * da/dz (derivative of cost with respect to activation * derivative of activation with respect to input)

        db = dCda_dadz # change in cost with respect to bias = dC/da * da/dz * 1
        dw = np.einsum('i...,j...->ij...',dCda_dadz,activs[-2])
        # change in cost with respect to weight = dC/da * da/dz * a(l-1)


        dw = np.mean(dw,axis=2)
        db = np.mean(db,axis=1).reshape(db.shape[0],1)


        delta_b.append(db)
        delta_w.append(dw)

        for l in range(2,len(self.layers)):
            weights = self.weights[-l+1]
            dCda_dadz = np.matmul(weights.T,dCda_dadz)*self.d_activ(zs[-l],-l)

            #dC/da * da/dz (derivative of cost with respect to activation * derivative of activation with respect to input)

            db = dCda_dadz # change in cost with respect to bias = dC/da * da/dz * 1



            dw = np.einsum('i...,j...->ij...',dCda_dadz.astype(np.float32),activs[-l-1].astype(np.float32))
            # change in cost with respect to weight = dC/da * da/dz * a(l-1)

            dw = np.mean(dw,axis=2)
            db = np.mean(db,axis=1).reshape(db.shape[0],1)


            delta_b.append(db)
            delta_w.append(dw)

        delta_b = delta_b[::-1]
        delta_w = delta_w[::-1]

        if self.save_mem:
            del activs,zs,y_hat,dCda_dadz,db,dw,weights
            self.weights = [w.astype(np.float32) for w in self.weights]
            self.biases = [b.astype(np.float32) for b in self.biases]
            delta_w = [dw.astype(np.float32) for dw in delta_w]
            delta_b = [db.astype(np.float32) for db in delta_b]

        # adjust the weights and biases based on the calculated gradients
        self.weights = [w + self.alpha*dw for w,dw in zip(self.weights,delta_w)]
        self.biases = [b + self.alpha*db for b,db in zip(self.biases,delta_b)]

        return Cost


if __name__ == '__main__':
    #the following code is used to test the neural network, and the functionality of all the parameters

    # this is really just to ensure that the NN works like it should

    nn = NeuralNetwork([4,3,2],['relu','sigmoid'],0.5,'SSE')

    x = np.array([0,1,0,1])
    y = np.array([0,1])
    print(nn.forward_propogation(x)[0])

    for n in range(10000):
        nn.backpropogation(x,y)
    #print(nn.costs)

    plt.plot([n for n in range(10000)],nn.costs)
    plt.yscale('log')
    plt.xscale('log')
    plt.show()
