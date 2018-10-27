import numpy as np
import nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

'''
This is a supervised machine learning problem to test the function regression capabilities of my neural network
The network will be trained to predict the quality of wine based on several piecies of numerical information
'''

def sum_squares_error(y_hat,y_):
    return np.mean(0.5*(y_hat-y_)**2)

data = np.loadtxt('winequality-red.csv', delimiter=',', skiprows=1) # load data into numpy matrix

x = data[:,:data.shape[1]-1] # take all but the last column as input
y = data[:,data.shape[1]-1] # take the last columna as the expected output

# scale the x data between 0 and 1
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

# instantiate the neural network
model= nn.NeuralNetwork([x.shape[1],3,1],['linear','linear'],0.1,'SSE',kind='batch',rem_costs=True)

train_costs = []
test_costs = []

x, x_test, y, y_test = train_test_split(x,y,test_size=0.2)

# stochastic gradient descent: randomly pick 20% of data each iteration to train on
for i in range(1000):

    r = np.random.randint(0,x.shape[0],x.shape[0]//5)


    sx = x[r,:].T
    sy = y[r]
    print(i)
    train_costs.append(model.backpropogation(sx,sy))

    y_hat = model.forward_propogation(x_test.T)[0][-1].ravel()
    test_costs.append(sum_squares_error(y_hat,y_test))



print()

fig,axes = plt.subplots(nrows=2,ncols=1)
ax=axes[0]
ax.plot(train_costs,label='Training Costs')
ax.plot(test_costs,label='Testing Costs')

ax.set_yscale('log');ax.set_xscale('log')
ax.legend()
ax.set_title('Cost Over Training')
ax.set_xlabel('Training Iteration')
ax.set_ylabel('Sum of Squares Error')
ax.set_ylim(np.amin(train_costs[1:]),np.amax(train_costs[1:]))

y_hat = model.forward_propogation(x_test.T)[0][-1].ravel()
ax = axes[1]
xs = [n for n in range(30)]
ax.scatter(xs,y_test[xs],label='Real Quality')
ax.scatter(xs,y_hat[xs],label='Predicted Quality')
ax.legend()
ax.set_title('Residuals of ' +str(len(xs))+ ' Wines in Test Data')
ax.set_ylabel('Wine Quality')

bottom,top = ax.get_ylim()
for n in xs:
    ax.axvline(n,(min(y_test[n],y_hat[n])-bottom)/(top-bottom),(max(y_test[n],y_hat[n])-bottom)/(top-bottom),color='red')

plt.subplots_adjust(hspace=0.4)
plt.show()
