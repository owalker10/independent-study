from graphics import *
import pickle
import numpy as np
from math import ceil

'''
This script produces a visual of the brain of one of the evolved EL players using Graphics
'''

WIDTH, HEIGHT = 1200,800

win = GraphWin('Neural Network',WIDTH,HEIGHT)

win.setBackground('white')

with open('population5.pkl','rb') as file:
    population = pickle.load(file)

brain = population.generations[229].players[-1].brain

i_h0 = np.zeros((15,13))
h0_h1 = np.zeros((15,16))
h1_o = np.zeros((1,16))

i_h0[:,:12] = brain.weights[0]
h0_h1[:,:15] = brain.weights[1]
h1_o[:,:15] = brain.weights[2]

i_h0[:,12] = brain.biases[0]
h0_h1[:,15] = brain.biases[1]
h1_o[:,15] = brain.biases[2]




for ws,xs in zip([i_h0,h0_h1,h1_o], [(100,450),(450,800),(800,960)]):
    pos = ws[np.where(ws>0)]
    neg = ws[np.where(ws<=0)]
    max_pos = pos.max()
    max_neg = (-1*neg).max()*-1
    for j in range(ws.shape[0]):
        p2 = Point(xs[1],50+700*j//(ws.shape[0]))
        if ws is h1_o: p2 = Point(xs[1],HEIGHT//2)
        for i in range(ws.shape[1]):
            weight = Line(Point(xs[0],50+700*i//(ws.shape[1]-1)),p2)
            if ws[j,i] > 0:
                thickness = ceil(ws[j,i]/max_pos*3)
                color = 'green'
            else:
                thickness = ceil(ws[j,i]/max_neg*3)
                color = 'red'
            weight.setOutline(color)
            weight.setWidth(thickness)
            weight.draw(win)


inputs = []
for i in range(12):
    neuron = Circle(Point(100,50+700*i//12),17)
    neuron.setFill('blue')
    neuron.draw(win)
    inputs.append(neuron)


hidden0 = []
for i in range(15):
    neuron = Circle(Point(450,50+700*i//15),14)
    neuron.setFill(color_rgb(255,165,0))
    neuron.draw(win)
    hidden0.append(neuron)

    relu = Image(Point(450,50+700*i//15),'relu.png')
    relu.draw(win)

hidden1 = []
for i in range(15):
    neuron = Circle(Point(800,50+700*i//15),14)
    neuron.setFill(color_rgb(255,165,0))
    neuron.draw(win)
    hidden1.append(neuron)

    relu = Image(Point(800,50+700*i//15),'relu.png')
    relu.draw(win)


output = Circle(Point(960,HEIGHT//2),30)
output.setFill(color_rgb(221,160,221))
output.draw(win)
sigmoid = Image(Point(960,HEIGHT//2),'sigmoid.png')
sigmoid.draw(win)




biases = []
for i,x,r in zip(range(3),[100,450,800],[17,14,14]):
    neuron = Circle(Point(x,750),r)
    neuron.setFill('grey')
    neuron.draw(win)
    biases.append(neuron)

# labels of input neurons
text = ['row 1','row 2','row 3','row 4','row 5','row 6','row 7','row 8','row 9','row 10','y speed','height','biases']
for i,label in enumerate(text):
    t = Text(Point(40,50+700*i//12),label)
    t.draw(win)

    if i != 12:
        arrow = Line(Point(65,50+700*i//12),Point(100,50+700*i//12))
        arrow.setArrow('last')
        arrow.draw(win)

text = ['input layer','hidden layer 1 (ReLU)','hidden layer 2 (ReLU)']
for label,x in zip(text,[100,450,800]):
    t = Text(Point(x,20),label)
    t.draw(win)
t = Text(Point(990,250),'output layer (sigmoid)')
t.draw(win)

arrow = Line(Point(975,400),Point(1025,400))
arrow.setArrow('last')
arrow.draw(win)

text = Text(Point(1100,400),'jump confidence')
text.draw(win)

while True:
    key = win.getKey()
    if key == 'q': exit()
