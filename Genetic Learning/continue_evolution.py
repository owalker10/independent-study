import species
from matplotlib import pyplot as plt
import numpy as np
import pickle
import game

'''
This script takes an evolved population and continues to evolve it
'''

imr = 0.15
amr = 0.00
size = 100


species.X = 150
species.Y = 300
species.R = 25
game.jump_speed = 20
game.WIDTH,game.HEIGHT = 800,600
game.speed = 6
game.barrier_prob = 40
game.barrier_width = 75
game.vision_rows = game.rows(10)

with open('population4.pkl','rb') as file:
    population = pickle.load(file)

fitnesses = []
num_gens = 200
for i in range(population.gen+1,population.gen+num_gens+1):
    population.evolve()
    population.calculate_fitness()
    fitness = population.max_fitnesses[-1]
    fitnesses.append(fitness)
    print('Generation',population.gen,':',fitness)


plt.plot(fitnesses)

with open('population4.2.pkl','wb') as file:
    pickle.dump(population,file)

plt.show()
