from nn import NeuralNetwork
import functools
import game
from random import randint
import numpy as np
dist = game.dist

X,Y,R = 150,150,15 # starting coordinates and radius of player ball
THRESHOLD = 0.8 # decision threshold (if the NN output is greater than threshold, jump!)

'''
This script contains several classes that represent a species of Flappy Ball players
    - Player: one player, contains a Neural Network brain and can be told to play the game and evaluate its fitness
    - Generation: one generation of players, can be told to sort players by fitness and evolve a new generation
    - Population: every generation evolved in the training
'''

# player class that plays the game and holds the neural network, like one organism in the species
@functools.total_ordering # decorator for ordering and sorting
class Player():
    def __init__(self,num,imr,amr,io,hidden_layers,activations):
        layers = [io[0]] + hidden_layers + [io[1]]
        self.brain = NeuralNetwork(layers,activations,'SSE',imr=imr,amr=amr)

        self.num = num
        self.imr = imr
        self.amr = amr
        self.io = io
        self.hidden_layers = hidden_layers
        self.activations = activations

        self.fitness = 0

        self.x = X
        self.y = Y
        self.r = R
    # the player plays the game and returns its score (frames survived)
    def calculate_fitness(self):
        fitness = game.play(self)
        self.fitness = fitness

    def mutate(self):
        self.brain.mutate()

    def crossover(self,other):
        brain = self.brain.crossover(other.brain)
        child = Player(self.num,self.imr,self.amr,self.io,self.hidden_layers,self.activations)
        child.brain = brain
        return child

    def move(self,dx,dy):
        self.x += dx
        self.y += dy

    def clone(self):
        dolly = Player(self.num,self.imr,self.amr,self.io,self.hidden_layers,self.activations)
        dolly.brain = self.brain.clone()
        return dolly

    def decide(self,vision,n):
        x = np.array(vision)
        x[0:n] = x[0:n]/(game.WIDTH - self.x + self.r) # bound the values [0,1]
        p = self.brain.forward_propogation(x)[0][-1]
        return p > THRESHOLD

    def intersects(self,other):
        # if the type is a rectangular barrier, check to see if any points on the barrier's perimeter is within the circle's radius
        if type(other) == game.Barrier:
            # check top and bottom
            for x in range(other.x,other.x+other.width):
                for y in [other.y,other.y+other.height]:
                    if dist([x,y],[self.x,self.y]) < self.r:
                        return True; break
            # check left and right
            for y in range(other.y,other.y+other.height):
                for x in [other.x,other.x+other.width]:
                    if dist([x,y],[self.x,self.y]) < self.r:
                        return True; break
            return False

    # less than method used for sorting players by fitness
    def __lt__(self,other):
        return self.fitness < other.fitness

    # equals method used for sorting players by fitness
    def __eq__(self,other):
        return self.fitness == other.fitness

# generation of players created by an evolution
class Generation():
    def __init__(self,imr,amr,size,num,io,hidden_layers,activations):
        self.num = num
        self.imr = imr
        self.amr = amr
        self.size = size
        self.io = io
        self.hidden_layers = hidden_layers
        self.activations = activations

        self.players = []
        self.fitnesses = []

        if num == 0:
            for i in range(size):
                player = Player(i,imr,amr,io,hidden_layers,activations)
                self.players.append(player)

    # create a new generation by evolving this one based on fitness
    def evolve(self):
        new_players = []

        best_player = self.players[-1].clone() # take the best player and clone it for the next generation
        best_player.num = 0

        new_players.append(best_player)

        num_cross = (self.size-1)//2 # number of players of the new generation who will be bred from two new parents
        num_survive = self.size - num_cross - 1 # number of players of the new generation who will be survivors of the old generation

        parents = []
        for n in range(num_cross):
            parent1 = self.random_by_fitness()
            parent2 = self.random_by_fitness()
            while parent1.num == parent2.num or (parent1.num,parent2.num) in parents or (parent2.num,parent1.num) in parents: # avoid having two identical parents or same parents as another child
                parent1 = self.random_by_fitness()
                parent2 = self.random_by_fitness()
            parents.append((parent1.num,parent2.num))

            child = parent1.clone().crossover(parent2.clone()) # clone parents to disconnect their weight and bias references
            child.num = n + 1

            child.mutate() # mutate childs weights and biases

            new_players.append(child)

        survivors = []
        for n in range(num_survive):
            survivor = self.random_by_fitness()
            while survivor.num in survivors: # avoid having the same survivor
                survivor = self.random_by_fitness()
            survivors.append(survivor.num)

            survivor = survivor.clone()

            survivor.num = n + 1 + num_cross

            survivor.mutate()

            new_players.append(survivor)

        # create the new generation with the selected players
        next_gen = Generation(self.imr,self.amr,self.size,self.num+1,self.io,self.hidden_layers,self.activations)
        next_gen.players = new_players

        return next_gen

    # calculate fitnesses for all players in the generation
    def calculate_fitness(self):
        for n in range(self.size):
            self.players[n].calculate_fitness()
            self.fitnesses.append(self.players[n].fitness)

        self.players.sort()
        self.fitnesses.sort()
        max_fitness = self.players[-1].fitness

        return max_fitness

    # randomly pick a player based on a fitness distribution (higher fitness means more likely to be chosen)
    def random_by_fitness(self):
        total_fitness = sum(self.fitnesses)
        r = randint(0,total_fitness)

        count = 0
        for n,fit in enumerate(self.fitnesses):
            count+= fit
            if r < count:
                return self.players[n]
        return(self.players[-1])

# whole population of players, ordered by generations; number of generations increases over time
class Population():
    def __init__(self,imr,amr,size,io=(8,1),hidden_layers=[10,10],activations=['relu','relu','sigmoid']):
        self.generations = []
        self.max_fitnesses = []
        self.gen = 0

        self.generations.append(Generation(imr,amr,size,0,io,hidden_layers,activations))

    # evolve the current generation into a new one
    def evolve(self):
        new_gen = self.generations[self.gen].evolve()
        self.gen+=1
        self.generations.append(new_gen)

    # calculate the fitness for the current generation
    def calculate_fitness(self):
        max_fit = self.generations[self.gen].calculate_fitness()
        self.max_fitnesses.append(max_fit)
        return max_fit
