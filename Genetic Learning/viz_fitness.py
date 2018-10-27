from matplotlib import pyplot as plt
import pickle

with open('population0.pkl','rb') as file:
    population = pickle.load(file)

max_f = population.max_fitnesses
avg_f = [sum(gen.fitnesses[-10:])/10 for gen in population.generations]

plt.plot(max_f)
plt.plot(avg_f)

b,t = plt.ylim()
plt.ylim(b,25000)
plt.show()
