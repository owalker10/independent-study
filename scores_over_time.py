import numpy as np
from matplotlib import pyplot as plt

'''
This script plots the reinforcement agent's life lengths over time, to show how it learns
'''

scores = np.load('rl_scores.npy')

def average_scores(s,avg):
    return list(range(0,s.shape[0],avg)),np.array([np.mean(s[n:n+avg] if n+avg<=s.shape[0] else s[n:]) for n in range(0,s.shape[0],avg)])

x10,avg_scores10 = average_scores(scores,10)
#x50,avg_scores50 = average_scores(scores,50)
x250,avg_scores250 = average_scores(scores,250)
x750,avg_scores750 = average_scores(scores,750)


plt.plot(x10,avg_scores10,alpha=0.5,label='avg. over 10 lives')
#plt.plot(x50,avg_scores50,alpha=0.5)
plt.plot(x250,avg_scores250,color='red',alpha=0.8,label='avg. over 250 lives')
plt.plot(x750,avg_scores750,linestyle='--',color='black',label='avg. over 750 lives')

plt.title('Frames per Life of Autonomous Player')
plt.xlabel('Life')
plt.ylabel('Frames Alive')

plt.legend()

plt.show()
