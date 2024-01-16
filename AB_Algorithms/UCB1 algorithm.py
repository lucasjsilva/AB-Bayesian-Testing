#################################################################
#################################################################
#####                    UCB1 ALGORITHM                    ######           
#################################################################           
#################################################################
#################################################################

# Libraries

import numpy as np
import matplotlib.pyplot as plt

# Initial conditions
N_EVENTS = 10000
bandits_probabilities = [0.2, 0.5, 0.75]

# Definition of actions that a bandit does
class Bandit():
    def __init__(self, p):
        self.p = p
        self.p_estimate = 0. # Probability estimated for that bandit
        self.N = 1. # Number of events ran
        
    def pull(self):
        # Draw 1 with probability p
        return np.random.random() < self.p
    
    def update(self, x):
        # Updates the estimated probability and the count of the event
        self.N += 1
        self.p_estimate = ((self.N-1)*self.p_estimate + x)/self.N
        
# Definition of function that calculates the estimative of the mean value
def ucb(mean, n, nj):
    # Mean is the real mean value, n is the number of samples and nj the number of samples for the bandit
    return mean + np.sqrt(2*np.log(n)/nj)
        
# Running the experiment for the class Bandits considering the initial conditions
def experiment():
    bandits = [Bandit(p) for p in bandits_probabilities]
    # List of rewards
    rewards = np.zeros(N_EVENTS)
    # Total plays
    total_plays = 0
    
    # Play each bandit once in order to determine all of the js
    for j in range(len(bandits_probabilities)):
        x = bandits[j].pull()
        total_plays += 1
        bandits[j].update(x)
        
    
    # Loop for each event
    for i in range(N_EVENTS):
        j = np.argmax([ucb(b.p_estimate, total_plays, b.N) for b in bandits])
        x = bandits[j].pull()
        total_plays += 1
        bandits[j].update(x)
        # Saving the results that were succeeded
        rewards[i] = x
        
    # After the loop, plotting the results 
    cumulative_average = np.cumsum(rewards)/(np.arange(N_EVENTS) + 1)
    
    plt.plot(cumulative_average)
    plt.plot(np.ones(N_EVENTS)*np.max(bandits_probabilities))
    plt.xscale('log')
    plt.show()

    # plot moving average ctr linear
    plt.plot(cumulative_average)
    plt.plot(np.ones(N_EVENTS)*np.max(bandits_probabilities))
    plt.show()

    for b in bandits:
      print(b.p_estimate)

    print("total reward earned:", rewards.sum())
    print("overall win rate:", rewards.sum() / N_EVENTS)
    print("num times selected each bandit:", [b.N for b in bandits])

    return cumulative_average

if __name__ == '__main__':
    experiment()
        
        