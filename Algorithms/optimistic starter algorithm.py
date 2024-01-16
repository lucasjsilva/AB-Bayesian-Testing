#################################################################
#################################################################
#####             OPTIMISTIC STARTER ALGORITHM             ######           
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
class Bandit:
    def __init__(self, p):
        self.p = p
        self.p_estimate = 5. # Probability estimated for that bandit
        self.N = 1. # Number of events ran
        
    def pull(self):
        # Draw 1 with probability p
        return np.random.random() < self.p
    
    def update(self, x):
        # Updates the estimated probability and the count of the event
        self.N += 1
        self.p_estimate = ((self.N - 1)*self.p_estimate + x) / self.N
        

# Running the experiment for the class Bandits considering the initial conditions
def experiment():
    bandits = [Bandit(p) for p in bandits_probabilities]
    # List of the granted rewards sized as total events
    rewards = np.zeros(N_EVENTS)
    
    # Loop for each event
    for i in range (N_EVENTS):
        j = np.argmax([b.p_estimate for b in bandits])
        
        # Pull the arm of the Bandit
        x = bandits[j].pull()
        # Save result in rewards
        rewards[i] = x
        
        # Update the bandit
        bandits[j].update(x)
        
    # After all the events the mean value of each bandit will be printed
    for b in bandits:
        print(f"The mean value of the bandit with probability {b.p} is {b.p_estimate}")
        
    # Print of the results
    print(f'Total of rewards gained is {rewards.sum()}')
    print(f'The rate of wins is {rewards.sum()/N_EVENTS}')
    print(f"Number of times each bandit was selected {[b.N for b in bandits]}")
    
    # plot the results
    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(N_EVENTS) + 1)
    plt.ylim([0, 1])
    plt.plot(win_rates)
    plt.plot(np.ones(N_EVENTS)*np.max(bandits_probabilities))
    plt.show()

if __name__ == "__main__":
  experiment()
    
