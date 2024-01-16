#################################################################
#################################################################
#####              GAUSSIAN THOMPSON SAMPLING              ######           
#################################################################           
#################################################################
#################################################################

# Libraries
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Initial conditions
N_EVENTS = 10000
bandit_means = [1, 2, 3]

# Definition of actions that a bandit does
class Bandit():
    def __init__(self, true_mean):
        # Prior parameters for X
        self.true_mean = true_mean
        self.precision = 1 # 1/variance
        # Prior parameters for mu ~ N(0,1)
        self.m = 0
        self.lambda_ = 1
        self.N = 0
    
    # Action of pulling the bandit
    def pull(self):
        # Extraction of sample from a normal distribution with mean true_mean and precision
        return np.random.randn()/np.sqrt(self.precision) + self.true_mean
    
    # Generation of a sample to update the bandit
    def sample(self):
        # As the posterior has normal distribution a sample is extracted with mean m and precision lambda_ 
        return np.random.randn()/np.sqrt(self.lambda_) + self.m
    
    # Updating parameters of the bandit
    def update(self, x):
        self.m = (self.lambda_*self.m + self.precision*x)/(self.precision + self.lambda_)
        self.lambda_ += self.precision
        self.N += 1
        
# Inside of the experiment plots of the distribution will printed as the number of elements grow
def plot(bandits, trial):
    x = np.linspace(-3, 6, 200)
    for b in bandits:
        y = norm.pdf(x, b.m, np.sqrt(1. /b.lambda_))
        plt.plot(x, y, label=f"real mean: {b.true_mean:.4f}, num plays: {b.N}")
    
    plt.title(f"Bandit distributions after {trial} trials")
    plt.legend()
    plt.show()
    
# Running the experiment
def experiment():
    bandits = [Bandit(p) for p in bandit_means]

    sample_points = [5,10,20,50,100,200,500,1000,1500,1999]
    rewards = np.empty(N_EVENTS)

    for i in range(N_EVENTS):
        # Thompson sampling
        j = np.argmax([b.sample() for b in bandits])

        # plot the posteriors
        if i in sample_points:
          plot(bandits, i)
    
        # pull the arm for the bandit with the largest sample
        x = bandits[j].pull()
        # update the distribution for the bandit whose arm we just pulled
        bandits[j].update(x)

        # update rewards
        rewards[i] = x

    cumulative_average = np.cumsum(rewards) / (np.arange(N_EVENTS) + 1)

    # plot moving average ctr
    plt.plot(cumulative_average)
    for m in bandit_means:
      plt.plot(np.ones(N_EVENTS)*m)
    plt.show()

    return cumulative_average

if __name__ == "__main__":
  experiment()

        
    
        
        