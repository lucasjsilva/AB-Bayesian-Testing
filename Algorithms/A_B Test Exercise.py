#################################################################
#################################################################
#####    EXERCISE: A/B TESTING FOR TWO DIFFERENT GROUPS    ######
#####   CLICKING OR NOT IN ADVERSTISEMENT. THE MAIN GOAL   ######
#####           IS CHECK FOR DIFFERENCE IN CTR             ######
#################################################################
#################################################################

import numpy as np
from scipy import stats
import pandas as pd

# First of all a seed is going to be set for replicability
np.random.seed(0)

# Reading the data base file
df = pd.read_csv('advertisement_clicks.csv')
print(df.head())

# Now, let's divide both groups so we can get different sets of information
A = df[df['advertisement_id'] == "A"]['action'].dropna().to_numpy()
B = df[df['advertisement_id'] == "B"]['action'].dropna().to_numpy()

# In order to check for any significant difference, it's possible to run a T-test
# (because we know the standard deviation) and see if t is between -1.96 and 1.96 
# or if the p-value is lower than 0.05
t, p = stats.ttest_ind(A, B)
print("T-Test for both sets: ,", t, p)

# As it's possible to reproduce the calculation "manually" that's what's going
# to be done
mu_A = np.mean(A)
mu_B = np.mean(B)
sigma_A = A.var()
sigma_B = B.var()
N1 = len(A)
N2 = len(B)

mu = mu_A - mu_B
sigma = np.sqrt(sigma_A/N1 + sigma_B/N2)
t = mu/sigma

nu1 = N1 - 1
nu2 = N2 - 1
df = (sigma_A / N1 + sigma_B / N2)**2 / ( (sigma_A*sigma_A) / (N1*N1 * nu1) + (sigma_B*sigma_B) / (N2*N2 * nu2) )
p = (1 - stats.t.cdf(np.abs(t), df=df))*2

print("Manual result: ", t, p)