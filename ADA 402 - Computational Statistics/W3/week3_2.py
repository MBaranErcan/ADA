import matplotlib.pyplot as plt
import scipy
from scipy import stats
import numpy as np
from statsmodels.graphics.gofplots import qqplot



def return_sample(n: int) -> list:
    sample_list = []
    for i in range(n):
        for j in range(n):
            u = np.random.rand()
            x = np.random.rand()
            if u < x*(1-x):
                sample_list.append(x)
            else:
                continue
    return np.array(sample_list)
    
sample = return_sample(1000)

print(len(sample))

plt.hist(sample, bins = 100, density= True)

beta_twotwo = stats.beta(2, 2)

real_sample = beta_twotwo.rvs(len(sample))

plt.hist(real_sample, bins= 1000, density= True)


qqplot(sample, beta_twotwo)

np.random.rand