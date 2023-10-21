import matplotlib.pyplot as plt
import scipy
from scipy import stats
import numpy as np
from statsmodels.graphics.gofplots import qqplot

uniform_rv = stats.uniform(0,1)
sample = uniform_rv.rvs(1000)


plt.hist(sample, label = "uniform on [0,1]", density= True, bins = 10)
plt.legend()
plt.show()

normal = stats.norm(0,1)
pseudo_normal_sample = normal.ppf(sample)

plt.hist(pseudo_normal_sample, label = "pseude_normal_sample", density= True, bins = 100)
plt.legend()
plt.show()


qqplot(pseudo_normal_sample, normal);  #### we are doing great job!!!!!!


scipy.stats.shapiro(pseudo_normal_sample)  #### we got it right!!!!!!! great successs!!!!!
