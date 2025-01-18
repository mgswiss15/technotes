# MG 2025-01-18: Likelihood of data-sample from given generative process

import torch
import matplotlib.pyplot as plt
from torch.distributions import Categorical, Normal, MixtureSameFamily, Independent
import math

# gaussian mixture 
class GaussianMixture(MixtureSameFamily):
  "2D gaussian mixture"

  def __init__(self, centers=[(-5., -5.), (-5., 5.), (5., -5.), (5., 5.)]):
    mix = Categorical(torch.ones(len(centers)))
    means = torch.stack([torch.tensor(center) for center in centers], 0)
    components = Independent(Normal(means, torch.ones(len(centers),2)), 1)
    super().__init__(mix, components)

gmm = GaussianMixture()
data = gmm.sample((1000,))

# normalize
data_mean = data.mean()
data_std = data.std()
data = (data - data_mean)/data_std

# plot
plt.figure(figsize=[5,5])
plt.scatter(data[:,0],data[:,1])
plt.axis([-2., 2., -2., 2.])
plt.show()

# log-likelihood
data = data*data_std + data_mean
loglikelihood = gmm.log_prob(data).mean()
print(f'Data log-likelihood: {loglikelihood}')

# swiss role
# https://jlmelville.github.io/smallvis/swisssne.html
n_samples = 5000
std = 1
phi = torch.rand(n_samples) * 3. * math.pi + 1.5 * math.pi
x = phi * torch.cos(phi) + torch.randn_like(phi)*std
y = phi * torch.sin(phi) + torch.randn_like(phi)*std

plt.figure(figsize=[5,5])
plt.scatter(x,y)
# plt.axis([-2., 2., -2., 2.])
plt.show()

#TODO: Not sure how to get the likelihood in this case

