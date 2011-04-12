
import numpy.random

numpy.random.seed()

def iter_uniform_plus_eps(num_arms, eps=0.01):
	while True:
		rewards = numpy.random.uniform(low=0.0, high=1.0, size=num_arms)
		rewards[0] += eps
		yield rewards

def iter_normal(num_arms, mus, sigmas):
	assert len(mus) == len(sigmas) == num_arms
	while True:
		rewards = [numpy.random.normal(mu, sigma, 1)[0]
				   for (mu, sigma) in zip(mus, sigmas)]
		yield rewards

def iter_bernoulli(num_arms, ps, low=0.0, high=1.0):
	assert num_arms == len(ps)
	while True:
		rewards = []
		for i in range(num_arms):
			success = numpy.random.random() < ps[i]
			rewards.append(high if success else low)
		yield rewards


		

