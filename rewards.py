import csv
import os

import numpy.random

numpy.random.seed()

# Pre-load the network latencies into memory
NETWORK_LATENCIES_UNNORMALIZED = [
	[float(v) for v in row.itervalues()]
	for row in csv.DictReader(open('datasets/univ-latencies/univ-latencies.txt'))]

min_latency = min([min(row) for row in NETWORK_LATENCIES_UNNORMALIZED])
max_latency = max([max(row) for row in NETWORK_LATENCIES_UNNORMALIZED])

NETWORK_LATENCIES = [[(float(v) - min_latency) / max_latency for v in row]
					 for row in NETWORK_LATENCIES_UNNORMALIZED]

NETWORK_LATENCY_MUS = []
for site in range(len(NETWORK_LATENCIES[0])):
	NETWORK_LATENCY_MUS.append(sum(row[site] for row in NETWORK_LATENCIES) / len(NETWORK_LATENCIES))
	
print "Done loading network latencies"

def iter_network_latencies(num_arms):
	for row in NETWORK_LATENCIES:
		return row[:num_arms]

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


		

