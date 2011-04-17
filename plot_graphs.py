from __future__ import with_statement

import os

import matplotlib.pyplot as plt
import numpy
import simplejson

GRAPHS_ROWS = 4
GRAPHS_COLS = 4

GRAPHS = (
	
	# Uniform

	(# eps greedies
	),
	
	('UCB_Uniform_eps=0.01_K=100.json',
	 'UCB-Bernoulli_Uniform_eps=0.01_K=100.json',
	),
	
	('EXP3_Uniform_eps=0.001_K=100.json',
	),
	
	(# explorers
	),
	
	
	# Normal

	(# eps greedies
	),
	
	('UCB_Normal_K=100.json',
	 'UCB-Bernoulli_Normal_K=100.json',
	),
	
	('EXP3_Normal_K=100.json',
	),
	
	(# explorers
	),


	# Bernoulli

	(# eps greedies
	),
	
	('UCB_Bernoulli_K=100.json',
	 'UCB-Bernoulli_Bernoulli_K=100.json',
	),
	
	('EXP3_Bernoulli_K=100.json',
	),
	
	(# explorers
	),
	
	
	# Network Latencies

	(# eps greedies
	),
	
	('UCB_Network_Latencies_K=760.json',
	 'UCB-Bernoulli_Network_Latencies_K=760.json',
	),
	
	('EXP3_Network_Latencies_K=760.json',
	),
	
	(# explorers
	),
)

def plot_graph(fnames):
	if not fnames:
		return
	
	# Load all data
	datas = [simplejson.load(open("records/%s" % fname)) for fname in fnames]
	
	distro_name = datas[0][0]['distro_name']
	
	plt.title(distro_name)
	plt.ylabel("% Optimal Gain")
	plt.xlabel("Time Elapsed ($t$)")
	
	plots = []
	names = []
	plt.hold(True)
	for simu_params, run_data in datas:
		opt_arm_rwd, policy_rewards = run_data
		
		ys = numpy.cumsum(policy_rewards) / numpy.cumsum(opt_arm_rwd)
		ts = range(len(policy_rewards))

		p, = plt.plot(ts, ys)
		names.append(simu_params['name'])
		plots.append(p)
		
	plt.legend(plots, names)
	plt.hold(False)
	
def plot_all_graphs():
	f = plt.figure()
	for subplot_idx, fnames in enumerate(GRAPHS):
		plt.subplot(GRAPHS_ROWS, GRAPHS_COLS, subplot_idx + 1)
		plot_graph(fnames)
#		f.subplots_adjust(bottom=-0.2)
	
	plt.show()

if __name__ == '__main__':
	plot_all_graphs()
