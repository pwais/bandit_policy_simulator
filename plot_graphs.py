from __future__ import with_statement

import os

import matplotlib.pyplot as plt
import numpy
import simplejson

GRAPHS_ROWS = 4
GRAPHS_COLS = 4

GRAPHS = (
	
	# Uniform Distro

	('Epsilon_Greedy_eps=0.001_Uniform_eps=0.01_K=100.json',
	 'Epsilon_Greedy_eps=0.1_Uniform_eps=0.01_K=100.json',
	 'Epsilon-t_Greedy_c=5.0_eps=0.1_Uniform_eps=0.01_K=100.json',
	 'Epsilon-t_Greedy_c=1.0_eps=0.1_Uniform_eps=0.01_K=100.json',
	 'Epsilon-t_Greedy_c=0.1_eps=0.1_Uniform_eps=0.01_K=100.json',
	),
	
	('UCB_Uniform_eps=0.01_K=100.json',
	 'UCB-Normal_Uniform_eps=0.01_K=100.json',
	 'UCB-Bernoulli_Uniform_eps=0.01_K=100.json',
	 'UCB2_alpha=0.01_Uniform_eps=0.01_K=100.json',
	 'UCB2-Non-Sequential_alpha=0.01_Uniform_eps=0.01_K=100.json',
	 'Poker_Uniform_eps=0.01_K=100.json',
	),
	
	('EXP3_Uniform_eps=0.001_K=100.json',
	 'EXP3_gamma=0.01_Uniform_eps=0.001_K=100.json',
	 'SoftMix_Uniform_eps=0.001_K=100.json',
	),
	
	('Naive_Explorer_eps=0.01_delta=0.01_Uniform_eps=0.01_K=100.json',
	 'Successive_Elimination_delta=0.01_Uniform_eps=0.01_K=100.json',
	 'Successive_Elimination_with_Unknown_Biases_eps=0.01_delta=0.01_Uniform_eps=0.01_K=100.json',
	 'Median_Elimination_eps=0.01_delta=0.01_Uniform_eps=0.01_K=100.json',
	),
	
	
	# Normal

	('Epsilon_Greedy_eps=0.001_Normal_K=100.json',
	 'Epsilon_Greedy_eps=0.1_Normal_K=100.json',
	 'Epsilon-t_Greedy_c=5.0_eps=0.1_Normal_K=100.json',
	 'Epsilon-t_Greedy_c=1.0_eps=0.1_Normal_K=100.json',
	 'Epsilon-t_Greedy_c=0.1_eps=0.1_Normal_K=100.json',
	),
	
	('UCB_Normal_K=100.json',
	 'UCB-Normal_Normal_K=100.json',
	 'UCB-Bernoulli_Normal_K=100.json',
	 'UCB2_alpha=0.01_Normal_K=100.json',
	 'UCB2-Non-Sequential_alpha=0.01_Normal_K=100.json',
	 'Poker_Normal_K=100.json',
	),
	
	('EXP3_Normal_K=100.json',
	 'EXP3_gamma=0.01_Normal_K=100.json',
	 'SoftMix_Normal_K=100.json',
	),
	
	('Naive_Explorer_eps=0.01_delta=0.01_Normal_K=100.json',
	 'Successive_Elimination_delta=0.01_Normal_K=100.json',
	 'Successive_Elimination_with_Unknown_Biases_eps=0.01_delta=0.01_Normal_K=100.json',
	 'Median_Elimination_eps=0.01_delta=0.01_Normal_K=100.json',
	),


	# Bernoulli

	('Epsilon_Greedy_eps=0.001_Bernoulli_K=100.json',
	 'Epsilon_Greedy_eps=0.1_Bernoulli_K=100.json',
	 'Epsilon-t_Greedy_c=5.0_eps=0.1_Bernoulli_K=100.json',
	 'Epsilon-t_Greedy_c=1.0_eps=0.1_Bernoulli_K=100.json',
	 'Epsilon-t_Greedy_c=0.1_eps=0.1_Bernoulli_K=100.json',
	),
	
	('UCB_Bernoulli_K=100.json',
	 'UCB-Normal_Bernoulli_K=100.json',
	 'UCB-Bernoulli_Bernoulli_K=100.json',
	 'UCB2_alpha=0.01_Bernoulli_K=100.json',
	 'UCB2-Non-Sequential_alpha=0.01_Bernoulli_K=100.json',
	 'Poker_Bernoulli_K=100.json',
	),
	
	('EXP3_Bernoulli_K=100.json',
	 'EXP3_gamma=0.01_Bernoulli_K=100.json',
	 'SoftMix_Bernoulli_K=100.json',
	),
	
	('Naive_Explorer_eps=0.01_delta=0.01_Bernoulli_K=100.json',
	 'Successive_Elimination_delta=0.01_Bernoulli_K=100.json',
	 'Successive_Elimination_with_Unknown_Biases_eps=0.01_delta=0.01_Bernoulli_K=100.json',
	 'Median_Elimination_eps=0.01_delta=0.01_Bernoulli_K=100.json',
	),
	
	
	# Network Latencies

	('Epsilon_Greedy_eps=0.001_Network_Latencies_K=100.json',
	 'Epsilon_Greedy_eps=0.1_Network_Latencies_K=100.json',
	 'Epsilon-t_Greedy_c=5.0_eps=0.1_Network_Latencies_K=100.json',
	 'Epsilon-t_Greedy_c=1.0_eps=0.1_Network_Latencies_K=100.json',
	 'Epsilon-t_Greedy_c=0.1_eps=0.1_Network_Latencies_K=100.json',
	),
	
	('UCB_Network_Latencies_K=100.json',
	 'UCB-Normal_Network_Latencies_K=100.json',
	 'UCB-Bernoulli_Network_Latencies_K=100.json',
	 'UCB2_alpha=0.01_Network_Latencies_K=100.json',
	 'UCB2-Non-Sequential_alpha=0.01_Network_Latencies_K=100.json',
	 'Poker_Network_Latencies_K=100.json',
	),
	
	('EXP3_Network_Latencies_K=100.json',
	 'EXP3_gamma=0.01_Network_Latencies_K=100.json',
	 'SoftMix_Network_Latencies_K=100.json',
	),
	
	('Naive_Explorer_eps=0.01_delta=0.01_Network_Latencies_K=100.json',
	 'Successive_Elimination_delta=0.01_Network_Latencies_K=100.json',
	 'Successive_Elimination_with_Unknown_Biases_eps=0.01_delta=0.01_Network_Latencies_K=100.json',
	 'Median_Elimination_eps=0.01_delta=0.01_Network_Latencies_K=100.json',
	),
)

def plot_graph(fnames):
	if not fnames:
		return
	
	# Load all data
	datas = []
	for fname in fnames:
		try:
			datas.append(simplejson.load(open("records/%s" % fname)))
		except Exception:
			print "Probably bad file %s" % fname
	
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
