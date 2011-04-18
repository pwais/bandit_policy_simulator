from __future__ import with_statement

import os

from matplotlib.font_manager import FontProperties
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
	
	('EXP3_Uniform_eps=0.01_K=100.json',
	 'EXP3_gamma=0.01_Uniform_eps=0.01_K=100.json',
	 'SoftMix_Uniform_eps=0.01_K=100.json',
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
	
	if not datas:
		return
	
	distro_name = datas[0][0]['distro_name']
	
	plt.title(distro_name)
	plt.ylabel("Regret")
	plt.xlabel("Time Elapsed (Time Steps)")
	
	plots = []
	names = []
	plt.hold(True)
	for simu_params, run_data in datas:
		opt_arm_rwd, policy_rewards = run_data
		
		ys = numpy.cumsum(opt_arm_rwd) - numpy.cumsum(policy_rewards)
		ts = range(len(policy_rewards))

		p, = plt.plot(ts, ys)
		#plt.yticks(numpy.arange(0.0, 1.4, 0.2))
		#plt.xticks(numpy.arange(0, len(ts), len(ts) / 10.0))
		
		# Simplify legend titles
		name = simu_params['name']
		name = name.replace('alpha', '$\\alpha$')
		name = name.replace('gamma', '$\\gamma$')
		name = name.replace('eps', '$\\epsilon$')
		name = name.replace('Epsilon-t', '$\\epsilon_t$')
		name = name.replace('Epsilon', '$\\epsilon$')
		name = name.replace('delta', '$\\delta$')
		name = name.replace('Naive Explorer', 'NE')
		name = name.replace('Successive Elimination', 'SE')
		name = name.replace(' with Unknown Biases', '-UB')
		name = name.replace('Median Elimination', 'ME')
		name = name.replace('-Non-Sequential', '-NS')
		name = name.replace('-Normal', '-N')
		name = name.replace('-Bernoulli', '-B')
		
		names.append(name)
		plots.append(p)
	
	fontP = FontProperties()
	fontP.set_size('small')
	plt.legend(plots, names, loc='upper left', prop=fontP, borderpad=0.1, labelspacing=0.2)
	plt.hold(False)
	
def plot_all_graphs():
	f = plt.figure()
	for subplot_idx, fnames in enumerate(GRAPHS):
		plt.subplot(GRAPHS_ROWS, GRAPHS_COLS, subplot_idx + 1)
		plot_graph(fnames)
		f.subplots_adjust(hspace=0.5)
	
	plt.show()

if __name__ == '__main__':
	plot_all_graphs()
