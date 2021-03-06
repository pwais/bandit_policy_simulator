import sys
from multiprocessing import Pool
from multiprocessing import Queue

import numpy.random

import policy
from simulation import Simulation
import rewards

max_time = 10000

#all_num_arms = (100,)
#distro_eps = (0.01,)
#
#eps_greedy_epsilons = (0.1, 0.001)
#
#eps_t_greedy_cs = (0.1, 1.0, 5.0)
#
#ucb2_alphas = (0.01,)
#
#exp3_gammas = (0.01, None)
#
#exploration_first_eps = (0.01,)
#exploration_first_deltas = (0.01,)
#succ_elim_eps = exploration_first_eps + (None,)


all_num_arms = (10, 100)
distro_eps = (0.1, 0.01, 0.001)

eps_greedy_epsilons = (0.5, 0.1, 0.01, 0.001)

eps_t_greedy_cs = (0.1, 0.2, 0.4, 1.0, 2.0, 5.0)

ucb2_alphas = (0.001, 0.01)

exp3_gammas = (0.1, 0.01, None)

exploration_first_eps = (0.1, 0.01, 0.001)
exploration_first_deltas = (0.1, 0.01, 0.001)
succ_elim_eps = exploration_first_eps + (None,)


def iter_distro_params():
	for num_arms in all_num_arms:
		for eps in distro_eps:
			yield {
			 'distro_name': "Uniform eps=%s K=%s" % (eps, num_arms),
			 'reward_gen_name': 'iter_uniform_plus_eps',
			 'reward_gen_args': (num_arms,),
			 'reward_gen_kwargs': {'eps': eps},
			 'd': eps,
			 'mus': [0.5 + eps] + ([0.5] * (num_arms - 1)),
			 'num_arms': num_arms,
			}
	
	for num_arms in all_num_arms:
		mus = list(numpy.random.uniform(low=0.0, high=1.0, size=num_arms))
		sigmas = list(numpy.random.uniform(low=0.0, high=1.0, size=num_arms))
		sorted_mus = sorted(mus, reverse=True)
		d = sorted_mus[0] - sorted_mus[1]
		yield {
			 'distro_name': "Normal K=%s" % num_arms,
			 'reward_gen_name': 'iter_normal',
			 'reward_gen_args': (num_arms, mus, sigmas),
			 'reward_gen_kwargs': {},
			 'd': d,
			 'mus': mus,
			 'num_arms': num_arms,
			}
	
	for num_arms in all_num_arms:
		ps = list(numpy.random.uniform(low=0.0, high=1.0, size=num_arms))
		sorted_ps = sorted(ps, reverse=True)
		d = sorted_ps[0] - sorted_ps[1]
		yield {
			 'distro_name': "Bernoulli K=%s" % num_arms,
			 'reward_gen_name': 'iter_bernoulli',
			 'reward_gen_args': (num_arms, ps),
			 'reward_gen_kwargs': {},
			 'd': d,
			 'mus': ps,
			 'num_arms': num_arms,
			}
	
	for num_arms in (100,):
		sorted_mus = sorted(rewards.NETWORK_LATENCY_MUS[:num_arms])
		d = sorted_mus[0] - sorted_mus[1]
		yield {
		 'distro_name': "Network Latencies K=%s" % num_arms,
		 'reward_gen_name': 'iter_network_latencies',
		 'reward_gen_args': (num_arms,),
		 'reward_gen_kwargs': {},
		 'd': d,
		 'mus': rewards.NETWORK_LATENCY_MUS[:num_arms],
		 'num_arms': num_arms,
		}
		

def iter_eps_greedy_sim_params(reward_gen_params):
	num_arms = reward_gen_params['num_arms']
	for eps in eps_greedy_epsilons:
		params = {
			'name': 'Epsilon Greedy eps=%s' % eps,
			'policy_class_name': 'EpsGreedy',
			'policy_args': (),
			'policy_kwargs': {'num_arms': num_arms,
							  'max_time': max_time}
		}
		params.update(reward_gen_params)
		yield params

def iter_eps_t_greedy_sim_params(reward_gen_params):
	num_arms = reward_gen_params['num_arms']
	for eps in eps_greedy_epsilons:
		for c in eps_t_greedy_cs:
			params = {
				'name': 'Epsilon-t Greedy c=%s eps=%s' % (c, eps),
				'policy_class_name': 'EpsTGreedy',
				'policy_args': (reward_gen_params['d'],),
				'policy_kwargs': {'c': c,
								  'num_arms': num_arms,
								  'max_time': max_time}
			}
			params.update(reward_gen_params)
			yield params

def iter_UCB_sim_params(reward_gen_params):
	num_arms = reward_gen_params['num_arms']
	params = {
		'name': 'UCB',
		'policy_class_name': 'UCB',
		'policy_args': (),
		'policy_kwargs': {'num_arms': num_arms,
						  'max_time': max_time}
	}
	params.update(reward_gen_params)
	yield params

def iter_UCBBernoulli_sim_params(reward_gen_params):
	num_arms = reward_gen_params['num_arms']
	params = {
		'name': 'UCB-Bernoulli',
		'policy_class_name': 'UCBBernoulli',
		'policy_args': (),
		'policy_kwargs': {'num_arms': num_arms,
						  'max_time': max_time}
	}
	params.update(reward_gen_params)
	yield params

def iter_UCBNormal_sim_params(reward_gen_params):
	num_arms = reward_gen_params['num_arms']
	params = {
		'name': 'UCB-Normal',
		'policy_class_name': 'UCBNormal',
		'policy_args': (),
		'policy_kwargs': {'num_arms': num_arms,
						  'max_time': max_time}
	}
	params.update(reward_gen_params)
	yield params

def iter_UCB2SequentialEpochs_sim_params(reward_gen_params):
	num_arms = reward_gen_params['num_arms']
	for alpha in ucb2_alphas:
		params = {
			'name': 'UCB2 alpha=%s' % alpha,
			'policy_class_name': 'UCB2SequentialEpochs',
			'policy_args': (),
			'policy_kwargs': {'alpha': alpha,
							  'num_arms': num_arms,
							  'max_time': max_time}
		}
		params.update(reward_gen_params)
		yield params

def iter_UCB2NonSequentialEpochs_sim_params(reward_gen_params):
	num_arms = reward_gen_params['num_arms']
	for alpha in ucb2_alphas:
		params = {
			'name': 'UCB2-Non-Sequential alpha=%s' % alpha,
			'policy_class_name': 'UCB2NonSequentialEpochs',
			'policy_args': (),
			'policy_kwargs': {'alpha': alpha,
							  'num_arms': num_arms,
							  'max_time': max_time}
		}
		params.update(reward_gen_params)
		yield params

def iter_Poker_sim_params(reward_gen_params):
	num_arms = reward_gen_params['num_arms']
	params = {
		'name': 'Poker',
		'policy_class_name': 'Poker',
		'policy_args': (),
		'policy_kwargs': {'num_arms': num_arms,
						  'max_time': max_time}
	}
	params.update(reward_gen_params)
	yield params

def iter_SoftMix_sim_params(reward_gen_params):
	num_arms = reward_gen_params['num_arms']
	params = {
		'name': 'SoftMix',
		'policy_class_name': 'SoftMix',
		'policy_args': (reward_gen_params['d'],),
		'policy_kwargs': {'num_arms': num_arms,
						  'max_time': max_time}
	}
	params.update(reward_gen_params)
	yield params

def iter_EXP3_sim_params(reward_gen_params):
	num_arms = reward_gen_params['num_arms']
	for gamma in exp3_gammas:
		params = {
			'name': 'EXP3 gamma=%s' % gamma if gamma is not None else 'EXP3',
			'policy_class_name': 'EXP3',
			'policy_args': (),
			'policy_kwargs': {'gamma': gamma,
							  'num_arms': num_arms,
							  'max_time': max_time}
		}
		params.update(reward_gen_params)
		yield params

def iter_NaiveSequentialExplorer_sim_params(reward_gen_params):
	num_arms = reward_gen_params['num_arms']
	for eps in exploration_first_eps:
		for delta in exploration_first_deltas:
			params = {
				'name': 'Naive Explorer eps=%s delta=%s' % (eps, delta),
				'policy_class_name': 'NaiveSequentialExplorer',
				'policy_args': (eps, delta),
				'policy_kwargs': {'num_arms': num_arms,
								  'max_time': max_time}
			}
			params.update(reward_gen_params)
			yield params

def iter_SuccessiveEliminationSequentialExplorer_sim_params(reward_gen_params):
	num_arms = reward_gen_params['num_arms']
	for delta in exploration_first_deltas:
		params = {
			'name': 'Successive Elimination delta=%s' % (delta,),
			'policy_class_name': 'SuccessiveEliminationSequentialExplorer',
			'policy_args': (delta, reward_gen_params['mus']),
			'policy_kwargs': {'num_arms': num_arms,
							  'max_time': max_time}
		}
		params.update(reward_gen_params)
		yield params

def iter_SuccessiveEliminationUnknownBiasesUniformExplorer_sim_params(reward_gen_params):
	num_arms = reward_gen_params['num_arms']
	for eps in succ_elim_eps:
		for delta in exploration_first_deltas:
			params = {
				'name': 'Successive Elimination with Unknown Biases eps=%s delta=%s' % (eps, delta)
						if eps is not None else 'Successive Elimination with Unknown Biases delta=%s' % (delta,),
				'policy_class_name': 'SuccessiveEliminationUnknownBiasesUniformExplorer',
				'policy_args': (delta,),
				'policy_kwargs': {'epsilon': eps,
								  'num_arms': num_arms,
								  'max_time': max_time}
			}
			params.update(reward_gen_params)
			yield params

def iter_MedianEliminationSequentialExplorer_sim_params(reward_gen_params):
	num_arms = reward_gen_params['num_arms']
	for eps in exploration_first_eps:
		for delta in exploration_first_deltas:
			params = {
				'name': 'Median Elimination eps=%s delta=%s' % (eps, delta),
				'policy_class_name': 'MedianEliminationSequentialExplorer',
				'policy_args': (eps, delta),
				'policy_kwargs': {'num_arms': num_arms,
								  'max_time': max_time}
			}
			params.update(reward_gen_params)
			yield params

fam_1 = (
	iter_eps_greedy_sim_params,
	iter_eps_t_greedy_sim_params,
)

fam_2 = (
	iter_UCB_sim_params,
	iter_UCBBernoulli_sim_params,
	iter_UCBNormal_sim_params,
	iter_UCB2SequentialEpochs_sim_params,
	iter_UCB2NonSequentialEpochs_sim_params,
	iter_Poker_sim_params,
)

fam_3 = (
	iter_SoftMix_sim_params,
	iter_EXP3_sim_params,
)

fam_4 = (
	iter_NaiveSequentialExplorer_sim_params,
	iter_SuccessiveEliminationSequentialExplorer_sim_params,
	iter_SuccessiveEliminationUnknownBiasesUniformExplorer_sim_params,
	iter_MedianEliminationSequentialExplorer_sim_params,
)

policy_param_gens = fam_1 + fam_2 + fam_3 + fam_4

def iter_all_experiment_params():
	for reward_gen_params in iter_distro_params():
		for policy_generator in policy_param_gens:
			for params in policy_generator(reward_gen_params):
				yield params

def iter_family(fam):
	def param_gen():
		for reward_gen_params in iter_distro_params():
			for policy_generator in fam:
				for params in policy_generator(reward_gen_params):
					yield params
	return param_gen


def run_simu(params):
	simu = Simulation(params, max_time=max_time)
	simu.run()
	simu.save()
	return simu

def get_experiments(simu_family):
	if simu_family == 0:
		return iter_all_experiment_params
	elif simu_family == 1:
		return iter_family(fam_1)
	elif simu_family == 2:
		return iter_family(fam_2)
	elif simu_family == 3:
		return iter_family(fam_3)
	elif simu_family == 4:
		return iter_family(fam_4)

def segment(seq, n):
	chunk = []
	for s in seq:
		chunk.append(s)
		if len(chunk) == n:
			yield chunk
			chunk = []
	if chunk:
		yield chunk

if __name__ == '__main__':
	num_procs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
	simu_family = int(sys.argv[2]) if len(sys.argv) > 2 else 0
	
	iter_experiment_params = get_experiments(simu_family)
	
	if num_procs == 1:
		for params in iter_experiment_params():
			simu = run_simu(params)
	else:
		pool = Pool(processes=num_procs)
		for chunk in segment(iter_experiment_params(), num_procs):
			simus = pool.map(run_simu, chunk)

