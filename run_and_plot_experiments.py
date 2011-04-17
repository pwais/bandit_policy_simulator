import sys
from multiprocessing import Pool
from multiprocessing import Queue
#from threading import Thread

import numpy.random

import policy
from simulation import Simulation
import rewards

max_time = 2000

all_num_arms = (10, 25, 50, 100, 1000)
distro_eps = (0.1, 0.01, 0.001, 0.0001)

eps_greedy_epsilons = (0.5, 0.1, 0.01, 0.001, 0.0001)

eps_t_greedy_cs = (0.05, 0.1, 0.15, 0.2, 0.4, 1.0, 2.0, 5.0)

ucb2_alphas = (0.001, 0.01)

exp3_gammas = (0.1, 0.01, None)

exploration_first_eps = (0.1, 0.01, 0.001)
exploration_first_deltas = (0.1, 0.01, 0.001)
succ_elim_eps = exploration_first_eps + (None,)


#class ExpGreedyConstructor(object):
#	def __init__(self, num_arms=10, eps=0.1, *args, **kwargs):
#		super(ExpGreedySimu, self).__init__(*args, **kwargs)
#		self.num_arms = num_arms
#		self.eps = eps
#		self.name = "Epsilon Greedy eps=%s" % (num_arms, eps)
#		
#	def init(self):
#		p = policy.EpsGreedy(num_arms=self.num_arms, eps=self.eps)
#		rwds = synthetic.iter_uniform_plus_eps(self.num_arms)
#		return p, rwds

def iter_distro_params():
	for num_arms in all_num_arms:
		for eps in distro_eps:
			yield {
			 'reward_gen_name': 'iter_uniform_plus_eps',
			 'reward_gen_args': (num_arms,),
			 'reward_gen_kwargs': {'eps': eps},
			 'd': eps,
			 'mus': [0.5 + eps] + ([0.5] * (num_arms - 1))
			}
	
	for num_arms in all_num_arms:
		mus = numpy.random.uniform(low=0.0, high=1.0, size=num_arms)
		sigmas = numpy.random.uniform(low=0.0, high=1.0, size=num_arms)
		sorted_mus = sorted(mus, reverse=True)
		d = sorted_mus[0] - sorted_mus[1]
		yield {
			 'reward_gen_name': 'iter_normal',
			 'reward_gen_args': (num_arms, mus, sigmas),
			 'reward_gen_kwargs': {},
			 'd': d,
			 'mus': mus
			}
	
	for num_arms in all_num_arms:
		ps = numpy.random.uniform(low=0.0, high=1.0, size=num_arms)
		sorted_ps = sorted(ps, reverse=True)
		d = sorted_ps[0] - sorted_ps[1]
		yield {
			 'reward_gen_name': 'iter_bernoulli',
			 'reward_gen_args': (num_arms, ps),
			 'reward_gen_kwargs': {},
			 'd': d,
			 'mus': ps
			}
	
	for num_arms in (76, 760):
		sorted_mus = sorted(rewards.NETWORK_LATENCY_MUS[:num_arms])
		d = sorted_mus[0] - sorted_mus[1]
		yield {
		 'reward_gen_name': 'iter_network_latencies',
		 'reward_gen_args': (num_arms,),
		 'reward_gen_kwargs': {},
		 'd': d,
		 'mus': rewards.NETWORK_LATENCY_MUS[:num_arms]
		}
		

def iter_eps_greedy_sim_params(reward_gen_params):
	for num_arms in all_num_arms:
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
	for num_arms in all_num_arms:
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
	for num_arms in all_num_arms:
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
	for num_arms in all_num_arms:
		params = {
			'name': 'UCBBernoulli',
			'policy_class_name': 'UCBBernoulli',
			'policy_args': (),
			'policy_kwargs': {'num_arms': num_arms,
							  'max_time': max_time}
		}
		params.update(reward_gen_params)
		yield params

def iter_UCBNormal_sim_params(reward_gen_params):
	for num_arms in all_num_arms:
		params = {
			'name': 'UCBNormal',
			'policy_class_name': 'UCBNormal',
			'policy_args': (),
			'policy_kwargs': {'num_arms': num_arms,
							  'max_time': max_time}
		}
		params.update(reward_gen_params)
		yield params

def iter_UCB2SequentialEpochs_sim_params(reward_gen_params):
	for num_arms in all_num_arms:
		for alpha in ucb2_alphas:
			params = {
				'name': 'UCB2SequentialEpochs alpha=%s' % alpha,
				'policy_class_name': 'UCB2SequentialEpochs',
				'policy_args': (),
				'policy_kwargs': {'alpha': alpha,
								  'num_arms': num_arms,
								  'max_time': max_time}
			}
			params.update(reward_gen_params)
			yield params

def iter_UCB2NonSequentialEpochs_sim_params(reward_gen_params):
	for num_arms in all_num_arms:
		for alpha in ucb2_alphas:
			params = {
				'name': 'UCB2NonSequentialEpochs alpha=%s' % alpha,
				'policy_class_name': 'UCB2NonSequentialEpochs',
				'policy_args': (),
				'policy_kwargs': {'alpha': alpha,
								  'num_arms': num_arms,
								  'max_time': max_time}
			}
			params.update(reward_gen_params)
			yield params

def iter_Poker_sim_params(reward_gen_params):
	for num_arms in all_num_arms:
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
	for num_arms in all_num_arms:
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
	for num_arms in all_num_arms:
		for gamma in exp3_gammas:
			params = {
				'name': 'EXP3 gamma=%s' % gamma,
				'policy_class_name': 'EXP3',
				'policy_args': (),
				'policy_kwargs': {'gamma': gamma,
								  'num_arms': num_arms,
								  'max_time': max_time}
			}
			params.update(reward_gen_params)
			yield params

def iter_NaiveSequentialExplorer_sim_params(reward_gen_params):
	for num_arms in all_num_arms:
		for eps in exploration_first_eps:
			for delta in exploration_first_deltas:
				params = {
					'name': 'NaiveSequentialExplorer eps=%s delta=%s' % (eps, delta),
					'policy_class_name': 'NaiveSequentialExplorer',
					'policy_args': (eps, delta),
					'policy_kwargs': {'num_arms': num_arms,
									  'max_time': max_time}
				}
				params.update(reward_gen_params)
				yield params

def iter_SuccessiveEliminationSequentialExplorer_sim_params(reward_gen_params):
	for num_arms in all_num_arms:
		for delta in exploration_first_deltas:
			params = {
				'name': 'SuccessiveEliminationSequentialExplorer delta=%s' % (delta,),
				'policy_class_name': 'SuccessiveEliminationSequentialExplorer',
				'policy_args': (delta, reward_gen_params['mus']),
				'policy_kwargs': {'num_arms': num_arms,
								  'max_time': max_time}
			}
			params.update(reward_gen_params)
			yield params

def iter_SuccessiveEliminationUnknownBiasesUniformExplorer_sim_params(reward_gen_params):
	for num_arms in all_num_arms:
		for eps in succ_elim_eps:
			for delta in exploration_first_deltas:
				params = {
					'name': 'SuccessiveEliminationUnknownBiasesUniformExplorer eps=%s delta=%s' % (eps, delta),
					'policy_class_name': 'SuccessiveEliminationUnknownBiasesUniformExplorer',
					'policy_args': (delta, eps),
					'policy_kwargs': {'num_arms': num_arms,
									  'max_time': max_time}
				}
				params.update(reward_gen_params)
				yield params

policy_param_gens = (
	iter_eps_greedy_sim_params,
	iter_eps_t_greedy_sim_params,
	iter_UCB_sim_params,
	iter_UCBBernoulli_sim_params,
	iter_UCBNormal_sim_params,
	iter_UCB2SequentialEpochs_sim_params,
	iter_UCB2NonSequentialEpochs_sim_params,
	iter_Poker_sim_params,
	iter_SoftMix_sim_params,
	iter_EXP3_sim_params,
	iter_NaiveSequentialExplorer_sim_params,
	iter_SuccessiveEliminationSequentialExplorer_sim_params,
	iter_SuccessiveEliminationUnknownBiasesUniformExplorer_sim_params
)

def iter_all_experiment_params():
	for reward_gen_params in iter_distro_params():
		for policy_generator in policy_param_gens:
			for params in policy_generator(reward_gen_params):
				yield params

def run_simu(params):
	simu = Simulation(params, max_time=max_time)
	simu.run()
	simu.save()
	return simu


#def run_simu(sim):
#	sim.run(verbose=True)

##
## From http://bytes.com/topic/python/answers/552476-why-cant-you-pickle-instancemethods
## We may need the code below to make multiprocessing work correctly
##

#def _pickle_method(method):
#	func_name = method.im_func.__name__
#	obj = method.im_self
#	cls = method.im_class
#	return _unpickle_method, (func_name, obj, cls)
#
#def _unpickle_method(func_name, obj, cls):
#	for cls in cls.mro():
#		try:
#			func = cls.__dict__[func_name]
#		except KeyError:
#			pass
#		else:
#			break
#		return func.__get__(obj, cls)
#
#import copy_reg
#import types
#copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)

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
	all_simus = []
	if num_procs == 1:
		for params in iter_all_experiment_params():
			simu = run_simu(params)
			all_simus.append(simu)
	else:
		pool = Pool(processes=num_procs)
		for chunk in segment(iter_all_experiment_params(), num_procs):
			simus = pool.map(run_simu, chunk)
			all_simus.extend(simus)

#if __name__ == '__main__':
#	num_threads = 10
#	all_simus = []
#	for chunk in segment(iter_experiments(), num_threads):
#		workers = [FunctionWorker(run_simu, [sim]) for sim in chunk]
#		run_workers(workers)
#		all_simus.extend(chunk)
