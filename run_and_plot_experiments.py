import sys
from multiprocessing import Pool
from multiprocessing import Queue
#from threading import Thread

import policy
from simulation import Simulation
import synthetic

num_arms = (10, 25, 50, 100)
eps_greedy_epsilons = (0.5, 0.1, 0.01, 0.001, 0.0001)
max_time = 100000

class ExpGreedySimu(Simulation):
	def __init__(self, num_arms=10, eps=0.1, *args, **kwargs):
		super(ExpGreedySimu, self).__init__(*args, **kwargs)
		self.num_arms = num_arms
		self.eps = eps
		self.name = "Epsilon Greedy K=%s epsilon=%s" % (num_arms, eps)
		
	def init(self):
		p = policy.EpsGreedy(num_arms=self.num_arms, eps=self.eps)
		rwds = synthetic.iter_uniform_plus_eps(self.num_arms)
		return p, rwds

def iter_exp_greedy_simus():
	for arms in num_arms:
		for eps in eps_greedy_epsilons:
			sim = ExpGreedySimu(num_arms=arms, 
								eps=eps,
								max_time=max_time)
			yield sim


def iter_experiments():
	for sim in iter_exp_greedy_simus():
		yield sim

def segment(seq, n):
	chunk = []
	for s in seq:
		chunk.append(s)
		if len(chunk) == n:
			yield chunk
			chunk = []
	if chunk:
		yield chunk

def run_simu(sim):
	sim.run(verbose=True)

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

if __name__ == '__main__':
	num_procs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
	pool = Pool(processes=num_procs)
	all_simus = []
	for chunk in segment(iter_experiments(), num_procs):
		pool.map(run_simu, chunk)
		all_simus.extend(chunk)


#if __name__ == '__main__':
#	num_threads = 10
#	all_simus = []
#	for chunk in segment(iter_experiments(), num_threads):
#		workers = [FunctionWorker(run_simu, [sim]) for sim in chunk]
#		run_workers(workers)
#		all_simus.extend(chunk)
