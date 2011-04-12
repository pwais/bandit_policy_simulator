from threading import Thread

import policy
import simulation
import synthetic

def iter_experiments():
	
	num_arms = 10
	p = policy.EpsGreedy(num_arms)
	rwds = synthetic.iter_uniform_plus_eps(num_arms)
	sim = simulation.Simulation(p, rwds, 100000)
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

class FunctionWorker(Thread):
	def __init__(self, func, args):
		super(FunctionWorker, self).__init__()
		self.func = func
		self.args = args
		
		# This thread should not block the main thread
		self.setDaemon(True)
		
		self.__result = None

	def run(self):
		self.__result = self.func(*self.args)
	
	def result(self):
		return self.__result

def run_workers(workers):
	for w in workers:
		w.start()
	for w in workers:
		w.join()

##
## From http://bytes.com/topic/python/answers/552476-why-cant-you-pickle-instancemethods
##
#
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
	num_threads = 10
	all_simus = []
	for chunk in segment(iter_experiments(), num_threads):
		workers = [FunctionWorker(run_simu, [sim]) for sim in chunk]
		run_workers(workers)
		all_simus.extend(chunk)
