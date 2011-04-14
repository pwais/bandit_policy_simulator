import sys

class Simulation(object):
	
	def __init__(self, max_time=100, name=None):
		self.max_time = max_time
		self.name = name
		
		# Pre-allocate to significantly increase performance
		self.rewards = [[]] * max_time
		self.policy_rewards = [0] * max_time
	
	def init(self):
		pass
	
	def run(self, verbose=True, report_interval=1000):
		policy, iter_rewards = self.init()
		
		for t, rewards_t in enumerate(iter_rewards):
			if t >= self.max_time:
				break
			
			self.rewards[t] = rewards_t
			
			arm = policy.choose_arm()
			
			reward = rewards_t[arm]
			self.policy_rewards[t] = reward
			policy.update(arm, reward)
			
			if verbose and t % report_interval == 0:
				print >>sys.stderr, "%s: ran %s time steps" % (self.name, t)
	
	def regret_vs_arm(self, arm):
		arm_gain = sum(rewards[arm] for rewards in self.rewards)
		algo_gain = sum(rwd for rwd in self.policy_rewards)
		return arm_gain - algo_gain

class NSimulations(object):
	
	def __init__(self, max_time=100, name=None, num_sims=100):
		self.name = name
		
		self.simus = [Simulation(max_time=max_time, name=None)
					  for _ in range(num_sims)]
	
	def run(self, verbose=True, report_interval=1000):
		for i, simu in enumerate(self.simus):
			print >>sys.stderr, "%s: running simulation %s" % (self.name, i)
			simu.run(verbose=verbose, report_interval=report_interval)