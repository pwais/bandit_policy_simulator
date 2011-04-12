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
			if t > self.max_time:
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


	