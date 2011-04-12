import sys

class Simulation(object):
	
	def __init__(self, 
				 policy, 
				 iter_rewards,
				 max_time):
		self.policy = policy
		self.iter_rewards = iter_rewards
		self.max_time = max_time
		
		# Pre-allocate to signficantly increase performance
		self.rewards = [[]] * max_time
		self.policy_rewards = [0] * max_time
	
	def run(self, verbose=True, report_interval=1000):
		for t, rewards_t in enumerate(self.iter_rewards):
			if t > self.max_time:
				break
			
			self.rewards[t] = rewards_t
			
			arm = self.policy.choose_arm()
			
			reward = rewards_t[arm]
			self.policy_rewards[t] = reward
			self.policy.update(arm, reward)
			
			if verbose and t % report_interval == 0:
				print >>sys.stderr, "Ran %s time steps" % t
	
	def regret_vs_arm(self, arm):
		arm_gain = sum(rewards[arm] for rewards in self.rewards)
		algo_gain = sum(rwd for rwd in self.policy_rewards)
		return arm_gain - algo_gain

			