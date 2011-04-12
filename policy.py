
import numpy.random

import math

numpy.random.seed()

def mean(xs):
	if not xs:
		return 0.0
	else:
		return float(sum(xs)) / len(xs)

class Policy(object):
	
	def __init__(self, num_arms=10, *args, **kwargs):
		self.num_arms = num_arms
		self.rewards = [[]] * num_arms
		self.time = 0
	
	def choose_arm(self):
		pass
	
	def update(self, arm, reward):
		self.time += 1
		self.rewards[arm].append(reward)

class EpsGreedy(Policy):
	
	def __init__(self, eps=0.01, *args, **kwargs):
		super(EpsGreedy, self).__init__(*args, **kwargs)
		self.epsilon = eps
	
	def _get_eps(self):
		return self.epsilon
	
	def choose_arm(self):
		if numpy.random.random() < self._get_eps():
			# Explore
			return int(numpy.random.random() * self.num_arms)
		else:
			# Exploit
			max_sample_mean = mean(self.rewards[0])
			best_arm = 0
			for i in range(1, self.num_arms):
				sample_mean = mean(self.rewards[i])
				if sample_mean > max_sample_mean:
					best_arm = i
			
			return best_arm
	
class EpsTGreedy(EpsGreedy):
	
	def __init__(self, d, c=5.0, *args, **kwargs):
		super(EpsTGreedy, self).__init__(*args, **kwargs)
		self.c = c
		self.d = d
	
	def _get_eps(self):
		return min(1, 
				   (float(self.c) * self.num_arms) / 
				    ((self.d ** 2) (self.time + 1)))
		
class UCB(Policy):
	
	def choose_arm(self):
		# Play each arm exactly once
		if self.time < self.num_arms:
			return self.time
		
		max_est_reward = None
		best_arm = 0
		for i in range(self.num_arms):
			sample_mean = mean(self.rewards[i])
			cb = math.sqrt((2.0 * math.log(self.time)) / len(self.rewards[i]))
			est_reward = sample_mean + cb
			if est_reward > max_est_reward:
				max_est_reward = est_reward
				best_arm = i
		
		return best_arm



