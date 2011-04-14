
import numpy.random

import math

numpy.random.seed()

def imax(xs):
	max_i = None
	max_v = None
	for i, v in enumerate(xs):
		if v > max_v:
			max_v = v
			max_i = i
	return max_i, max_v

def imin(xs):
	min_i = None
	min_v = None
	for i, v in enumerate(xs):
		if v < min_v or min_v is None:
			min_v = v
			min_i = i
	return min_i, min_v

class SampleStats(object):
	
	def __init__(self, num_arms):
		self.counts = [0] * num_arms
		self.sums = [0] * num_arms
		self.sum_squares = [0] * num_arms
	
	def update(self, arm, reward):
		self.sums[arm] += reward
		self.sum_squares[arm] += reward ** 2
		self.counts[arm] += 1
	
	def get_sample_means(self):
		return [float(sum) / count if count else 0 
			    for sum, count in zip(self.sums, self.counts)]
	
	def get_sample_variances(self):
		sample_means = self.get_sample_means()
		return [float(sum_sqs) / count - sample_mu ** 2 if count else 0
				for sum_sqs, count, sample_mu in zip(self.sum_squares, self.counts, sample_means)]

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

class SampleStatsPolicy(Policy):
	
	def __init__(self, num_arms=10, *args, **kwargs):
		super(SampleStatsPolicy, self).__init__(*args, **kwargs)
		self.reward_sample_stats = SampleStats(num_arms)
	
	def update(self, arm, reward):
		super(SampleStatsPolicy, self).update(arm, reward)
		self.reward_sample_stats.update(arm, reward)

class EpsGreedy(SampleStatsPolicy):
	
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
			sample_means = self.reward_sample_stats.get_sample_means()
			best_arm, _ = imax(sample_means)
			if best_arm is None:
				best_arm = 0
			
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
		
class UCB(SampleStatsPolicy):

	def choose_arm(self):
		# Play each arm exactly once
		if self.time < self.num_arms:
			return self.time
		
		sample_means = self.reward_sample_stats.get_sample_means()
		max_est_reward = None
		best_arm = None
		for i, sample_mean in enumerate(sample_means):
			cb = math.sqrt((2.0 * math.log(self.time)) / len(self.rewards[i]))
			est_reward = sample_mean + cb
			if est_reward > max_est_reward:
				max_est_reward = est_reward
				best_arm = i
		
		return best_arm

class UCBBernoulli(SampleStatsPolicy):
	"""Tuned for Bernoulli random variables"""

	def choose_arm(self):
		# Play each arm exactly once
		if self.time < self.num_arms:
			return self.time
		
		sample_vars = self.reward_sample_stats.get_sample_variances()
		sample_means = self.reward_sample_stats.get_sample_means()
		max_est_reward = None
		best_arm = None
		for i, (sample_var, sample_mean) in enumerate(zip(sample_vars, sample_means)):
			cb = math.sqrt((math.log(self.time) / len(self.rewards[i])) * 
						   min(1.0/4, sample_var))
			est_reward = sample_mean + cb
			if est_reward > max_est_reward:
				max_est_reward = est_reward
				best_arm = i
		
		return best_arm

class UCBNormal(SampleStatsPolicy):
	"""Tuned for Normal random variables"""

	def choose_arm(self):
		# Play each arm a minimum number of times
		arm, count = imin(self.reward_sample_stats.counts)
		if count < math.ceil(8.0 * math.log(self.time + 1)):
			return arm
		
		sample_means = self.reward_sample_stats.get_sample_means()
		max_est_reward = None
		best_arm = None
		iter_data = enumerate(zip(sample_means, 
								  self.reward_sample_stats.sum_squares,
								  self.reward_sample_stats.counts))
		for i, (sample_mean, sum_sq, count) in iter_data:
			cb = math.sqrt(16.0 * 
						   ((sum_sq - count * (sample_mean ** 2)) / (count - 1)) * 
						   (math.log(self.time - 1) / count))
			est_reward = sample_mean + cb
			if est_reward > max_est_reward:
				max_est_reward = est_reward
				best_arm = i
		
		return best_arm

### TODO UCB2, Poker, Exp3, softmax, intervalest...
