
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

class SampleMeans(object):
	
	def __init__(self, num_means):
		self.counts = [0] * num_means
		self.sums = [0] * num_means
	
	def update(self, arm, reward):
		self.sums[arm] += reward
		self.counts[arm] += 1
	
	def get_sample_means(self):
		return [float(sum) / count if count else 0 
			    for sum, count in zip(self.sums, self.counts)]

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

class SampleMeanPolicy(Policy):
	
	def __init__(self, num_arms=10, *args, **kwargs):
		super(SampleMeanPolicy, self).__init__(*args, **kwargs)
		self.reward_sample_means = SampleMeans(num_arms)
	
	def update(self, arm, reward):
		super(SampleMeanPolicy, self).update(arm, reward)
		self.reward_sample_means.update(arm, reward)

class EpsGreedy(SampleMeanPolicy):
	
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
			sample_means = self.reward_sample_means.get_sample_means()
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
		
class UCB(SampleMeanPolicy):

	def choose_arm(self):
		# Play each arm exactly once
		if self.time < self.num_arms:
			return self.time
		
		sample_means = self.reward_sample_means.get_sample_means()
		max_est_reward = None
		best_arm = None
		for i, sample_mean in enumerate(sample_means):
			cb = math.sqrt((2.0 * math.log(self.time)) / len(self.rewards[i]))
			est_reward = sample_mean + cb
			if est_reward > max_est_reward:
				max_est_reward = est_reward
				best_arm = i
		
		return best_arm

