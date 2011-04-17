import math
import random

import numpy
import numpy.random
import scipy.stats

numpy.random.seed()

# BEGIN
# From: 
#  http://stackoverflow.com/questions/457408/is-there-an-easily-available-implementation-of-erf-for-python
# which was adapted from:
#  http://rads.stackoverflow.com/amzn/click/0486612724
def erf(x):
	# save the sign of x
	sign = 1
	if x < 0: 
		sign = -1
	x = abs(x)

	# constants
	a1 =  0.254829592
	a2 = -0.284496736
	a3 =  1.421413741
	a4 = -1.453152027
	a5 =  1.061405429
	p  =  0.3275911

	# A&S formula 7.1.26
	t = 1.0/(1.0 + p*x)
	y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*math.exp(-x*x)
	return sign*y # erf(-x) = -erf(x)
# END 

def q_func(x, mu, sigma):
	"""Tail probability for a normal distribution N(mu, sigma).  See
	# See http://en.wikipedia.org/wiki/Q-function """
	return 0.5 - 0.5 * erf(float(x - mu) / (sigma * math.sqrt(2)))

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

def weighted_choice(probs):
	sample_p = list(probs)
	sum_p = sum(sample_p)
	if sum_p == 0:
		return 0 # Choose arbitrarily
	elif sum_p != 1.0:
		sample_p = [p / sum_p for p in sample_p]
	
	rand = numpy.random.random()
	cumsum = 0.0
	for i, p in enumerate(sample_p):
		cumsum += p
		if rand < cumsum:
			return i





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
	
	def __init__(self, num_arms=10, max_time=10**5, *args, **kwargs):
		self.num_arms = num_arms
		self.rewards = [[]] * self.num_arms
		self.time = 0
		self.max_time = max_time
	
	def choose_arm(self):
		pass
	
	def update(self, arm, reward):
		self.time += 1
		self.rewards[arm].append(reward)

class SampleStatsPolicy(Policy):
	
	def __init__(self, *args, **kwargs):
		super(SampleStatsPolicy, self).__init__(*args, **kwargs)
		self.sample_stats = SampleStats(self.num_arms)
	
	def update(self, arm, reward):
		super(SampleStatsPolicy, self).update(arm, reward)
		self.sample_stats.update(arm, reward)







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
			sample_means = self.sample_stats.get_sample_means()
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
		return min(1, (float(self.c) * self.num_arms) / ((self.d ** 2) * (self.time + 1)))
		
class UCB(SampleStatsPolicy):

	def choose_arm(self):
		# Play each arm exactly once
		if self.time < self.num_arms:
			return self.time
		
		sample_means = self.sample_stats.get_sample_means()
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
		
		sample_vars = self.sample_stats.get_sample_variances()
		sample_means = self.sample_stats.get_sample_means()
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
		arm, count = imin(self.sample_stats.counts)
		
		# Note that we will make the policy play each arm at least once
		if count < math.ceil(8.0 * math.log(self.time + 2)):
			return arm
		
		sample_means = self.sample_stats.get_sample_means()
		max_est_reward = None
		best_arm = None
		iter_data = enumerate(zip(sample_means, 
								  self.sample_stats.sum_squares,
								  self.sample_stats.counts))
		for i, (sample_mean, sum_sq, count) in iter_data:
			cb = math.sqrt(16.0 * 
						   ((sum_sq - count * (sample_mean ** 2)) / (count - 1)) * 
						   (math.log(self.time - 1) / count))
			est_reward = sample_mean + cb
			if est_reward > max_est_reward:
				max_est_reward = est_reward
				best_arm = i
		
		return best_arm

class UCB2SequentialEpochs(SampleStatsPolicy):

	def __init__(self, alpha=0.001, *args, **kwargs):
		super(UCB2SequentialEpochs, self).__init__(*args, **kwargs)
		self.alpha = alpha
		self.r_i = [0] * self.num_arms
		self.in_epoch = False
		self.current_best_arm = None
		self.epoch_counter = -1
		self.epoch_length = 0

	def tao(self, r):
		return math.ceil((1 + self.alpha)**r)

	def conf_bound(self, t, r):
		tao_r = self.tao(r)
		return math.sqrt(((1 + self.alpha) * 
						  math.log(math.e * (t / tao_r))) / 
						 (2.0 * tao_r))

	def _choose_best_arm(self):
		sample_means = self.sample_stats.get_sample_means()
		max_est_reward = None
		best_arm = None
		iter_data = enumerate(zip(sample_means, 
								  self.r_i))
		for i, (sample_mean, r) in iter_data:
			est_reward = sample_mean + self.conf_bound(self.time + 1, r)
			if est_reward > max_est_reward:
				max_est_reward = est_reward
				best_arm = i
		
		return best_arm

	def choose_arm(self):
		# Play each arm exactly once
		if self.time < self.num_arms:
			return self.time
		
		if self.in_epoch and self.epoch_counter < self.epoch_length:
			self.epoch_counter += 1
			return self.current_best_arm
		elif self.epoch_counter == self.epoch_length:
			self.r_i[self.current_best_arm] += 1
		
		self.current_best_arm = self._choose_best_arm()
		self.epoch_length = (self.tao(self.r_i[self.current_best_arm] + 1) -
							 self.tao(self.r_i[self.current_best_arm]))
		self.epoch_counter = 0
		
		return self.current_best_arm
		
class UCB2NonSequentialEpochs(SampleStatsPolicy):
	# TODO refactor out common UCB2 functionality

	def __init__(self, alpha=0.001, *args, **kwargs):
		super(UCB2NonSequentialEpochs, self).__init__(*args, **kwargs)
		self.alpha = alpha
		self.r_i = [0] * self.num_arms
		self.epoch_counters = [-1] * self.num_arms
		self.epoch_lengths = [0] * self.num_arms

	def tao(self, r):
		return math.ceil((1 + self.alpha)**r)

	def conf_bound(self, t, r):
		tao_r = self.tao(r)
		return math.sqrt(((1 + self.alpha) * 
						  math.log(math.e * (t / tao_r))) / 
						 (2.0 * tao_r))

	def _choose_best_arm(self):
		sample_means = self.sample_stats.get_sample_means()
		max_est_reward = None
		best_arm = None
		iter_data = enumerate(zip(sample_means, 
								  self.r_i))
		for i, (sample_mean, r) in iter_data:
			est_reward = sample_mean + self.conf_bound(self.time + 1, r)
			if est_reward > max_est_reward:
				max_est_reward = est_reward
				best_arm = i
		
		return best_arm

	def choose_arm(self):
		# Play each arm exactly once
		if self.time < self.num_arms:
			return self.time
		
		best_arm = self._choose_best_arm()
		
		if self.epoch_counters[best_arm] < self.epoch_lengths[best_arm]:
			self.epoch_counters[best_arm] += 1
			return best_arm
		elif self.epoch_counters[best_arm] == self.epoch_lengths[best_arm]:
			self.r_i[best_arm] += 1
		
		# Find the new best arm and start a new epoch for that arm
		best_arm = self._choose_best_arm()
		self.epoch_lengths[best_arm] = (self.tao(self.r_i[best_arm] + 1) -
							 			self.tao(self.r_i[best_arm]))
		self.epoch_counters[best_arm] = 0
		
		return best_arm

class Poker(SampleStatsPolicy):
	
	def __init__(self, horizon=None, *args, **kwargs):
		super(Poker, self).__init__(*args, **kwargs)
		self.horizon = horizon
		self.all_rewards = []
	
	def update(self, arm, reward):
		super(SampleStatsPolicy, self).update(arm, reward)
		self.all_rewards.append(reward)
	
	def choose_arm(self):
		q = sum(1 for sums in self.sample_stats.sums if sums > 0) or 1
		sample_means = self.sample_stats.get_sample_means()
		sample_vars = self.sample_stats.get_sample_variances()
		_, best_mean = imax(sample_means)
		sorted_means = sorted(sample_means)
		
		delta_mu = float(best_mean - sorted_means[int(math.sqrt(q))]) / math.sqrt(q)
		
		if self.horizon:
			horizon = self.horizon
		else:
			horizon = self.max_time - self.time
		
		best_arm = None
		max_p = None
		for i in range(self.num_arms):
			if self.sample_stats.counts[i] > 0:
				mu = sample_means[i]
			elif len(self.all_rewards):
				mu = sum(self.all_rewards) / len(self.all_rewards)
			else:
				mu = 0
			
			if self.sample_stats.counts[i] > 1:
				sigma = math.sqrt(sample_vars[i])
			else:
				sigma = numpy.std(self.all_rewards)
			
			p = mu + delta_mu * horizon * q_func(best_mean + delta_mu, mu, sigma)
			if p > max_p:
				p = max_p
				best_arm = i
		
		return best_arm

class SoftMix(SampleStatsPolicy):
	
	def __init__(self, d, *args, **kwargs):
		super(SoftMix, self).__init__(*args, **kwargs)
		self.d = d
		self.reward_sums = [0] * self.num_arms
		self.last_p = None
	
	def update(self, arm, reward):
		super(SoftMix, self).update(arm, reward)
		self.reward_sums[arm] += float(reward) / self.last_p
	
	def choose_arm(self):
		if self.time > 2:
			gamma_t = min(1.0, ((5 * self.num_arms * math.log(self.time - 1)) /
							    ((self.d ** 2) * (self.time - 1))))
		else:
			gamma_t = 1.0
		
		K_div_gamma = float(self.num_arms) / gamma_t 
		temp = ((1.0 / (K_div_gamma + 1)) * 
				  math.log(1 + ((self.d * (K_div_gamma + 1)) /
				  				(2.0 * K_div_gamma - self.d ** 2))))
		
		exp_terms = [math.exp(r_sum * temp) for r_sum in self.reward_sums]
		sum_exp_terms = sum(exp_terms)
		probs = [(1.0 - gamma_t) * (exp_term / sum_exp_terms) + (gamma_t / self.num_arms)
				 for exp_term in exp_terms]
		arm = weighted_choice(probs)
		self.last_p = probs[arm]
		
		return arm

class EXP3(SampleStatsPolicy):
	
	def __init__(self, gamma=None, max_time=10**5, *args, **kwargs):
		super(EXP3, self).__init__(*args, **kwargs)
		self.weights = [1] * self.num_arms
		self.probs = [0] * self.num_arms
		if gamma is None:
			self.gamma = min(1, math.sqrt((self.num_arms * math.log(self.num_arms)) / 
									 	  ((math.e - 1) * max_time)))
		else:
			self.gamma = gamma
	
	def update(self, arm, reward):
		super(EXP3, self).update(arm, reward)
		
		x_hat = float(reward) / self.probs[arm]
		self.weights[arm] *= math.exp((self.gamma * x_hat) / self.num_arms)
	
	def choose_arm(self):
		sum_weights = sum(self.weights)
		self.probs = [((1 - self.gamma) * (weight / sum_weights) + self.gamma / self.num_arms)
					  for weight in self.weights]
		
		arm = weighted_choice(self.probs)
		return arm

class NaiveSequentialExplorer(SampleStatsPolicy):
	
	def __init__(self, epsilon, delta, *args, **kwargs):
		super(NaiveSequentialExplorer, self).__init__(*args, **kwargs)
		self.epsilon = epsilon
		self.delta = delta
		self.arm_count = [0] * self.num_arms
		self.current_arm = 0
		
		self.sample_count = (4.0 / epsilon ** 2) * math.log((2.0 * self.num_arms) / delta)
		
		self.best_arm = None
	
	def choose_arm(self):
		if self.arm_count[self.current_arm] < self.sample_count:
			self.arm_count[self.current_arm] += 1
			return self.current_arm
		else:
			self.current_arm += 1
		
		if self.current_arm > self.num_arms:
			if not self.best_arm:
				self.best_arm, _ = imax(self.sample_stats.get_sample_means())
			return self.best_arm
		
class SuccessiveEliminationSequentialExplorer(SampleStatsPolicy):
	
	def __init__(self, delta, mus, *args, **kwargs):
		super(SuccessiveEliminationSequentialExplorer, self).__init__(*args, **kwargs)
		self.delta = delta
		self.active_arms = [True] * self.num_arms
		
		self.t_is = [((8.0 / ((mus[0] - mu) ** 2)) * math.log(2.0*self.num_arms / delta))
					 if mus[0] - mu else 0
					 for mu in mus]
		
		# t_{i+1} = 0 by definition
		self.t_is.append(0)
		
		self.best_arm = None
		
		self.arm_chooser = self.make_arm_chooser()
	
	def make_arm_chooser(self):
		for i in range(self.num_arms - 1):
			times_to_sample = self.t_is[self.num_arms - 1 - i] - self.t_is[self.num_arms - i]
			for i in range(self.num_arms):
				if self.active_arms[i]:
					for _ in range(int(times_to_sample)):
						yield i
			
			worst_arm, _ = imin(self.sample_stats.get_sample_means())
			self.active_arms[worst_arm] = False
	
	def choose_arm(self):
		try:
			arm = self.arm_chooser.next()
			return arm
		except StopIteration:
			if not self.best_arm:
				for i, active in enumerate(self.active_arms):
					if active:
						self.best_arm = i
						break
			
			return self.best_arm

class SuccessiveEliminationUnknownBiasesUniformExplorer(SampleStatsPolicy):
	"""Note that setting epsilon makes this an (eps,delta)-PAC algo"""
	
	def __init__(self, delta, epsilon=None, c=5, *args, **kwargs):
		super(SuccessiveEliminationUnknownBiasesUniformExplorer, self).__init__(*args, **kwargs)
		self.delta = delta
		self.epsilon = epsilon
		self.c = c
		self.active_arms = [True] * self.num_arms
		self.best_arm = None
		self.arm_chooser = self.make_arm_chooser()
	
	def stop_sampling(self):
		num_active_arms = sum(1.0 for active in self.active_arms if active)
		if self.epsilon:
			min_num_samples = ((1.0 / (self.epsilon ** 2)) * 
							   math.log(num_active_arms / self.delta))
			arms_need_more_sampling = [i for i, count in enumerate(self.sample_stats.counts)
									     if count < min_num_samples]
			return len(arms_need_more_sampling) == 0
		else:
			return num_active_arms == 1
	
	def make_arm_chooser(self):
		while not self.stop_sampling():
			sample_means = self.sample_stats.get_sample_means()
			_, best_mean = imax(sample_means)
			alpha_t = math.sqrt(
						math.log((self.c * self.num_arms * (self.time ** 2)) / self.delta) /
						self.time)
			for i in range(self.num_arms):
				if best_mean - sample_means[i] >= 2 * alpha_t:
					self.active_arms[i] = False
			
			# Choose an arm uniformly; this is not necessarily part of Succesive Elimination
			# with Unknown Biases
			active_arms = [i for i, active in enumerate(self.active_arms) if active]
			yield random.choice(active_arms)
	
	def choose_arm(self):
		if self.time < self.num_arms:
			return self.time
		
		try:
			arm = self.arm_chooser.next()
			return arm
		except StopIteration:
			if not self.best_arm:
				for i, active in enumerate(self.active_arms):
					if active:
						self.best_arm = i
						break
			
			return self.best_arm

class MedianEliminationSequentialExplorer(SampleStatsPolicy):
	
	def __init__(self, epsilon, delta, c=5, *args, **kwargs):
		super(MedianEliminationSequentialExplorer, self).__init__(*args, **kwargs)
		self.epsilon = epsilon
		self.delta = delta
		self.c = c
		self.active_arms = [True] * self.num_arms
		self.best_arm = None
		self.arm_chooser = self.make_arm_chooser()
	
	def update(self, arm, reward):
		super(MedianEliminationSequentialExplorer, self).update(arm, reward)
		self.epoch_rewards[arm].append(reward)
	
	def make_arm_chooser(self):
		
		eps_current = self.epsilon / 4.0
		delta_current = self.delta / 2.0
		
		while sum(1 for active in self.active_arms if active) > 1:
			self.epoch_rewards = [[]] * self.num_arms
			
			times_to_sample = (1.0 / ((eps_current / 2.0) ** 2)) * math.log(3.0 / delta_current)
			for i in range(self.num_arms):
				if self.active_arms[i]:
					for _ in range(times_to_sample):
						yield i
			
			epoch_means = [sum(rwds) / len(rwds) for rwds in self.epoch_rewards]
			median_rwd = scipy.stats.scoreatpercentile(epoch_means, 50)
			
			for i in range(self.num_arms):
				if epoch_means[i] < median_rwd:
					self.active_arms[i] = False
			
			eps_current *= (3.0 / 4)
			delta_current *= 0.5
	
	def choose_arm(self):
		try:
			arm = self.arm_chooser.next()
			return arm
		except StopIteration:
			if not self.best_arm:
				for i, active in enumerate(self.active_arms):
					if active:
						self.best_arm = i
						break
			
			return self.best_arm

