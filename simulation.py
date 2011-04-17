import os
import sys

import simplejson

import policy
from policy import imax
import rewards

#class Rewards(object):
#	
#	def __init__(self):
#		self.sums = []
#		self.counts = []
#	
#	def update(self, rwd):
#		self.sums += rwd
#		self.counts = []

def construct_policy(simu_params):
	policy_class_name = simu_params['policy_class_name']
	policy_args = simu_params['policy_args']
	policy_kwargs = simu_params['policy_kwargs']
	
	policy_class = getattr(policy, policy_class_name)
	return policy_class(*policy_args, **policy_kwargs)

def construct_iter_rewards(simu_params):
	reward_gen_name = simu_params['reward_gen_name']
	reward_gen_args = simu_params['reward_gen_args']
	reward_gen_kwargs = simu_params['reward_gen_kwargs']
	
	reward_generator = getattr(rewards, reward_gen_name)
	return reward_generator(*reward_gen_args, **reward_gen_kwargs)

class Simulation(object):
	
	def __init__(self, simu_params, max_time=100, num_sims=1):
		self.max_time = max_time
		self.num_sims = num_sims
		self.simu_params = simu_params
		
		self.run_to_rewards = []
		self.run_to_policy_rewards = []
	
	def init(self):
		pass
	
	def _run_once(self, verbose=True):
		
		policy = construct_policy(self.simu_params)
		iter_rewards = construct_iter_rewards(self.simu_params)
		
		# Pre-allocate to significantly increase performance
		rewards = [[]] * self.max_time
		policy_rewards = [0] * self.max_time
		reward_sums = [0] * policy.num_arms
		arm_choices = []
		
		for t, rewards_t in enumerate(iter_rewards):
			if t >= self.max_time:
				break
			
			rewards[t] = rewards_t
			
			arm = policy.choose_arm()
			arm_choices.append(arm)
			
			reward = rewards_t[arm]
			policy_rewards[t] = reward
			policy.update(arm, reward)
			
			for i, r in enumerate(rewards_t):
				reward_sums[i] += r
			
			if verbose and t % 1000 == 0:
				print >>sys.stderr, "%s: ran %s time steps" % (self.simu_params['name'], t)
		
		return rewards, reward_sums, policy_rewards, arm_choices
	
	def run(self, verbose=True):
		opt_arm_s = [[]] * self.num_sims
		opt_arm_rwd_s = [[]] * self.num_sims
		policy_rewards_s = [[]] * self.num_sims
		
		for s in range(self.num_sims):
			rewards, reward_sums, policy_rewards, arm_choices = self._run_once(verbose=verbose)
			
			best_arm, best_sum = imax(reward_sums)
			
			opt_arm_s[s] = [chosen_arm == best_arm for chosen_arm in arm_choices]
			opt_arm_rwd_s[s] = [r[best_arm] for r in rewards]
			policy_rewards_s[s] = policy_rewards
		
		self.run_data = (opt_arm_s, opt_arm_rwd_s, policy_rewards_s)

	def save(self):
		data = [self.simu_params, self.run_data]
#		simplejson.dump(data, open(self.simu_params['name'], 'w'))
			


#class NSimulations(object):
#	
#	def __init__(self, num_sims=100, simu_class):
#		self.name = name
#		self.simus = [simu_class(max_time=max_time, name=None)
#					  for _ in range(num_sims)]
#	
#	def run(self, verbose=True, report_interval=1000):
#		for i, simu in enumerate(self.simus):
#			print >>sys.stderr, "%s: running simulation %s" % (self.name, i)
#			simu.run(verbose=verbose, report_interval=report_interval)