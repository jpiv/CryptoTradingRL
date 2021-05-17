from spinup import ppo_pytorch as ppo
from strategy_env import StrategyEnv
from freqtrade_files.Strategies.BB2 import BB2
import models
import torch
import gym
import numpy as np
import matplotlib.pyplot as plt

ac = None
env = None
use_model = False
epochs = 1000 

def run_test_with_strat(env):
	total_reward = 0
	obs, r, d, _ = env.step(None)
	while not d:
		obs, reward, d, _ = env.step(None)
		total_reward += reward
	print("TOTAL: " + str(total_reward))
	env.render()

def run_test_with_ac(env, ac):
	total_reward = 0
	o, r, d, _ = env.step(0)
	while not d:
	    a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))
	    next_o, r, d, _ = env.step(a) # take a random action
	    o = next_o
	    total_reward += r
	print("TOTAL: " + str(total_reward))
	return total_reward, env.render()


def create_env():
	env = gym.make('StrategyEnv-v0', load_bt=False)
	return env

def run_tests(model_name = None):
	global epochs
	global env
	global ac
	env = gym.make('StrategyEnv-v0')
	test_tf = '20200201-20200207'
	# test_tf = '20200401-'
	env.set_timeframe(test_tf)
	env.full_reset()

	def make_env():
		env = gym.make('StrategyEnv-v0')
		env.set_timeframe('20191110-20200131')
		env.randomize_timeframe(True)
		env.set_ac(True)
		env.full_reset()
		return env

	#ask joey to run normal BT to compare
	run_test_with_strat(env)
	env.run_normal_bt()

	if model_name:
		torch.manual_seed(10000)
		np.random.seed(10000)
		ac = models.load_model(model_name, env.observation_space, env.action_space)
	else:
		# ac = ppo(make_env, epochs=epochs, target_kl=0.001, steps_per_epoch=7200, max_ep_len=100000)
		ac = ppo(make_env, epochs=epochs, steps_per_epoch=7200, max_ep_len=100000)
		model_name = models.save_model(ac)

	run_model_test(env, ac, model_name)

def run_model_test(env, ac, model_name):
	global use_model
	env.set_ac(True)
	use_model = True
	# Need to unit test normal bt vs ac bt (last check good)
	print('Running AC with strategy')
	env.full_reset()
	env.run_normal_bt()
	
	use_model = False
	env.set_export_file('ac_results.json')
	print('Running AC with backtest')
	env.full_reset()
	run_test_with_ac(env, ac)

	env.set_ac(False)
	env.set_export_file('normal_bt.json')
	env.full_reset()
	print('Running original strategy')
	run_test_with_strat(env)

	env.set_ac(True)
	env.set_export_file('general_test.json')
	env.set_timeframe('20191110-20200131')
	print('Running general test with AC')
	env.full_reset()
	tr, res = run_test_with_ac(env, ac)
	result_data = agg_results(res)
	models.log_model_performance(model_name, epochs, 'general', tr, result_data, 'add total prof on trade')

	env.set_ac(False)
	env.set_export_file('general_normal.json')
	print('Running general test with original strategy')
	env.full_reset()
	run_test_with_strat(env)

def agg_results(results):
	data = results['BB2']
	return {
		'buys': len(data.index),
		'prof': data.profit_abs.sum(),
        'wins': len(data[data.profit_abs > 0]),
       	'losses': len(data[data.profit_abs < 0]),
	}

min_rets = []
max_rets = []
avg_rets = []

def add_results(avg_ret, min_ret, max_ret):
	global min_rets
	global max_rets
	global avg_rets
	min_rets.append(min_ret)
	max_rets.append(max_ret)
	avg_rets.append(avg_ret)

def plot_results():
	global min_rets
	global max_rets
	global avg_rets
	fig, ax = plt.subplots()

	# Using set_dashes() to modify dashing of an existing line
	line1 = ax.plot(avg_rets, label='Avg Ret')
	line2 = ax.plot(max_rets, label='Max Ret')
	line3 = ax.plot(min_rets, label='Min Ret')

	ax.legend()
	plt.show()
