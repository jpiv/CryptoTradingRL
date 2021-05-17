from glob import glob
from spinup.algos.pytorch.ppo.core import MLPActorCritic
import torch
import re

models_path = '/home/jpiv/src/rl/models/'

def save_model(ac):
	print('Saving model...')
	get_model_num = re.compile(r'\d+')
	model_nums = [int(get_model_num.search(filename)[0]) for filename in glob(models_path + '*')]
	model_num = max(model_nums) + 1 if len(model_nums) else 0

	model = ac.pi
	model_name = 'ac_' + str(model_num)
	torch.save(model.state_dict(), models_path + model_name)
	return model_name

def load_model(model_name, obs_space, action_space):
	# print('Loading model', model_name)
	ac = MLPActorCritic(obs_space, action_space)
	ac.pi.load_state_dict(torch.load(models_path + model_name)) 
	return ac

def log_model_performance(model_name, epochs, test_type, total_reward, data, notes=""):
	print('Logging model performance...')
	with open('performance_log.csv', 'a') as log:
		log.write('\n')
		log.write(
			','.join([
				model_name,
				str(epochs),
				test_type,
				str(total_reward),
				str(data['buys']),
				str(data['wins']) + '/' + str(data['losses']),
				str(data['prof']),
				notes
			])
		)


