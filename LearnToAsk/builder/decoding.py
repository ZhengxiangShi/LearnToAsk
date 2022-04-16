import torch, sys, torch.nn as nn, re
sys.path.append('..')
from utils import *
from builder.diff import is_feasible_next_placement

def is_feasible_action(built_config_post_last_action, new_action_label):
	# new_action_label in 0-7624
	new_action = details2struct(label2details.get(new_action_label))

	if new_action.action != None:
		if new_action.action.action_type == "placement":
			return is_feasible_next_placement(block=new_action.action.block, built_config=built_config_post_last_action, extra_check=True)
		else:
			return is_feasible_next_removal(block=new_action.action.block, built_config=built_config_post_last_action)
	else: # stop action
		return True

def get_feasibility_bool_mask(built_config):
	bool_mask = []

	for action_label in range(7*11*9*11):
		bool_mask.append(is_feasible_action(built_config, action_label))

	return bool_mask

def update_built_config(built_config_post_last_action, new_action_label): # TODO: see that logic is air tight == feasibility too
	# new_action_label in 0-7624
	new_action = details2struct(label2details.get(new_action_label)) # returen a BuilderActionExample

	if new_action.action != None:
		if new_action.action.action_type == "placement":
			if is_feasible_next_placement(block=new_action.action.block, built_config=built_config_post_last_action, extra_check=True):
				# print("here")
				new_built_config = built_config_post_last_action + [new_action.action.block]
			else:
				# print("there")
				new_built_config = built_config_post_last_action
		else:
			# print(built_config_post_last_action)
			if is_feasible_next_removal(block=new_action.action.block, built_config=built_config_post_last_action):
				new_built_config = list(filter(
					lambda block: block["x"] != new_action.action.block["x"] or block["y"] != new_action.action.block["y"] or block["z"] != new_action.action.block["z"],
					built_config_post_last_action
				))
			else:
				# print("there")
				new_built_config = built_config_post_last_action
			# print(new_built_config)
			# print("\n\n")
	else: # stop action
		new_built_config = built_config_post_last_action

	return new_built_config

def update_action_history(action_history_post_last_action, new_action_label, built_config_post_last_action):
	# new_action_label in 0-7624
	new_action = details2struct(label2details.get(new_action_label))

	if new_action.action != None:
		if new_action.action.action_type == "placement":
			if is_feasible_next_placement(block=new_action.action.block, built_config=built_config_post_last_action, extra_check=True):
				new_action_history = action_history_post_last_action + [new_action.action]
			else:
				new_action_history = action_history_post_last_action
		else:
			if is_feasible_next_removal(block=new_action.action.block, built_config=built_config_post_last_action):
				new_action_history = action_history_post_last_action + [new_action.action]
			else:
				new_action_history = action_history_post_last_action
		# new_action_history = action_history_post_last_action + [new_action.action] # TODO: check use of extra parens in data loader, color can be None for removals
	else: # stop action
		# print("added stop action to action history")
		new_action_history = action_history_post_last_action + [None] # TODO: replace None?

	return new_action_history
