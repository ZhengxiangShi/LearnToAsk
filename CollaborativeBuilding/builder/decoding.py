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

class ActionSeq:
	def __init__(self, decoder_hidden, last_idx, built_config_post_last_action, action_history_post_last_action, seq_idxes=[], seq_scores=[], action_feasibilities=[]):
		if(len(seq_idxes) != len(seq_scores)):
			raise ValueError("length of indexes and scores should be the same")
		self.decoder_hidden = decoder_hidden
		self.last_idx = last_idx
		self.seq_idxes =  seq_idxes
		self.seq_scores = seq_scores
		self.built_config_post_last_action = built_config_post_last_action
		self.action_history_post_last_action = action_history_post_last_action
		self.action_feasibilities = action_feasibilities

	def likelihoodScore(self):
		"""
			log likelihood score
		"""
		if len(self.seq_scores) == 0:
			return -99999999.999 # TODO: check
		# return mean of sentence_score
		# TODO: Relates to the normalized loss function used when training?
		# NOTE: No need to length normalize when making selection for beam. Only needed during final selection.
		return sum(self.seq_scores) / len(self.seq_scores) # NOTE: works without rounding error because these are float tensors

	def addTopk(self, topi, topv, decoder_hidden, beam_size, EOS_tokens):
		terminates, seqs = [], []
		for i in range(beam_size):
			idxes = self.seq_idxes[:] # pass by value
			scores = self.seq_scores[:] # pass by value

			idxes.append(topi[0][i])
			scores.append(topv[0][i])

			is_feasible = is_feasible_action(self.built_config_post_last_action, topi[0][i].item())
			action_feasibilities = self.action_feasibilities[:] # pass by value
			action_feasibilities.append(is_feasible) # TODO: don't recompute feasibility in following code

			built_config_post_last_action = update_built_config(self.built_config_post_last_action, topi[0][i].item())
			action_history_post_last_action = update_action_history(self.action_history_post_last_action, topi[0][i].item(), self.built_config_post_last_action)

			seq = ActionSeq(
				decoder_hidden=decoder_hidden, last_idx=topi[0][i], built_config_post_last_action=built_config_post_last_action,
				action_history_post_last_action=action_history_post_last_action, seq_idxes=idxes, seq_scores=scores,
				action_feasibilities=action_feasibilities
			)

			if topi[0][i] in EOS_tokens:
				terminates.append((
					[idx.item() for idx in seq.seq_idxes], # TODO: need the eos token?
					seq.likelihoodScore(),
					seq.action_feasibilities,
					seq.built_config_post_last_action
				)) # tuple(word_list, score_float, action feasibilities, end_built_config)
			else:
				seqs.append(seq)

		return terminates, seqs # NOTE: terminates can be of size 0 or 1 only

def beam_decode_action_seq(model, raw_inputs, encoder_inputs, labels, location_mask,
    beam_size, max_length, testdataset, num_top_seqs,
	initial_grid_repr_input):

	terminal_seqs, prev_top_seqs, next_top_seqs = [], [], []
	prev_top_seqs.append(
		ActionSeq(
			decoder_hidden=None, last_idx=torch.tensor(-1), # start token assigned action id of -1
			built_config_post_last_action=raw_inputs.initial_prev_config_raw, # same as post SOS token
			action_history_post_last_action=raw_inputs.initial_action_history_raw,
			seq_idxes=[], seq_scores=[], action_feasibilities=[]
		)
	)

	for _ in range(max_length):
		for seq in prev_top_seqs:
			# never stop action here -- .get is actually not needed
			# print(seq.last_idx)
			action_repr_input = action_label2action_repr(seq.last_idx.item()).view(1, 1, -1) # NOTE: should be [1, 1, x]
			# print(action_repr_input.shape)

			grid_repr_input = testdataset.get_repr(
	            BuilderActionExample(
	                action=None, # only ever used for computing output label which we don't need -- so None is okay
	                built_config=None,
	                prev_config=seq.built_config_post_last_action,
	                action_history=seq.action_history_post_last_action
	            ),
	            raw_inputs.perspective_coords
	        )[0].unsqueeze(0)
			# print(grid_repr_input.shape)

			# print(encoder_inputs.shape, grid_repr_input.shape, action_repr_input.shape, labels.shape, location_mask.shape)
			loss, test_acc, test_predicted_seq = model(encoder_inputs.long().cuda(), grid_repr_input.cuda(), action_repr_input.squeeze(0).cuda(), labels.long().cuda(), location_mask.cuda(), raw_input=raw_inputs, dataset=testdataset)
			# decoder_output
			# print(decoder_output.shape) # [1, 7624]

			# m = nn.LogSoftmax()
			# decoder_output = m(decoder_output)

			# topv, topi = decoder_output.topk(beam_size) # topv : tensor([[-0.4913, -1.9879, -2.4969, -3.6227, -4.0751]])
			topv, topi = torch.tensor([1]).view(1,1), torch.tensor([convert_to_scalar_label(test_predicted_seq)]).view(1,1)
			term, top = seq.addTopk(topi, topv, None, beam_size, [torch.tensor(7*11*9*11)])
			terminal_seqs.extend(term)
			next_top_seqs.extend(top)

		next_top_seqs.sort(key=lambda s: s.likelihoodScore(), reverse=True)
		prev_top_seqs = next_top_seqs[:beam_size]
		next_top_seqs = []

	terminal_seqs += [
		([idx.item() for idx in seq.seq_idxes], seq.likelihoodScore(), seq.action_feasibilities, seq.built_config_post_last_action) for seq in prev_top_seqs
	]
	terminal_seqs.sort(key=lambda x: x[1], reverse=True)

	# print(terminal_seqs)

	if num_top_seqs is not None:
		top_terminal_seqs = list(map(lambda x: (prune_seq(x[0], should_prune_seq(x[0])), prune_seq(x[2], should_prune_seq(x[0])), x[3]), terminal_seqs[:num_top_seqs]))
	else:
		top_terminal_seqs = list(map(lambda x: (prune_seq(x[0], should_prune_seq(x[0])), prune_seq(x[2], should_prune_seq(x[0])), x[3]), terminal_seqs))

	return top_terminal_seqs # terminal_seqs[0][0][:-1]

def generate_action_pred_seq(model, test_item_batches, beam_size, max_length, testdataset):
	model.eval()

	generated_seqs, to_print = [], []
	total_examples = str(len(test_item_batches)) 

	try:
		with torch.no_grad():
			for i, data in enumerate(test_item_batches, 0):

				# get the inputs; data is a list of [inputs, labels]
				encoder_inputs, grid_repr_inputs, action_repr_inputs, labels, location_mask, raw_inputs = data
				encoder_inputs, grid_repr_inputs, action_repr_inputs, labels, location_mask = encoder_inputs.unsqueeze(0), grid_repr_inputs.unsqueeze(0), action_repr_inputs.unsqueeze(0), labels.unsqueeze(0), location_mask.unsqueeze(0)
				# print(encoder_inputs.shape, grid_repr_inputs.shape, action_repr_inputs.shape, labels.shape, location_mask.shape)
				# loss, test_acc, test_predicted_seq = model(encoder_inputs.long().cuda(), grid_repr_inputs.cuda(), action_repr_inputs.cuda(), labels.long().cuda(), location_mask.cuda(), raw_input=raw_inputs, dataset=testdataset)
				"""
				encoder_inputs: [batch_size, max_length]
				grid_repr_inputs: [batch_size=1, act_len, 8, 11, 9, 11]
				action_repr_inputs: [batch_size=1, act_len, 11]
				location_mask: [batch_size=1, act_len, 1089]
				labels: [batch_size=1, act_len, 7]
				"""
				generated_seq = beam_decode_action_seq(model, raw_inputs, encoder_inputs, labels[:,0], location_mask[:,0],
					beam_size, max_length, testdataset, 1,
					initial_grid_repr_input=grid_repr_inputs[:,0]
				) # list of tuples -- [(seq, feas, end_built_configs)]

				# list(map(lambda x: x[0], generated_seq))
				# list(map(lambda x: x[1], generated_seq))

				generated_seqs.append(
					{
						"generated_seq": list(map(lambda x: x[0], generated_seq)),
                        "ground_truth_seq": labels,
						"prev_utterances": None, # encoder_inputs.prev_utterances,
						"action_feasibilities": list(map(lambda x: x[1], generated_seq)),
						"generated_end_built_config": list(map(lambda x: x[2], generated_seq)),
						"ground_truth_end_built_config": raw_inputs.end_built_config_raw,
						"initial_built_config": raw_inputs.initial_prev_config_raw,
						"initial_action_history": raw_inputs.initial_action_history_raw
					}
				)

				if i % 20 == 0:
					print(
						timestamp(),
						'['+str(i)+'/'+total_examples+']',
						list(map(
							lambda x: ", ".join(list(map(lambda y: str(y), x))),
							list(map(lambda x: x[0], generated_seq))
						))
					)

				to_print.append(
					list(map(
						lambda x: ", ".join(list(map(lambda y: str(y), x))),
						list(map(lambda x: x[0], generated_seq))
					))
				)
	except KeyboardInterrupt:
		print("Generation ended early; quitting.")

	return generated_seqs, to_print


def convert_to_scalar_label(pred):
    stop_action_label = 7*11*9*11
    location_pred, action_type_pred, color_pred = pred
    if action_type_pred==2:
        return int(stop_action_label)
    elif action_type_pred==1:
        return int(location_pred*7+6)
    else:
        return int(location_pred*7+color_pred)