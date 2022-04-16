import re
from builder.decoding import update_built_config, is_feasible_action, update_action_history
from builder.diff import diff, get_diff, dict_to_tuple
from utils import action_label2action_repr, BuilderActionExample
import copy

stop_action_label = 7*11*9*11

def is_a_id(string):
    line = re.sub(r'\d+', '', string)
    if line == 'B-A-C-': return True
    return False

def split_line(line):
    line_elems = line.split('\t')
    if len(line_elems) == 2:
        utterance = line_elems[0]
        label = line_elems[1]
        corrected_utterance = None
        return utterance, label, corrected_utterance
    elif len(line_elems) == 3:
        utterance = line_elems[0]
        label = line_elems[1]
        corrected_utterance = line_elems[2]
        return utterance, label, corrected_utterance
    else:
        print('Error', line)
        return None, None, None

def compute_action_prf(fn, fp, tp):
    action_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    action_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    action_f1 = ((2 * action_precision * action_recall) / (action_precision + action_recall)) if (action_precision + action_recall) > 0 else 0.0
    return action_precision, action_recall, action_f1

def compute_metrics(raw_input, action_seq):
    initial_built_config = raw_input.initial_prev_config_raw
    ground_truth_end_built_config = raw_input.end_built_config_raw
    
    built_config_post_last_action = initial_built_config
    for action_label in action_seq:
        built_config_post_last_action = update_built_config(built_config_post_last_action, action_label)

    generated_end_built_config = built_config_post_last_action

    net_change_generated = diff(gold_config=generated_end_built_config, built_config=initial_built_config)
    net_change_gt = diff(gold_config=ground_truth_end_built_config, built_config=initial_built_config)

    net_change_generated = diff2actions(net_change_generated)
    net_change_gt = diff2actions(net_change_gt)

    fn, fp, tp = get_pos_neg_count(net_change_generated, net_change_gt)

    return fn, fp, tp

def evaluate_metrics(valid_pred_seqs, valid_raw_inputs):
    total_val_fn = 0
    total_val_fp = 0
    total_val_tp = 0
    assert len(valid_pred_seqs) == len(valid_raw_inputs)
    for val_raw, val_pred_seq in zip(valid_raw_inputs, valid_pred_seqs):
        val_sscalar_label_seq = list(map(convert_to_scalar_label, val_pred_seq))
        val_fn, val_fp, val_tp = compute_metrics(val_raw, val_sscalar_label_seq)
        total_val_fn += val_fn
        total_val_fp += val_fp
        total_val_tp += val_tp      
    val_action_precision, val_action_recall, val_action_f1 = compute_action_prf(total_val_fn, total_val_fp, total_val_tp)
    return val_action_precision, val_action_recall, val_action_f1



def diff2actions(diff):
    placements = diff["gold_minus_built"]
    placements = list(map(
        lambda block: add_action_type(block, "placement"),
        placements
    ))

    removals = diff["built_minus_gold"]
    removals = list(map(
        lambda block: add_action_type(block, "removal"), # TODO: color None?
        removals
    ))

    return placements + removals

def add_action_type(action, placement_or_removal):
	assert placement_or_removal in ["placement", "removal"]

	action_copy = copy.deepcopy(action)
	action_copy["action_type"] = placement_or_removal

	return action_copy

def get_pos_neg_count(generated_actions, gt_actions):
    net_change_generated = set(map(dict_to_tuple, generated_actions))
    net_change_gt = set(map(dict_to_tuple, gt_actions))

    fn = len(net_change_gt - net_change_generated)
    fp = len(net_change_generated - net_change_gt)
    tp = len(net_change_generated & net_change_gt)

    return fn, fp, tp

def get_feasibile_location(built_config):
    """
    location_mask: generate a list with 1089 locations. 1 means feasible. 
    """
    location_mask = []
    for action_label in range(0, 7*11*9*11, 7):
        is_feasible = 0 
        for action_id in [0, 6]: # Test placement and removal respectively
            action_label_test = action_label + action_id
            if is_feasible_action(built_config, action_label_test):
                is_feasible = 1
                break
        location_mask.append(is_feasible)
    return location_mask

def update_last_action(action_location, action_action_type, color_pred):
    action_label = convert_to_scalar_label((action_location, action_action_type, color_pred))
    return action_label2action_repr(action_label)

def update_obs(prev_config, action_history, action_location, action_action_type, color_pred, dataloader, perspective_coords):
    action_label = convert_to_scalar_label((action_location, action_action_type, color_pred))
    built_config_post_last_action = update_built_config(prev_config, action_label) 
    new_action_history = update_action_history(action_history, action_label, prev_config)
    cell_samples_reprs_4d, label, location_mask = dataloader.get_repr(
        BuilderActionExample(
            action=None, # only ever used for computing output label which we don't need -- so None is okay
            built_config=None,
            prev_config=built_config_post_last_action,
            action_history=new_action_history),
            perspective_coords
	        )
    return cell_samples_reprs_4d, location_mask, built_config_post_last_action, new_action_history

def convert_to_scalar_label(pred):
    location_pred, action_type_pred, color_pred = pred
    if action_type_pred==2:
        return int(stop_action_label)
    elif action_type_pred==1:
        return int(location_pred*7+6)
    else:
        return int(location_pred*7+color_pred)


# bounds of the build region
x_min = -5
x_max = 5
y_min = 1
y_max = 9
z_min = -5
z_max = 5 
location2xyz = {}
location_id = 0
for x in range(x_min, x_max + 1):
    for y in range(y_min, y_max + 1):
        for z in range(z_min, z_max + 1):
            location2xyz[location_id] = (x, y, z)
            location_id += 1
