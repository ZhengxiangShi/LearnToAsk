import sys, os, re, json, argparse, random, nltk, torch, pickle, numpy as np, copy, git, csv
from glob import glob
from datetime import datetime
from os.path import join, isdir
import xml.etree.ElementTree as ET
from torch.autograd import Variable
from sklearn.model_selection import train_test_split as tt_split

class BuilderActionExample():
	def __init__(self, action, built_config, prev_config, action_history):
		self.action = action # of type BuilderAction or None
		self.built_config = built_config
		self.prev_config = prev_config
		self.action_history = action_history

	def is_action(self):
		return isinstance(self.action, BuilderAction)

	def is_stop_token(self):
		return self.action == None

	def __eq__(self, other):
		if not isinstance(other, BuilderActionExample):
			# don't attempt to compare against unrelated types
			return NotImplemented

		return self.action == other.action and self.built_config == other.built_config \
			and self.prev_config == other.prev_config and self.action_history == other.action_history

class BuilderAction():
    """ Class representing a builder's action. """
    def __init__(self, block_x, block_y, block_z, block_type,
        action_type, weight=None):
        """
        Args:
            block_x (int): x-coordinate of block involved in action.
            block_y (int): y-coordinate of block involved in action.
            block_z (int) z-coordinate of block involved in action.
            block_type (string): block type (i.e., color).
            action_type (string): either "pickup" or "putdown".
        """
        assert action_type in ["putdown", "pickup"]

        self.action_type = "placement" if action_type == "putdown" else "removal"  # Is this correct?
        self.block = {
            "x": block_x,
            "y": block_y,
            "z": block_z,
            "type": block_type
        }
        self.weight = weight

    def print(self):
        print("action type: " + str(self.action_type))
        print("x: " + str(self.block["x"]))
        print("y: " + str(self.block["y"]))
        print("z: " + str(self.block["z"]))
        print("type: " + str(self.block["type"]))
        print("weight: " + str(self.weight))

    def __eq__(self, other):
        if not isinstance(other, BuilderAction):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.action_type == other.action_type and self.block == other.block \
            and self.weight == other.weight

color_regex = re.compile("red|orange|purple|blue|green|yellow") # TODO: Obtain from other repo

# assigning IDs to block types aka colors
type2id = {
	"orange": 0,
	"red": 1,
	"green": 2,
	"blue": 3,
	"purple": 4,
	"yellow": 5
}

id2type = {v: k for k, v in type2id.items()}

# assigning IDs to block placement/removal actions
action2id = {
	"placement": 0,
	"removal": 1
}

# bounds of the build region
x_min = -5
x_max = 5
y_min = 1
y_max = 9
z_min = -5
z_max = 5 # TODO: Obtain from other repo
x_range = x_max - x_min + 1
y_range = y_max - y_min + 1
z_range = z_max - z_min + 1

# map from label to detailed info about label
label2details = {}
label_index = 0
for x in range(x_min, x_max + 1):
	for y in range(y_min, y_max + 1):
		for z in range(z_min, z_max + 1):
			for cell_action_label in range(7): # 7623 times -- 0 through 7622
				label2details[label_index] = (x, y, z, cell_action_label)
				label_index += 1

details2struct_dict = {}
for details in label2details.values():
	struct = BuilderActionExample( # TODO: can this be simplified by using BuilderAction instead?
		action = BuilderAction(
			block_x = details[0], block_y = details[1], block_z = details[2],
			block_type = id2type[details[3]] if details[3] < 6 else None,
			action_type = "putdown" if details[3] < 6 else "pickup",
			weight=None
		),
		built_config = None,
		prev_config = None,
		action_history = None
	)
	details2struct_dict[details] = struct

stop_action_label = 7*11*9*11
stop_action_label_tensor = torch.tensor(stop_action_label)
if torch.cuda.is_available():
	stop_action_label_tensor = stop_action_label_tensor.cuda()
stop_action_details = None
stop_action_struct = BuilderActionExample(
	action = None,
	built_config = None,
	prev_config = None,
	action_history = None
)

def details2struct(details):
	if details != None:
		return details2struct_dict[details]
	else: # stop action
		return stop_action_struct

# get repr for decoder input # TODO: dict needed -- like embedding matrix?
def  f2(builder_action):
    if builder_action.action == None: # start action # TODO: organize cases better
        return torch.Tensor([0] * 11)

    action_type = builder_action.action.action_type
    action_id = action2id[action_type]
    action_type_one_hot_vec = [0] * len(action2id)
    action_type_one_hot_vec[action_id] = 1

    color_one_hot_vec = [0] * len(type2id)
    if action_type == "placement":
        color = builder_action.action.block["type"]
        color_id = type2id[color]
        color_one_hot_vec[color_id] = 1

    x = builder_action.action.block["x"]
    y = builder_action.action.block["y"]
    z = builder_action.action.block["z"]
    location_vec = [x, y, z]

    repr = action_type_one_hot_vec + color_one_hot_vec + location_vec ## 2 + 6 + 3 (zshi)

    return torch.Tensor(repr)

def action_label2action_repr(action_label):
	assert action_label != stop_action_label
	return f2(details2struct(label2details.get(action_label)))

# map from label to detailed info about label
# coords2index = {}
# cell_index = 0
# for x in range(x_min, x_max + 1):
# 	for y in range(y_min, y_max + 1):
# 		for z in range(z_min, z_max + 1):
# 			coords2index[(x, y, z)] = cell_index
# 			cell_index += 1

def should_prune_seq(seq):
	return seq[-1] == stop_action_label

def prune_seq(seq, should_prune_seq):
	return seq[:-1] if should_prune_seq else seq

class Logger(object):
	""" Simple logger that writes messages to both console and disk. """

	def __init__(self, logfile_path):
		"""
		Args:
			logfile_path (string): path to where the log file should be saved.
		"""
		self.terminal = sys.stdout
		self.log = open(logfile_path, "a")

	def write(self, message):
		""" Writes a message to both stdout and logfile. """
		self.terminal.write(message)
		self.log.write(message)
		self.log.flush()

	def flush(self):
		pass

class EncoderContext:
	"""
		Output of an encoder set up for use in a corresponding decoder
			- decoder_hidden, decoder_input_concat, etc. point to various ways of conditioning the decoder on the encoder's output
			- Each is initialized appropriately with the the encoder's output so as to be used in the decoder
	"""
	def __init__(self, decoder_hidden=None, decoder_input_concat=None, decoder_hidden_concat=None, decoder_input_t0=None, attn_vec=None):
		self.decoder_hidden = decoder_hidden
		self.decoder_input_concat = decoder_input_concat
		self.decoder_hidden_concat = decoder_hidden_concat
		self.decoder_input_t0 = decoder_input_t0
		self.attn_vec = attn_vec

def take_last_hidden(hidden, num_hidden_layers, bidirectional, batch_size, rnn_hidden_size):
	"""
		Args:
			hidden: Raw hidden returned from RNN
		Returns:
			reshape and take only last layer's hidden state
	"""
	hidden = hidden.view(num_hidden_layers, bidirectional, batch_size, rnn_hidden_size) # (num_layers, num_directions, batch, hidden_size)
	hidden = hidden[-1] # hidden: (num_directions, batch, hidden_size)

	return hidden

def get_logfiles(data_path, split=None):
	"""
	Gets all CwC observation files along without the corresponding gold config. According to a given split.
	Split can be "train", "test" or "val"
	"""
	return get_logfiles_with_gold_config(data_path=data_path, gold_configs_dir=None, split=split, with_gold_config=False)

def get_logfiles_with_gold_config(data_path, gold_configs_dir, split=None, with_gold_config=True, from_aug_data=False):
	"""
	Gets all CwC observation files along with the corresponding gold config, according to a given split.
	Split can be "train", "test" or "val"
	"""

	# get required configs
	# with open(data_path + "/splits.json") as json_data:
	with open(data_path + "splits.json") as json_data:
			data_splits = json.load(json_data)

	configs_for_split = data_splits[split]

	# get all postprocessed observation files along with gold config data
	jsons = []

	all_data_root_dirs = filter(lambda x: isdir(join(data_path, x)), os.listdir(data_path))
	for data_root_dir in all_data_root_dirs:
		logs_root_dir = join(data_path, data_root_dir, "logs")

		all_log_dirs = filter(lambda x: isdir(join(logs_root_dir, x)), os.listdir(logs_root_dir))
		for log_dir in all_log_dirs:
			config_name = re.sub(r"B\d+-A\d+-|-\d\d\d\d\d\d\d+", "", log_dir)

			if config_name not in configs_for_split:
				continue

			if with_gold_config:
				config_xml_file = join(gold_configs_dir, config_name + ".xml")
				config_structure = get_gold_config(config_xml_file)

			logfile = join(logs_root_dir, log_dir, "postprocessed-observations.json")
			with open(logfile) as f:
				loaded_json = json.loads(f.read())
				loaded_json["from_aug_data"] = from_aug_data

				if with_gold_config:
					loaded_json["gold_config_name"] = config_name
					loaded_json["gold_config_structure"] = config_structure
					loaded_json["log_dir"] = log_dir
					loaded_json["logfile_path"] = logfile

				jsons.append(loaded_json)

	return jsons

def get_gold_config(gold_config_xml_file): # TODO: Obtain from other repo
	"""
	Args:
		gold_config_xml_file: The XML file for a gold configuration

	Returns:
		The gold config as a list of dicts -- one dict per block
	"""
	with open(gold_config_xml_file) as f:
		all_lines = map(lambda t: t.strip(), f.readlines())

	gold_config_raw = map(ET.fromstring, all_lines)

	displacement = 100 # TODO: Obtain from other repo
	def reformat(block):
		return {
			"x": int(block.attrib["x"]) - displacement,
			"y": int(block.attrib["y"]),
			"z": int(block.attrib["z"]) - displacement,
			"type": color_regex.findall(block.attrib["type"])[0]
		}

	gold_config = list(map(reformat, gold_config_raw))

	return gold_config

def get_built_config(observation):
	"""
	Args:
		observation: The observations for a cetain world state

	Returns:
		The built config for that state as a list of dicts -- one dict per block
	"""

	built_config_raw = observation["BlocksInGrid"]
	built_config = list(map(reformat, built_config_raw))
	return built_config

def get_builder_position(observation):
	builder_position = observation["BuilderPosition"]

	builder_position = {
		"x": builder_position["X"],
		"y": builder_position["Y"],
		"z": builder_position["Z"],
		"yaw": builder_position["Yaw"],
		"pitch": builder_position["Pitch"]
	}

	return builder_position

def reformat(block):
	return {
		"x": block["AbsoluteCoordinates"]["X"],
		"y": block["AbsoluteCoordinates"]["Y"],
		"z": block["AbsoluteCoordinates"]["Z"],
		"type": color_regex.findall(str(block["Type"]))[0] # NOTE: DO NOT CHANGE! Unicode to str conversion needed downstream when stringifying the dict.
	}

def to_var(x, volatile=False):
	""" Returns an input as a torch Variable, cuda-enabled if available. """
	if torch.cuda.is_available():
		x = x.cuda()
	return Variable(x, volatile=volatile)

def timestamp():
	""" Simple timestamp marker for logging. """
	return "["+datetime.now().strftime('%Y-%m-%d %H:%M:%S')+"]"

def print_dir(path, n):
	path = os.path.abspath(path).split("/")
	return "/".join(path[len(path)-n:])

def tokenize(utterance):
	tokens = utterance.split()
	fixed = ""

	modified_tokens = set()
	for token in tokens:
		original = token

		# fix *word
		if len(token) > 1 and token[0] == '*':
			token = '* '+token[1:]

		# fix word*
		elif len(token) > 1 and token[-1] == '*' and token[-2] != '*':
			token = token[:-1]+' *'

		# fix word..
		if len(token) > 2 and token[-3] != '.' and ''.join(token[-2:]) == '..':
			token = token[:-2]+' ..'

		# split axb(xc) to a x b (x c)
		if len(token) > 2:
			m = re.match("([\s\S]*\d+)x(\d+[\s\S]*)", token)
			while m:
				token = m.groups()[0]+' x '+m.groups()[1]
				m = re.match("([\s\S]*\d+)x(\d+[\s\S]*)", token)

		if original != token:
			modified_tokens.add(original+' -> '+token)

		fixed += token+' '

	return nltk.tokenize.word_tokenize(fixed.strip()), modified_tokens

def get_config_params(config_file):
	with open(config_file, 'r') as f:
		config_content = f.read()

	config_params = {}
	ignore_params = ['model_path', 'data_dir', 'log_step', 'epochs', 'stop_after_n', 'num_workers', 'seed', 'suppress_logs']

	for line in config_content.split('\n'):
		if len(line.split()) != 2:
			continue
		(param, value) = line.split()
		if not any(ignore_param in param for ignore_param in ignore_params):
			config_params[param] = parse_value(value)

	return config_content, config_params

def parse_value(value):
	if value == 'None':
		return None

	try:
		return int(value)
	except ValueError:
		try:
			return float(value)
		except ValueError:
			if value.lower() == 'true' or value.lower() == 'false':
				return str2bool(value)
			return value

def str2bool(v):
	return v.lower() == "true"

def load_pkl_data(filename):
	with open(filename, 'rb') as f:
		data = pickle.load(f)
		print("Loaded data from '%s'" %os.path.realpath(f.name))

	return data

def save_pkl_data(filename, data, protocol=3):
	with open(filename, 'wb') as f:
		pickle.dump(data, f, protocol=protocol)
		print("Saved data to '%s'" %os.path.realpath(f.name))

def get_perspective_coordinates(x, y, z, yaw, pitch):
	# construct vector
	v = np.matrix('{}; {}; {}'.format(x, y, z))

	# construct yaw rotation matrix
	theta_yaw = np.radians(-1 * yaw)
	c, s = np.cos(theta_yaw), np.sin(theta_yaw)
	R_yaw = np.matrix('{} {} {}; {} {} {}; {} {} {}'.format(c, 0, -s, 0, 1, 0, s, 0, c))

	# multiply
	v_new = R_yaw * v

	# construct pitch rotation matrix
	theta_pitch = np.radians(pitch)
	c, s = np.cos(theta_pitch), np.sin(theta_pitch)
	R_pitch = np.matrix('{} {} {}; {} {} {}; {} {} {}'.format(1, 0, 0, 0, c, s, 0, -s, c))

	# multiply
	v_final = R_pitch * v_new
	x_final = v_final.item(0)
	y_final = v_final.item(1)
	z_final = v_final.item(2)
	return (x_final, y_final, z_final)

vf = np.vectorize(get_perspective_coordinates)

def get_perspective_coord_repr(builder_position):
	bx = builder_position["x"]
	by = builder_position["y"]
	bz = builder_position["z"]
	yaw = builder_position["yaw"]
	pitch = builder_position["pitch"]

	perspective_coords = np.zeros((3, x_range, y_range, z_range))
	for x in range(x_range):
		for y in range(y_range):
			for z in range(z_range):
				xm, ym, zm = x-bx, y-by, z-bz
				perspective_coords[0][x][y][z] = xm
				perspective_coords[1][x][y][z] = ym
				perspective_coords[2][x][y][z] = zm

	px, py, pz = vf(perspective_coords[0], perspective_coords[1], perspective_coords[2], yaw, pitch)
	return np.stack([px, py, pz])

def add_action_type(action, placement_or_removal):
	assert placement_or_removal in ["placement", "removal"]

	action_copy = copy.deepcopy(action)
	action_copy["action_type"] = placement_or_removal

	return action_copy

architect_prefix = "<Architect> "
builder_prefix = "<Builder> "

def get_data_splits(args):
	"""
	Writes a file containing the train-val-test splits at the config level
	"""

	# utils
	warmup_configs_blacklist = ["C3", "C17", "C32", "C38"] # TODO: import from another repo

	# get all gold configs

	gold_configs = []

	for gold_config_xml_file in glob(args.gold_configs_dir + '/*.xml'):
		gold_config = gold_config_xml_file.split("/")[-1][:-4]
		gold_configs.append(gold_config)

	# filter out warmup ones
	gold_configs = list(filter(lambda x: x not in warmup_configs_blacklist, gold_configs))

	# split
	train_test_split = tt_split(gold_configs, random_state=args.seed) # default is 0.75:0.25

	train_configs = train_test_split[0]
	test_configs = train_test_split[1]

	train_val_split = tt_split(train_configs, random_state=args.seed) # default is 0.75:0.25

	train_configs = train_val_split[0]
	val_configs = train_val_split[1]

	# write split to file
	splits = {
		"train": train_configs,
		"val": val_configs,
		"test": test_configs
	}

	with open(args.data_path + "/splits.json", "w") as file:
		json.dump(splits, file)

def get_augmented_data_splits(data_path, gold_configs_dir, splits_json_for_orig_data):

	def find_set(orig_gold_config, orig_data_splits):
		if orig_gold_config in orig_data_splits["train"]:
			return "train"
		elif orig_gold_config in orig_data_splits["val"]:
			return "val"
		elif orig_gold_config in orig_data_splits["test"]:
			return "test"
		else:
			return None # warmup config

	# load original data splits
	with open(splits_json_for_orig_data) as json_data:
            orig_data_splits = json.load(json_data)

	# get all gold configs in augmented data
	gold_configs = []

	for gold_config_xml_file in glob(gold_configs_dir + '/*.xml'):
		gold_config = gold_config_xml_file.split("/")[-1][:-4]
		gold_configs.append(gold_config)

	# split

	aug_data_splits = {
		"train": [],
		"val": [],
		"test": []
	}

	for gold_config in gold_configs:
		# find right set -- train/test/val
		corresponding_orig_gold_config = gold_config.split("_")[0]
		split_set = find_set(corresponding_orig_gold_config, orig_data_splits)
		# assign to a set iff it's not a warmup config
		if split_set:
			aug_data_splits[split_set].append(gold_config)

	with open(data_path + "/splits.json", "w") as f:
		json.dump(aug_data_splits, f)

	print("\nSaving git commit hashes ...\n")
	write_commit_hashes("..", data_path, filepath_modifier="_splits_json")

def is_feasible_next_removal(block, built_config):
	block_exists = any(
		existing_block["x"] == block["x"] and existing_block["y"] == block["y"] and existing_block["z"] == block["z"] for existing_block in built_config
	)

	return block_exists

def initialize_rngs(seed, use_cuda=False):
    np.random.seed(seed) # cpu vars
    torch.manual_seed(seed) # cpu  vars
    random.seed(seed) # Python
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # gpu vars
        # torch.backends.cudnn.deterministic = True  #needed
        # torch.backends.cudnn.benchmark = False

def get_commit_hashes(models_repo_path):
	models_repo = git.Repo(models_repo_path)
	models_repo_commit_hash = models_repo.head.object.hexsha

	return models_repo_commit_hash

def write_commit_hashes(models_repo_path, dir_to_write, filepath_modifier=""):
	models_repo_commit_hash = get_commit_hashes(models_repo_path)

	all_csv_content = [
		{
			"repo_type": "models_repo",
			"repo_path": os.path.abspath(models_repo_path),
			"commit_hash": models_repo_commit_hash
		}
	]

	keys = all_csv_content[0].keys()
	with open(os.path.join(dir_to_write, "commit_hashes" + filepath_modifier + ".csv"), 'w') as output_file:
		dict_writer = csv.DictWriter(output_file, keys)
		dict_writer.writeheader()
		dict_writer.writerows(all_csv_content)

if __name__ == "__main__":
	"""
	Use this section for generating the splits files (you shouldn't need to run this -- think carefully about what you are doing).
	"""

	parser = argparse.ArgumentParser()

	parser.add_argument('--data_path', type=str, default='../data/logs/', help='path for data jsons')
	parser.add_argument('--gold_configs_dir', type=str, default='../data/gold-configurations/', help='path for gold config xmls')

	parser.add_argument('--aug_data_dir', type=str, default='../data/augmented/', help='path for aug data')

	parser.add_argument('--seed', type=int, default=1234, help='random seed')

	args = parser.parse_args()

	initialize_rngs(args.seed, torch.cuda.is_available())

	# get_data_splits(args)

	get_augmented_data_splits(
		os.path.join(args.aug_data_dir, "logs"),
		os.path.join(args.aug_data_dir, "gold-configurations"),
		os.path.join(args.data_path, "splits.json")
	)
