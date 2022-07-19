from email.policy import default
import sys, os, nltk, pickle, argparse, gzip, csv, json, torch, numpy as np, torch.nn as nn
from collections import defaultdict

sys.path.append('..')
from utils import Logger, get_logfiles, tokenize, architect_prefix, builder_prefix, type2id, initialize_rngs, write_commit_hashes

class Vocabulary(object):
	"""Simple vocabulary wrapper."""
	def __init__(self, data_path='../data/logs/', vector_filename=None, embed_size=300, use_speaker_tokens=False, use_builder_action_tokens=False, add_words=True, lower=False, threshold=0, all_splits=False, add_builder_utterances=False, builder_utterances_only=False):
		"""
		Args:
			data_path (string): path to CwC official data directory.
			vector_filename (string, optional): path to pretrained embeddings file.
			embed_size (int, optional): size of word embeddings.
			use_speaker_tokens (boolean, optional): use speaker tokens <Architect> </Architect> and <Builder> </Builder> instead of sentence tokens <s> and </s>
			use_builder_action_tokens (boolean, optional): use builder action tokens for pickup/putdown actions, e.g. <builder_pickup_red> and <builder_putdown_red>
			add_words (boolean, optional): whether or not to add OOV words to vocabulary as random vectors. If not, all OOV tokens are treated as unk.
			lower (boolean, optional): whether or not to lowercase all tokens.
			keep_all_embeddings (boolean, optional): whether or not to keep embeddings in pretrained files for all words (even those out-of-domain). Significantly reduces file size, memory usage, and processing time if set.
			threshold (int, optional): rare word threshold (for the training set), below which tokens are treated as unk.
			add_builder_utterances (boolean, optional): whether or not to obtain examples for builder utterances as well
			builder_utterances_only (boolean, optional): whether or not to only include builder utterances
		"""
		# do not freeze embeddings if we are training our own
		self.data_path = data_path
		self.vector_filename = vector_filename
		self.embed_size = embed_size
		self.freeze_embeddings = vector_filename is not None
		self.use_speaker_tokens = use_speaker_tokens
		self.use_builder_action_tokens = use_builder_action_tokens
		self.add_words = add_words
		self.lower = lower
		self.threshold = threshold
		self.all_splits = all_splits
		self.add_builder_utterances = add_builder_utterances
		self.builder_utterances_only = builder_utterances_only

		print("Building vocabulary.\n\tdata path:", self.data_path, "\n\tembeddings file:", self.vector_filename, "\n\tembedding size:", self.embed_size,
			"\n\tuse speaker tokens:", self.use_speaker_tokens, "\n\tuse builder action tokens:", self.use_builder_action_tokens, "\n\tadd words:", self.add_words,
			"\n\tlowercase:", self.lower, "\n\trare word threshold:", self.threshold, "\n")

		# mappings from words to IDs and vice versa
		# this is what defines the vocabulary
		self.word2idx = {}
		self.idx2word = {}

		# mapping from words to respective counts in the dataset
		self.word_counts = defaultdict(int)
		# entire dataset in the form of tokenized utterances
		self.tokenized_data = []
		# words that are frequent in the dataset but don't have pre-trained embeddings
		self.oov_words = set()

		# store dataset in tokenized form and it's properties
		self.get_dataset_properties() # self.word_counts and self.tokenized_data populated

		# initialize word vectors
		self.init_vectors() # self.word_vectors, self.word2idx and self.idx2word populated for aux tokens

		# load pretrained word vectors
		if vector_filename is not None:
			self.load_vectors() # self.word_vectors, self.word2idx and self.idx2word populated for real words

		# add random vectors for oov train words -- words that are in data, frequent but do not have a pre-trained embedding
		if add_words or vector_filename is None: ## True
			self.add_oov_vectors()

		# create embedding variable
		self.word_embeddings = nn.Embedding(self.word_vectors.shape[0], self.word_vectors.shape[1])

		# initialize embedding variable
		self.word_embeddings.weight.data.copy_(torch.from_numpy(self.word_vectors))

		# freeze embedding variable
		if self.freeze_embeddings:
			self.word_embeddings.weight.requires_grad = False

		self.num_tokens = len(self.word2idx)
		self.print_vocab_statistics()

	def get_dataset_properties(self):
		jsons = get_logfiles(self.data_path, split='train')
		if self.all_splits:
			print("Using all three of train, val and test data...")
			jsons += get_logfiles(self.data_path, split='val') + get_logfiles(self.data_path, split='test')
		fixed_tokenizations = set()

		# compute word counts for all tokens in training dataset
		for i in range(len(jsons)):
			js = jsons[i]
			final_observation = js["WorldStates"][-1]

			for i in range(1, len(final_observation["ChatHistory"])):
				line = final_observation["ChatHistory"][i]

				speaker = "Architect" if "Architect" in line.split()[0] else "Builder"
				if speaker == "Architect":
					# Skip over architect if only want builder utterances
					if self.builder_utterances_only:
						continue
					else:
						utterance = line[len(architect_prefix):]
				else:
					# Include builder if either flag is on
					if self.add_builder_utterances or self.builder_utterances_only:
						utterance = line[len(builder_prefix):]
					else:
						continue

				if self.lower:
					utterance = utterance.lower()

				tokenized, fixed = tokenize(utterance)
				fixed_tokenizations.update(fixed)
				self.tokenized_data.append(tokenized) ## store all tokens in dataset into tokenized_data (zshi)

				for word in tokenized:
					self.word_counts[word] += 1

		print("\nTokenizations fixed:", fixed_tokenizations, "\n")

	def init_vectors(self):
		embed_size = self.embed_size

		# pad token
		vector_arr = np.zeros((1, embed_size))
		self.add_word('<pad>') # id 0 # vector of 0's

		if not self.vector_filename:
			vector_arr = np.concatenate((vector_arr, np.random.rand(1, embed_size)), 0)
			self.add_word('<unk>')

		# start & end of sentence symbols
		if not self.use_speaker_tokens: ## False (zshi)
			vector_arr = np.concatenate((vector_arr, np.random.rand(2, embed_size)), 0)
			self.add_word('<s>') # id 1 # random vector
			self.add_word('</s>') # id 2 # random vector

		# speaker tokens
		else:
			vector_arr = np.concatenate((vector_arr, np.random.rand(6, embed_size)), 0)
			self.add_word('<dialogue>') # id 1 # random vector
			self.add_word('</dialogue>') # id 2 # random vector
			self.add_word('<architect>') # id 3 # random vector
			self.add_word('</architect>') # id 4 # random vector
			self.add_word('<builder>') # id 5 # random vector
			self.add_word('</builder>') # id 6 # random vector

		if self.use_builder_action_tokens: ## False (zshi)
			vector_arr = np.concatenate((vector_arr, np.random.rand(12, embed_size)), 0)
			for color_key in type2id:
				self.add_word('<builder_pickup_'+color_key+'>')
				self.add_word('<builder_putdown_'+color_key+'>')

		self.word_vectors = vector_arr

	def load_vectors(self):
		vector_filename = self.vector_filename
		print("\nLoading word embedding vectors from", vector_filename, "...")

		embed_size = self.embed_size
		threshold = self.threshold

		f = None
		if vector_filename.endswith('.gz'): # TODO: this doesn't work. fix loading of word2vec pretrained model
			f = gzip.open(vector_filename, 'rt')
		else:
			f = open(vector_filename, 'r')

		# load word embeddings from file
		data = []
		data_rare = []
		for line in f:
			tokens = line.split()

			if len(tokens) != embed_size+1:
				print("Warning: expected line length:", str(embed_size+1)+", got:", str(len(tokens))+"; Line:", " ".join(line.split()[:3]), "...", " ".join(line.split()[-3:]))
				continue

			word = tokens[0]

			# discard out-of-domain embeddings, if desired
			if word not in self.word_counts or self.word_counts[word] < threshold: # if not in data or if rare
				if word in self.word_counts: # rare words in data but have embeddings
					for t in tokens[1:]:
						data_rare.append(float(t))
				continue

			for t in tokens[1:]:
				data.append(float(t))

			self.add_word(word)

		data_arr = np.reshape(data, newshape=(int(len(data)/embed_size), embed_size)) # real frequent words x embedding size
		self.word_vectors = np.concatenate((self.word_vectors, data_arr), 0)
		if data_rare:
			data_rare_arr = np.reshape(data_rare, newshape=(int(len(data_rare)/embed_size), embed_size)) # real rare words x embedding size
			avg_vector = np.expand_dims(np.mean(data_rare_arr, axis=0), axis=0)
			self.add_unk(avg_vector) # set unk's embedding to the average embedding of the rare words
		else:
			print("Adding random vector for rare bucket as there are no rare words...")
			self.add_unk(np.random.rand(1, embed_size))

	def add_unk(self, avg_vector):
		self.word_vectors = np.concatenate((self.word_vectors, avg_vector), 0)
		self.add_word('<unk>')

	def add_oov_vectors(self):
		print("\nAdding OOV vectors ...")

		threshold = self.threshold
		embed_size = self.embed_size

		# add words above the rare word threshold, but do not have pretrained embeddings, as OOV random vectors
		for utterance in self.tokenized_data:
			for word in utterance:
				if word not in self.word2idx and self.word_counts[word] >= threshold:
					self.word_vectors = np.concatenate((self.word_vectors, np.random.rand(1, embed_size)), 0)
					self.add_word(word)
					self.oov_words.add(word)

	def print_vocab_statistics(self):
		# compute number of rare word types and tokens
		rare_words = 0
		rare_tokens = 0
		for word in self.word_counts:
			if word not in self.word2idx:
				rare_words += 1
				rare_tokens += self.word_counts[word]

		# compute number of OOV word types and tokens
		oov_tokens, total_tokens = 0, 0
		for word in self.word_counts:
			total_tokens += self.word_counts[word]
			if word in self.oov_words:
				oov_tokens += self.word_counts[word]

		# compute number of utterances in training data with UNK in them
		unk_utterances = 0
		for utterance in self.tokenized_data:
			for word in utterance:
				if word not in self.word2idx:
					unk_utterances += 1
					break

		print("\n"+str(rare_words), "of", len(self.word_counts), "total word types ("+'{:.1%}'.format(rare_words/float(len(self.word_counts)))+") in the train set were treated as unk, as they did not meet the rare word threshold ("+str(self.threshold)+").")
		print("There are", rare_tokens, "total rare word tokens out of", total_tokens, "total tokens ("+'{:.1%}'.format(rare_tokens/float(total_tokens))+").")
		print("Added", len(self.oov_words), "OOV word types to vocabulary, out of a total of", len(self.word_counts), "word types in the train set ("+'{:.1%}'.format(len(self.oov_words)/float(len(self.word_counts)))+").")
		print("There are", oov_tokens, "total OOV tokens out of", total_tokens, "total tokens ("+'{:.1%}'.format(oov_tokens/float(total_tokens))+").\n")
		print("OOV words:", self.oov_words, "\n")
		print("There are", unk_utterances, "UNK utterances out of", len(self.tokenized_data), "total utterances ("+'{:.1%}'.format(unk_utterances/float(len(self.tokenized_data)))+").\n")

	def add_word(self, word):
		""" Adds a word to the vocabulary. """
		idx = len(self.word2idx)
		self.word2idx[word] = idx
		self.idx2word[idx] = word

	def __call__(self, word):
		""" Gets a word's index. If the word doesn't exist in the vocabulary, returns the index of the unk token. """
		if not word in self.word2idx:
			return self.word2idx['<unk>']
		return self.word2idx[word]

	def __len__(self):
		""" Returns size of the vocabulary. """
		return len(self.word2idx)

	def __str__(self):
		""" Prints vocabulary in human-readable form. """
		vocabulary = ""
		for key in self.idx2word.keys():
			vocabulary += str(key) + ": " + self.idx2word[key] + "\n"
		return vocabulary

def write_train_word_counts(vocab_path, word_counts, oov_words, threshold):
	sorted_word_counts = sorted(word_counts.items(), key=lambda k_v: k_v[1], reverse=True)
	with open(os.path.join('../vocabulary', vocab_path, 'word_counts.txt'), 'w') as f:
		for key, value in sorted_word_counts:
			suffix = 'unk' if value < threshold else 'oov' if key in oov_words else ''
			f.write(key.ljust(30)+str(value).ljust(10)+suffix+'\n')
	print("Wrote word counts to", os.path.join('../vocabulary', vocab_path, 'word_counts.txt'), '\n')

def main(args):
	""" Creates a vocabulary according to specified arguments and saves it to disk. """
	if not os.path.isdir('../vocabulary'):
		os.makedirs('../vocabulary')

	# create vocabulary filename based on parameter settings
	if args.vocab_name is None:
		lower = "-lower" if args.lower else ""
		threshold = "-"+str(args.threshold)+"r" if args.threshold > 0 else ""
		speaker_tokens = "-speaker" if args.use_speaker_tokens else ""
		action_tokens = "-builder_actions" if args.use_builder_action_tokens else ""
		oov_as_unk = "-oov_as_unk" if args.oov_as_unk else ""
		all_splits = "-all_splits" if args.all_splits else "-train_split"
		architect_only = "-architect_only" if not args.add_builder_utterances and not args.builder_utterances_only else ""
		builder_utterances_only = "-builder_only" if args.builder_utterances_only else ""

		if args.vector_filename is None:
			args.vocab_name = "no-embeddings-"+str(args.embed_size)+"d"
		else:
			args.vocab_name = args.vector_filename.split("/")[-1].replace('.txt','').replace('.bin.gz','')

		args.vocab_name += lower+threshold+speaker_tokens+action_tokens+oov_as_unk+all_splits+architect_only+builder_utterances_only+"/"

	args.vocab_name = os.path.join(args.base_vocab_dir, args.vocab_name)

	if not os.path.exists(args.vocab_name):
		os.makedirs(args.vocab_name)

	# logger
	sys.stdout = Logger(os.path.join(args.vocab_name, 'vocab.log'))
	print(args)

	# create the vocabulary
	vocabulary = Vocabulary(data_path=args.data_path, vector_filename=args.vector_filename,
		embed_size=args.embed_size, use_speaker_tokens=args.use_speaker_tokens, use_builder_action_tokens=args.use_builder_action_tokens,
		add_words=not args.oov_as_unk, lower=args.lower, threshold=args.threshold, all_splits=args.all_splits, add_builder_utterances=args.add_builder_utterances,
		builder_utterances_only=args.builder_utterances_only)

	if args.verbose:
		print(vocabulary)

	write_train_word_counts(args.vocab_name, vocabulary.word_counts, vocabulary.oov_words, vocabulary.threshold)

	# save the vocabulary to disk
	print("Saving the vocabulary ...")
	with open(os.path.join(args.vocab_name, 'vocab.pkl'), 'wb') as f:
		pickle.dump(vocabulary, f)
		print("Saved the vocabulary to '%s'" %os.path.realpath(f.name))

	print("Total vocabulary size: %d" %len(vocabulary))

	print("\nSaving git commit hashes ...\n")

	sys.stdout = sys.__stdout__

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--vocab_name', type=str, nargs='?', default=None,
		help='directory for saved vocabulary wrapper -- auto-generated from embeddings file name if not provided')
	parser.add_argument('--base_vocab_dir', type=str, default='../../data/vocabulary/', help='location for all saved vocabulary files')
	parser.add_argument('--vector_filename', type=str, default='../../data/glove.42B.300d.txt')
	parser.add_argument('--embed_size', type=int, default=300, help='size of word embeddings')
	parser.add_argument('--data_path', type=str, default='../../data/logs/', help='path for training data files')
	parser.add_argument('--oov_as_unk', default=False, action='store_true', help='do not add oov words to the vocabulary (instead, treat them as unk tokens)')
	parser.add_argument('--lower', default=False, action='store_true',  help='lowercase tokens in the dataset')
	parser.add_argument('--use_speaker_tokens', default=False, action='store_true', help='use speaker tokens <Architect> </Architect> and <Builder> </Builder> instead of sentence tokens <s> and </s>')
	parser.add_argument('--use_builder_action_tokens', default=False, action='store_true', help='use builder action tokens for pickup/putdown actions, e.g. <builder_pickup_red> and <builder_putdown_red>')
	parser.add_argument('--threshold', type=int, default=1, help='rare word threshold for oov words in the train set')
	parser.add_argument('--all_splits', default=False, action='store_true',  help='whether or not to use val and test data as well')
	parser.add_argument('--add_builder_utterances', default=False, action='store_true',  help='whether or not to include builder utterances')
	parser.add_argument('--builder_utterances_only', default=False, action='store_true',  help='whether or not to store only builder utterances')
	parser.add_argument('--verbose', action='store_true', help='print vocabulary in plaintext to console')
	parser.add_argument('--seed', type=int, default=1234, help='random seed')

	args = parser.parse_args()
	initialize_rngs(args.seed, torch.cuda.is_available())
	print('args.vector_filename', args.vector_filename)
	main(args)
