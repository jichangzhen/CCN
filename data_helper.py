from collections import Counter
import jieba.posseg as pseg
import re, itertools, jieba, glob, random
from tqdm import tqdm
import pandas as pd
from collections import OrderedDict
import numpy as np
from multiprocessing.pool import Pool

UNK = '<UNK>'
BOS = '<S>'
EOS = '</S>'
PAD = '<PAD>'
KEEP_VOCAB_SIZE = 100000


class NodeVocab(object):
	def __init__(self):
		self.node_size = 0
		self.node2id = {}
		self.id2node = []
		self.pad_id = 0
		self.bos_id = 1
		self.eos_id = 2
		self.unk_id = 3
	
	def node_to_id(self, node):
		return self.node2id.get(node, self.unk_id)
	
	def id_to_node(self, cur_id):
		return self.id2node[cur_id]
	
	def do_encode(self, path, mode=None):
		"""

		:param path: list of nodes
		:param mode: bos, eos, bos_eos
		:return:
		"""
		ids = [self.node_to_id(w) for w in path]
		
		if mode == "bos":
			ids = [self.bos_id] + ids
		elif mode == "eos":
			ids = ids + [self.eos_id]
		elif mode == "bos_eos":
			ids = [self.bos_id] + ids + [self.eos_id]
		
		sen_len = len(ids)
		return ids, sen_len
	
	def load(self, path):
		self.id2node.clear()
		self.node2id.clear()
		i = 0
		for f in open(path, "r", encoding="utf-8"):
			node = f.strip()
			self.id2node.append(node)
			self.node2id[node] = i
			i += 1
		
		self.node_size = len(self.id2node)
		
		print("current node size %d" % self.node_size)
	
	def __len__(self):
		return len(self.id2node)
	
	def __getitem__(self, word):
		return self.node2id.get(word, self.unk_id)
	
	def __contains__(self, word):
		return word in self.node2id


class Vocabulary(object):
	def __init__(self):
		self.vocab_size = 0
		self.id2word = []
		self.word2id = {}
		self.tf = {}
		self.idf = {}
		self.tfidf = {}
		self.pad_id = 0
		self.bos_id = 1
		self.eos_id = 2
		self.unk_id = 3
		self.pad = PAD
		self.bos = BOS
		self.eos = EOS
		self.unk = UNK
	
	def build(self, corpus, max_vocab_size=50000):
		"""

		:param corpus: tokenized_sentences
		:param max_vocab_size:
		:return:
		"""
		# tf-idf
		counter = Counter(itertools.chain(*corpus))
		self.tf = {x[0]: x[1] for x in counter.most_common(len(counter))}
		
		document_freq_dict = {}
		doc_num = len(corpus)
		for doc in corpus:
			word_set = set(doc)
			for word in word_set:
				document_freq_dict[word] = document_freq_dict.get(word, 0) + 1
		
		for word in self.tf:
			self.idf[word] = np.log(doc_num / document_freq_dict.get(word, 1))
			self.tfidf[word] = self.tf[word] * self.idf[word]
		
		# frequent words
		items = counter.most_common(max_vocab_size)
		self.id2word = [self.pad] + [self.bos] + [self.eos] + [self.unk] + [x[0] for x in items if x[0].strip()]
		self.word2id = dict([(w, i) for i, w in enumerate(self.id2word)])
		self.vocab_size = len(self.word2id)
		
		freq_list = list(self.tf.values())
		total_counts = sum(freq_list)
		cur_counts = sum([x[1] for x in items])
		percent = cur_counts / total_counts * 100
		
		percent_95_to_cover = 0.95
		percent_98_to_cover = 0.98
		ss = pd.Series(freq_list)
		min_count_10 = max(ss[ss >= 10].index)
		min_count_5 = max(ss[ss >= 5].index)
		percent_95_num = min(ss[(ss.cumsum() / total_counts) >= percent_95_to_cover].index)
		percent_98_num = min(ss[(ss.cumsum() / total_counts) >= percent_98_to_cover].index)
		print(
			"""
			Vocab Info:
			%d word in total
			%.2f freq stored in vocab
			%d word cover %.2f freq
			%d word cover %.2f freq
			min word frequency 10 need %d words
			min word frequency 5  need %d words
			current vocab len %d, min freq %d
			""" % (
				len(counter), percent, percent_95_num, percent_95_to_cover,
				percent_98_num, percent_98_to_cover,
				min_count_10, min_count_5, self.vocab_size, counter[self.id2word[-1]]
			))
	
	def word_to_id(self, word):
		return self.word2id.get(word, self.unk_id)
	
	def id_to_word(self, cur_id):
		return self.id2word[cur_id]
	
	def do_encode(self, sentence, mode=None):
		"""

		:param sentence: list of words
		:param mode: bos, eos, bos_eos
		:return:
		"""
		ids = [self.word_to_id(w) for w in sentence]
		
		if mode == "bos":
			ids = [self.bos_id] + ids
		elif mode == "eos":
			ids = ids + [self.eos_id]
		elif mode == "bos_eos":
			ids = [self.bos_id] + ids + [self.eos_id]
		
		sen_len = len(ids)
		return ids, sen_len
	
	def do_decode(self, tokenids):
		"""

		:param tokenids:
		:return:
		"""
		return [self.id_to_word(x) for x in tokenids]
	
	def do_decode_to_natural_sentence(self, tokenids):
		"""

		:param tokenids:
		:return:
		"""
		tokenids = [x for x in tokenids if x not in (self.pad_id, self.bos_id)]
		min_eos_index = tokenids.index(self.eos_id) if self.eos_id in tokenids else -1
		
		if min_eos_index > 0:
			tokenids = tokenids[:min_eos_index]
		
		return self.do_decode(tokenids)
	
	def write_vocab(self, path):
		with open(path, "w", encoding="utf-8") as f:
			for item in self.id2word:
				print(item, file=f)
	
	def dump(self, path):
		with open(path, "w", encoding="utf-8") as f:
			for word in self.id2word:
				s = "%s\t%d\t%.6f\t%.6f\n" % (
					word, self.tf.get(word, 0), self.idf.get(word, 0), self.tfidf.get(word, 0)
				)
				f.write(s)
	
	def load(self, path, keep_words=160000):
		self.id2word.clear()
		self.word2id.clear()
		self.tf.clear()
		self.idf.clear()
		self.tfidf.clear()
		
		with open(path, "r", encoding="utf-8") as f:
			for i, line in enumerate(f):
				s = line.strip().split("\t")
				if len(s) == 4:
					word, tf, idf, tfidf = line.strip().split("\t")
					tf, idf, tfidf = int(tf), float(idf), float(tfidf)
					self.id2word.append(word)
					self.word2id[word] = i
					self.tf[word] = tf
					self.idf[word] = idf
					self.tfidf[word] = tfidf
					if i == keep_words - 1:
						break
				elif len(s) == 1:
					word = line.strip()
					self.id2word.append(word)
					self.word2id[word] = i
					if i == keep_words - 1:
						break
				elif len(s) == 2:
					word, tf = line.strip().split("\t")
					tf = int(tf)
					self.id2word.append(word)
					self.word2id[word] = i
					self.tf[word] = tf
					if i == keep_words - 1:
						break
		
		self.vocab_size = len(self.id2word)
		
		# assert self.id2word[self.pad_id] == self.pad
		# assert self.id2word[self.bos_id] == self.bos
		# assert self.id2word[self.eos_id] == self.eos
		# assert self.id2word[self.unk_id] == self.unk
		
		print("current vocab size %d" % self.vocab_size)
	
	def __len__(self):
		return len(self.id2word)
	
	def __getitem__(self, word):
		return self.word2id.get(word, self.unk_id)
	
	def __contains__(self, word):
		return word in self.word2id


class Tokenizer(object):
	def __init__(self):
		pass
	
	@staticmethod
	def cut(text):
		return list(jieba.cut(text))


class DataHelper(object):
	def __init__(self, vocabulary):
		self.ptr = 0
		self.vocab = vocabulary
		self.tokenizer = Tokenizer()
	
	def text_vectorize(self, texts):
		with Pool(4) as pool:
			cut_words_list = pool.map(self.tokenizer.cut, tqdm(texts))
		pool.join()
		
		encode_ids = []
		lens = []
		for cut_words in cut_words_list:
			e, l = self.vocab.do_encode(cut_words)
			encode_ids.append(e)
			lens.append(l)
		
		return encode_ids, lens


class NodeTable(object):
	def __init__(self, embedding_path, vocab_path, edim):
		self.edim = edim
		self.embeddings, self.dictionary = self._load_vectors(embedding_path, vocab_path, self.edim)
	
	def __getitem__(self, item):
		if item in self.dictionary:
			return self.embeddings[self.dictionary[item], :]
		else:
			return self.embeddings[self.dictionary[UNK], :]
	
	def __contains__(self, item):
		return item in self.dictionary
	
	def unk_embedding(self):
		return self[UNK]
	
	def embedding_look_up(self, words):
		"""
		:return unk embedding if a word is not in vocabulary.
		:param words:
		:return: [timesteps, edim]
		"""
		embeddings = []
		for word in words:
			embeddings.append(
				self[word]
			)
		return np.array(embeddings)
	
	def embedding_look_up_iov(self, words):
		"""
		inside of vocabulary
		:param words:
		:return:
		"""
		embeddings = []
		for word in words:
			if word in self.dictionary:
				embeddings.append(self[word])
		return np.array(embeddings) if embeddings != [] else np.array([self.unk_embedding()])
	
	def _load_vectors(self, epath, vpath, dim):
		embeddings = []
		dictionary = OrderedDict()
		invalid = 0
		word = [0] * 300
		for i, line in enumerate(open(vpath, "r", encoding="utf-8")):
			word[i] = line.strip()
		
		for j, line in tqdm(enumerate(open(epath, "r", encoding="utf-8"))):
			line = line.strip().split()
			vector = [float(val) for val in line]
			
			if len(vector) != dim:
				invalid += 1
				raise TypeError("wrong word embedding dim: %d" % len(vector))
			if word[j] not in dictionary:
				dictionary.setdefault(word[j], j)
				embeddings.append(vector)
		
		embeddings = np.array(embeddings, np.float32)
		print("{0} embeddings loaded. embedding shape ({1},{2})".format(len(dictionary), embeddings.shape[0],
		                                                                embeddings.shape[1]))
		return embeddings, dictionary


class WordTable(object):
	def __init__(self, embedding_path, edim, keep_words=160000):
		self.edim = edim
		self.embeddings, self.dictionary = self._load_vectors(embedding_path, self.edim, keep_words=keep_words)
	
	def __getitem__(self, item):
		if item in self.dictionary:
			return self.embeddings[self.dictionary[item], :]
		else:
			return self.embeddings[self.dictionary[UNK], :]
	
	def __contains__(self, item):
		return item in self.dictionary
	
	def unk_embedding(self):
		return self[UNK]
	
	def embedding_look_up(self, words):
		"""
		:return unk embedding if a word is not in vocabulary.
		:param words:
		:return: [timesteps, edim]
		"""
		embeddings = []
		for word in words:
			embeddings.append(
				self[word]
			)
		return np.array(embeddings)
	
	def embedding_look_up_iov(self, words):
		"""
		inside of vocabulary
		:param words:
		:return:
		"""
		embeddings = []
		for word in words:
			if word in self.dictionary:
				embeddings.append(self[word])
		return np.array(embeddings) if embeddings != [] else np.array([self.unk_embedding()])
	
	@staticmethod
	def _load_vectors(path, dim, keep_words=160000):
		embeddings = []
		dictionary = OrderedDict()
		invalid = 0
		i = 0
		for n, line in tqdm(enumerate(open(path, "r", encoding="utf-8"))):
			line = line.split("\t")
			word = line[0]
			vector = [float(val) for val in line[1].split()]
			
			if len(vector) != dim:
				invalid += 1
				raise TypeError("wrong word embedding dim: %d" % len(vector))
			
			if word not in dictionary:
				dictionary.setdefault(word, i)
				embeddings.append(vector)
				i += 1
				if i == keep_words:
					break
		
		embeddings = np.array(embeddings, np.float32)
		print("{0} embeddings loaded. embedding shape ({1},{2})".format(len(dictionary), embeddings.shape[0],
		                                                                embeddings.shape[1]))
		return embeddings, dictionary


class CustomSegmentor(object):
	def __init__(self):
		# 匹配英文数字组合，或者纯粹数字组合
		self.alnum_pattern = re.compile("[0-9]+|[0-9]+[a-zA-Z]+|[a-zA-Z]+[0-9]+")
	
	def cut(self, text, clean=True):
		words = pseg.cut(text)
		
		clean_words = []
		for word, pos in words:
			if clean:
				word = self.clean_rule(word, pos)
			clean_words.append(word)
		return clean_words
	
	def clean_rule(self, word, pos):
		if re.match(self.alnum_pattern, word) is not None:
			return "alnum"
		elif pos in ("nr", "nrfg"):
			return "personname"
		else:
			return word


class IterDataset(object):
	"""
	Hold a iter data dataset.

	"""
	
	def __init__(self, filepattern, vocab, nepochs=3, test=False, shuffle_on_load=False,
	             tokenizer=None):
		'''
		filepattern = a glob string that specifies the list of files.
		vocab = an instance of Vocabulary or UnicodeCharsVocabulary
		reverse = if True, then iterate over tokens in each sentence in reverse
		test = if True, then iterate through all data once then stop.
			Otherwise, iterate forever.
		shuffle_on_load = if True, then shuffle the sentences after loading.
		'''
		
		self._vocab = vocab
		self._nepochs = nepochs
		self._all_shards = glob.glob(filepattern)
		print('Found %d shards at %s' % (len(self._all_shards), filepattern))
		self._shards_to_choose = []
		
		self._test = test
		self._shuffle_on_load = shuffle_on_load
		
		self._tokenizer = tokenizer
		
		self._ids = self._load_random_shard()
	
	def _choose_random_shard(self):
		if len(self._shards_to_choose) == 0:
			if self._nepochs > 0:
				self._nepochs -= 1
				self._shards_to_choose = list(self._all_shards)
				random.shuffle(self._shards_to_choose)
			else:
				raise StopIteration
		shard_name = self._shards_to_choose.pop()
		return shard_name
	
	def _load_random_shard(self):
		"""Randomly select a file and read it."""
		if self._test:
			if len(self._all_shards) == 0:
				# we've loaded all the data
				# this will propogate up to the generator in get_batch
				# and stop iterating
				raise StopIteration
			else:
				shard_name = self._all_shards.pop()
		else:
			# just pick a random shard
			shard_name = self._choose_random_shard()
		
		ids = self._load_shard(shard_name)
		self._i = 0
		self._nids = len(ids)
		return ids
	
	def _load_shard(self, shard_name):
		"""Read one file and convert to ids.

		Args:
			shard_name: file path.

		Returns:
			list of (features) tuples.
		"""
		print('Loading data from: %s' % shard_name)
		shard_data = pd.read_csv(shard_name, sep="\t", engine='python')
		
		shard_array_list = self._do_vec(shard_data)
		
		if self._shuffle_on_load:
			random.shuffle(shard_array_list)
		
		print('Loaded %d samples.' % len(shard_array_list))
		print('Finished loading')
		return shard_array_list
	
	def _do_vec(self, shard_data):
		"""
		normally shard_data is DataFrame or Json
		:param shard_data:
		:return: list
		"""
		agg = shard_data.groupby("doc")
		return [agg_name for agg_name, _ in agg]
	
	# return []
	
	def get_sample(self):
		while True:
			if self._i == self._nids:
				self._ids = self._load_random_shard()
			ret = self._ids[self._i]
			self._i += 1
			yield ret