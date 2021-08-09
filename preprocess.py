import sys, pickle
import tensorflow as tf
import numpy as np
import pandas as pd
from data_helper import Vocabulary
from argparser import model_opts

sys.path.append("..")


def load_data(args):
	vocab = Vocabulary()
	vocab.load(args.vocab_file, keep_words=args.vocab_size)
	
	df_train = pd.read_csv(args.train_data_file, sep="\t")
	df_train.fillna(value="", inplace=True)
	print("train:", df_train.shape)
	
	df_dev = pd.read_csv(args.dev_data_file, sep="\t")
	df_dev.fillna(value="", inplace=True)
	print("dev:", df_dev.shape)
	
	df_test = pd.read_csv(args.test_data_file, sep="\t")
	df_test.fillna(value="", inplace=True)
	print("test:", df_test.shape)
	
	df_train_sim = pd.read_csv(args.train_sim_file, sep="\t")
	df_train_sim.fillna(value="", inplace=True)
	print("train_sim:", df_train_sim.shape)
	
	df_dev_sim = pd.read_csv(args.dev_sim_file, sep="\t")
	df_dev_sim.fillna(value="", inplace=True)
	print("dev_sim:", df_dev_sim.shape)
	
	df_test_sim = pd.read_csv(args.test_sim_file, sep="\t")
	df_test_sim.fillna(value="", inplace=True)
	print("test_sim:", df_test_sim.shape)
	
	def _do_vectorize(df, name):
		df = df.copy()
		df["sentence"] = df["sentence"].map(eval)
		grouped = df.groupby("doc")
		
		sentence_nums = []
		sentence_cut_words = []
		sentence_word_ids = []
		sentences_lens = []
		roles = []
		
		zero = pd.Series([0])
		for agg_name, agg_df in grouped:
			if args.padding_data:
				sentence_nums.append(args.max_sentence_num)
				role = agg_df["role"]
				if len(role) >= args.max_sentence_num:
					roles.append(role[:args.max_sentence_num])
				else:
					for i in range(args.max_sentence_num - len(agg_df["role"])):
						role = role.append(zero)
					roles.append(role)
			
			else:
				if args.intercept and name:
					if len(agg_df) >= args.max_sentence_num:
						sentence_nums.append(args.max_sentence_num)
						roles.append(agg_df["role"][-args.max_sentence_num:])
					else:
						sentence_nums.append(len(agg_df))
						roles.append(agg_df["role"])
				else:
					sentence_nums.append(len(agg_df))
					roles.append(agg_df["role"])
			
			tmp_words = []
			i = 0
			for words in agg_df["sentence"]:
				i += 1
				if len(words) <= args.max_sequence_length:
					tmp_words.append(words)
				else:
					tmp_words.append(words[:args.max_sequence_length])
				if args.padding_data:
					if i == args.max_sentence_num:
						break
			if args.intercept and name:
				if len(tmp_words) > args.max_sentence_num:
					tmp_words = tmp_words[-args.max_sentence_num:]
			
			if args.padding_data:
				# sentences_lens.append([args.max_sequence_length for x in tmp_words])
				sentences_lens.append([args.max_sequence_length for i in range(args.max_sentence_num)])
			
			
			else:
				sentences_lens.append([len(x) for x in tmp_words])
			sentence_cut_words.append(tmp_words)
			word_ids = [vocab.do_encode(x)[0] for x in tmp_words]
			
			if args.padding_data:
				if len(word_ids) < args.max_sentence_num:
					for i in range(args.max_sentence_num - len(word_ids)):
						word_ids.append([0])
			
			word_ids = tf.keras.preprocessing.sequence.pad_sequences(word_ids,
			                                                         maxlen=args.max_sequence_length,
			                                                         padding="post",
			                                                         truncating="post",
			                                                         value=0)
			assert np.max(word_ids) < args.vocab_size
			assert np.max(agg_df["role"]) < 6
			sentence_word_ids.append(word_ids)
		
		return sentence_word_ids, roles, sentence_nums, sentences_lens
	
	def _do_label_vectorize(df):
		df = df.copy()
		df.index = range(len(df))
		df["sentence"] = df["sentence"].map(eval)
		grouped = df.groupby("doc")
		
		decoder_input_word_ids = []
		decoder_output_word_ids = []
		decoder_sentence_lens = []
		
		for agg_name, agg_df in grouped:
			question = {x for x in agg_df["question"]}
			question_text = question.pop()
			cut_words = eval(question_text)
			decoder_input_word_ids.append(
				vocab.do_encode(cut_words, mode="bos")[0]
			)
			decoder_output_word_ids.append(
				vocab.do_encode(cut_words, mode="eos")[0]
			)
			decoder_sentence_lens.append(
				len(cut_words) + 1
			)
		return decoder_input_word_ids, decoder_output_word_ids, decoder_sentence_lens
	
	train_sentence_word_ids, train_roles, train_sentence_nums, train_sentences_lens = _do_vectorize(df_train, name=True)
	dev_sentence_word_ids, dev_roles, dev_sentence_nums, dev_sentences_lens = _do_vectorize(df_dev, name=True)
	test_sentence_word_ids, test_roles, test_sentence_nums, test_sentences_lens = _do_vectorize(df_test, name=True)
	
	train_similar_word_ids, train_similar_roles, train_similar_nums, train_similar_lens = _do_vectorize(df_train_sim, name=True)
	dev_similar_word_ids, dev_similar_roles, dev_similar_nums, dev_similar_lens = _do_vectorize(df_dev_sim, name=True)
	test_similar_word_ids, test_similar_roles, test_similar_nums, test_similar_lens = _do_vectorize(df_test_sim, name=True)
	
	train_decoder_input_word_ids, train_decoder_output_word_ids, train_decoder_sentence_lens = _do_label_vectorize(df_train)
	dev_decoder_input_word_ids, dev_decoder_output_word_ids, dev_decoder_sentence_lens = _do_label_vectorize(df_dev)
	test_decoder_input_word_ids, test_decoder_output_word_ids, test_decoder_sentence_lens = _do_label_vectorize(df_test)
	
	with open(args.data_file, 'wb') as pkl_file:
		data = [
			list(zip(
				train_sentence_word_ids, train_roles, train_sentence_nums, train_sentences_lens,
				train_similar_word_ids, train_similar_roles, train_similar_nums, train_similar_lens,
				train_decoder_input_word_ids, train_decoder_output_word_ids, train_decoder_sentence_lens)),
			list(zip(
				dev_sentence_word_ids, dev_roles, dev_sentence_nums, dev_sentences_lens,
				dev_similar_word_ids, dev_similar_roles, dev_similar_nums, dev_similar_lens,
				dev_decoder_input_word_ids, dev_decoder_output_word_ids, dev_decoder_sentence_lens)),
			list(zip(
				test_sentence_word_ids, test_roles, test_sentence_nums, test_sentences_lens,
				test_similar_word_ids, test_similar_roles, test_similar_nums, test_similar_lens,
				test_decoder_input_word_ids, test_decoder_output_word_ids, test_decoder_sentence_lens))
		]
		pickle.dump(data, pkl_file)
	return data


if __name__ == '__main__':
	args = model_opts()
	load_data(args)