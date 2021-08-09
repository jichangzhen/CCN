import sys, logging, pickle
import tensorflow as tf
from sig_model import Model
from argparser import model_opts
from data_helper import WordTable, Vocabulary

sys.path.append("..")


def load_data_dict(file):
	def load_data(file):
		with open(file, 'rb') as pkl_file:
			return pickle.load(pkl_file)
		
	_, _, test_data = load_data(file)
	return test_data


def tokenid_to_sentenceid(tokenids):
	tokenids = [x for x in tokenids if x not in (0, 1)]
	min_eos_index = tokenids.index(2) if 2 in tokenids else -1
	
	if min_eos_index > 0:
		tokenids = tokenids[:min_eos_index]
	return tokenids


def padding_batch(data_set):
	tf_data_set = tf.data.Dataset.from_generator(lambda: data_set, (
		tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32,
		tf.int32)). \
		padded_batch(1, padded_shapes=(
		tf.TensorShape([None, None]),
		tf.TensorShape([None]),
		tf.TensorShape([]),
		tf.TensorShape([None]),
		tf.TensorShape([None, None]),
		tf.TensorShape([None]),
		tf.TensorShape([]),
		tf.TensorShape([None]),
		tf.TensorShape([None]),
		tf.TensorShape([None]),
		tf.TensorShape([])),
	                 padding_values=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
	valid_iterator = tf_data_set.make_one_shot_iterator()
	one_batch = valid_iterator.get_next()
	return one_batch


def one_step(session, one_batch, model, version, max_decoder_steps, dropout_keep_prob, train=True):
	input_x_batch, input_role_batch, input_sample_lens_batch, input_sentences_lens_batch, \
	similar_batch, similar_role_batch, similar_lens_batch, similar_sentences_lens_batch, \
	decoder_input_x_batch, decoder_output_x_batch, decoder_lens_batch = session.run(one_batch)
	if version != 1:
		decoder_input_x_batch = tf.keras.preprocessing.sequence.pad_sequences(decoder_input_x_batch,
		                                                                      maxlen=max_decoder_steps,
		                                                                      padding="post",
		                                                                      truncating="post",
		                                                                      value=0)
		decoder_output_x_batch = tf.keras.preprocessing.sequence.pad_sequences(decoder_output_x_batch,
		                                                                       maxlen=max_decoder_steps,
		                                                                       padding="post",
		                                                                       truncating="post",
		                                                                       value=0)
	feed_dict = {model.input_x: input_x_batch,
	             model.input_role: input_role_batch,
	             model.input_sample_lens: input_sample_lens_batch,
	             model.input_sentences_lens: input_sentences_lens_batch,
	             model.similar_x: similar_batch,
	             model.similar_role: similar_role_batch,
	             model.similar_sample_lens: similar_lens_batch,
	             model.similar_sentences_lens: similar_sentences_lens_batch,
	             model.decoder_inputs: decoder_input_x_batch,
	             model.decoder_outputs: decoder_output_x_batch,
	             model.decoder_lengths: decoder_lens_batch,
	             model.dropout_keep_prob: dropout_keep_prob,
	             model.training: train
	             }
	return feed_dict, decoder_output_x_batch


def train(args):
	vocab = Vocabulary()
	vocab.load(args.vocab_file, keep_words=args.vocab_size)
	test_data_set = load_data_dict(args.data_file)
	length = len(test_data_set)
	
	f1 = open(args.pre_file, 'w', encoding="utf-8")
	f2 = open(args.gro_file, 'w', encoding="utf-8")
	session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	session_conf.gpu_options.allow_growth = True
	session_conf = tf.ConfigProto()
	session_conf.gpu_options.allow_growth = True
	
	with tf.Session(graph=tf.Graph(), config=session_conf) as session:
		model = Model(args)
		logging.info(args)
		saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=args.num_checkpoints)
		
		initialize_op = tf.variables_initializer(
			[x for x in tf.global_variables() if x not in tf.trainable_variables()])
		session.run(initialize_op)
		saver.restore(session, args.checkpoint_path)
		train_one_batch = padding_batch(test_data_set)
		
		for batch_id in range(length):
			feed_dict, decoder_output_x_batch = one_step(session, train_one_batch, model, args.use_copy_version,
			                                             args.max_decoder_steps, args.dropout_keep_prob, train=True)
			
			fetches = [model.loss, model.decoder_loss, model.infer_predicts]
			loss, decoder_loss, batch_seq2seq_predict = session.run(fetches=fetches, feed_dict=feed_dict)
			print("The encoder is decoding......")
			for i in batch_seq2seq_predict:
				v0 = vocab.do_decode(i)
				# print(v0)
				for j in v0:
					if j == '</S>':
						break
					# print(j, end=' ')
					f1.write(j + '')
			# print('\n')
			f1.write('\n')
			for i in decoder_output_x_batch:
				v0 = vocab.do_decode(i)
				# print(v0)
				for j in v0:
					if j == '</S>':
						break
					# print(j, end=' ')
					f2.write(j + '')
			# print('\n')
			f2.write('\n')
		f1.close()
		f2.close()


def main(argv=None):
	args = model_opts()
	args.pre_word_embeddings = WordTable(args.word_emb_file, args.embedding_dim, args.vocab_size).embeddings
	train(args)


if __name__ == '__main__':
	tf.app.run()
