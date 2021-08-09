import datetime, os, sys, pickle
import tensorflow as tf
import numpy as np
from model import Model
from optimization import create_optimizer
from argparser import model_opts
from data_helper import WordTable

sys.path.append("..")

def load_data_dict(file):
	def load_data(file):
		with open(file, 'rb') as pkl_file:
			return pickle.load(pkl_file)
		
	train_data, val_data, _ = load_data(file)
	return train_data, val_data


def tokenid_to_sentenceid(tokenids):
	tokenids = [x for x in tokenids if x not in (0, 1)]
	min_eos_index = tokenids.index(2) if 2 in tokenids else -1
	
	if min_eos_index > 0:
		tokenids = tokenids[:min_eos_index]
	return tokenids


def padding_batch(data_set, num_epochs, batch_size, mode='train'):
	if mode == 'train':
		tf_train_data_set = tf.data.Dataset.from_generator(lambda: data_set, (
			tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32,
			tf.int32)). \
			shuffle(len(data_set)).repeat(num_epochs).padded_batch(batch_size, padded_shapes=(
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
		train_iterator = tf_train_data_set.make_one_shot_iterator()
		one_batch = train_iterator.get_next()
	
	else:
		tf_data_set = tf.data.Dataset.from_generator(lambda: data_set, (
			tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32,
			tf.int32)). \
			padded_batch(batch_size, padded_shapes=(
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
	content_x_batch, content_role_batch, content_sample_lens_batch, content_sentences_lens_batch, \
	sentence_batch, sentence_role_batch, sentence_lens_batch, sentence_sentences_lens_batch, \
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
	feed_dict = {model.content_x: content_x_batch,
	             model.content_role: content_role_batch,
	             model.content_sample_lens: content_sample_lens_batch,
	             model.content_sentences_lens: content_sentences_lens_batch,
	             model.sentence_x: sentence_batch,
	             model.sentence_role: sentence_role_batch,
	             model.sentence_sample_lens: sentence_lens_batch,
	             model.sentence_sentences_lens: sentence_sentences_lens_batch,
	             model.decoder_inputs: decoder_input_x_batch,
	             model.decoder_outputs: decoder_output_x_batch,
	             model.decoder_lengths: decoder_lens_batch,
	             model.dropout_keep_prob: dropout_keep_prob,
	             model.training: train
	             }
	return feed_dict


def dev_step(session, data_set, model, batch_size, max_decoder_steps, version, dropout_keep_prob):
	num_samples = len(data_set)
	div = num_samples % batch_size
	batch_num = num_samples // batch_size + 1 if div != 0 else num_samples // batch_size
	valid_one_batch = padding_batch(data_set, div, batch_size, mode='dev')
	losses = []
	decoder_losses = []
	for _ in range(batch_num):
		feed_dict, decoder_output_x_batch = one_step(session, valid_one_batch, model, version, max_decoder_steps,
		                                             dropout_keep_prob, train=False)
		
		fetches = [model.loss, model.decoder_loss]
		loss, decoder_loss = session.run(fetches=fetches, feed_dict=feed_dict)
		
		losses.append(loss)
		decoder_losses.append(decoder_loss)
	
	mean_loss = np.mean(losses)
	mean_decoder_loss = np.mean(decoder_losses)

	return mean_loss, mean_decoder_loss


def train(args):
	train_data, val_data = load_data_dict(args.data_file)
	train_num_samples = len(train_data)
	batch_num = (train_num_samples * args.num_epochs) // args.batch_size + 1
	
	session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	session_conf.gpu_options.allow_growth = True
	session_conf = tf.ConfigProto()
	session_conf.gpu_options.allow_growth = True
	
	with tf.Session(graph=tf.Graph(), config=session_conf) as session:
		model = Model(args)
		train_op, learning_rate, global_step = create_optimizer(model.loss,
		                                                        args.learning_rate,
		                                                        num_train_steps=batch_num,
		                                                        num_warmup_steps=int(
			                                                        batch_num * args.warm_up_steps_percent),
		                                                        use_tpu=False)
		
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		
		timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
		out_dir = os.path.abspath(os.path.join(os.path.curdir, "cross", timestamp))
		print("Writing to {}\n".format(out_dir))
		
		checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
		checkpoint_prefix = os.path.join(checkpoint_dir, "model")
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		
		saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=args.num_checkpoints)
		
		if args.continue_training:
			initialize_op = tf.variables_initializer(
				[x for x in tf.global_variables() if x not in tf.trainable_variables()])
			session.run(initialize_op)
			saver.restore(session, args.checkpoint_path)
		else:
			session.run(tf.global_variables_initializer())
		
		saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=args.num_checkpoints)
		train_one_batch = padding_batch(train_data, args.num_epochs, args.batch_size)
		
		minloss = 99999
		for batch_id in range(batch_num):
			feed_dict = one_step(session, train_one_batch, model, args.use_copy_version,
			                      args.max_decoder_steps, args.dropout_keep_prob, train=True)
			
			fetches = [update_ops, train_op, learning_rate, global_step, model.loss, model.decoder_loss,
			           model.infer_predicts]
			
			_, _, lr, step, loss, decoder_loss, batch_seq2seq_predict = session.run(fetches=fetches,
			                                                                        feed_dict=feed_dict)

			time_str = datetime.datetime.now().isoformat()
			print("\n{}:step {}, lr {:g}, loss {:g}, decoder_loss {:g}".format(
				time_str, step, lr, np.mean(loss), np.mean(decoder_loss)))

			current_step = tf.train.global_step(session, global_step)
			
			if current_step % args.checkpoint_every == 0:
				valid_loss, val_dec_loss = dev_step(session, val_data, model,
				                                    args.batch_size, args.max_decoder_steps,
				                                    args.use_copy_version, args.dropout_keep_prob)
				vloss = valid_loss + val_dec_loss
				minloss = min(minloss, vloss)
				
				if minloss == vloss:
					path = saver.save(session, checkpoint_prefix, global_step=current_step)
					print("Saved model checkpoint to {}\n".format(path))


def main(argv=None):
	args = model_opts()
	args.pre_word_embeddings = WordTable(args.word_emb_file, args.embedding_dim, args.vocab_size).embeddings
	train(args)


if __name__ == '__main__':
	tf.app.run()
