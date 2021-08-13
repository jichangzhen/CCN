import tensorflow as tf
from ops import gelu, get_shape_list, weight_noise, tensor_noise
import pgn_modeling as pgn_modeling
from bert_modeling import create_attention_mask_from_input_mask, transformer_model, get_assignment_map_from_checkpoint
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib.cudnn_rnn import CudnnCompatibleLSTMCell

VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER
Epsilon = 1e-5


class Model(object):
	
	def __init__(self, config):
		self.config = config
		self.activation_function = {
			"relu": tf.nn.relu,
			"swish": tf.nn.swish,
			"elu": tf.nn.elu,
			"crelu": tf.nn.crelu,
			"tanh": tf.nn.tanh,
			"gelu": gelu
		}[self.config.activation_function]
		self.pad_id = 0
		self.go_id = 1
		self.eos_id = 2
		self.unk_id = 3
		self.beam_width = 3
		
		with tf.name_scope("placeholder"):
			self.input_x = tf.placeholder(tf.int32, [None, None, None],
			                              name="input_x")  # batch_size, max_sentence_num, max_sequence_length
			self.input_role = tf.placeholder(tf.int32, [None, None], name="input_role")  # batch_size, max_sentence_num
			self.input_sample_lens = tf.placeholder(tf.int32, [None], name="input_sample_lens")  # batch_size
			self.input_sentences_lens = tf.placeholder(tf.int32, [None, None],
			                                           name="input_sentences_lens")  # batch_size, max_sentence_num
			
			self.similar_x = tf.placeholder(tf.int32, [None, None, None],
			                                name="similar_x")  # batch_size, max_sentence_num, max_sequence_length
			self.similar_role = tf.placeholder(tf.int32, [None, None],
			                                   name="similar_role")  # batch_size, max_sentence_num
			self.similar_sample_lens = tf.placeholder(tf.int32, [None], name="similar_sample_lens")  # batch_size
			self.similar_sentences_lens = tf.placeholder(tf.int32, [None, None],
			                                             name="similar_sentences_lens")  # batch_size, max_sentence_num
			
			self.input_sample_mask = tf.sequence_mask(self.input_sample_lens,
			                                          name="input_sample_mask")  # batch_size, max_sentence_num
			self.input_sentences_mask = tf.sequence_mask(self.input_sentences_lens,
			                                             name="input_sentences_mask")  # batch_size, max_sentence_num, max_sequence_length
			
			self.similar_sample_mask = tf.sequence_mask(self.similar_sample_lens,
			                                            name="similar_sample_mask")  # batch_size, max_sentence_num
			self.similar_sentences_mask = tf.sequence_mask(self.similar_sentences_lens,
			                                               name="similar_sentences_mask")  # batch_size, max_sentence_num, max_sequence_length
			
			self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
			self.training = tf.placeholder(tf.bool, name="bn_training")
			self.input_y = tf.placeholder(tf.int32, [None], name="input_y")
			
			batch_size, max_sentence_num, max_sequence_length = get_shape_list(self.input_x)
			_, max_similar_num, max_similar_length = get_shape_list(self.similar_x)
			self.batch_size = batch_size
			self.max_sentence_num = max_sentence_num
			self.max_sequence_length = max_sequence_length
			self.max_similar_num = max_similar_num
			self.max_similar_length = max_similar_length
		
		with tf.name_scope("embedding"):
			with tf.device("/cpu:0"):
				self.word_table = tf.Variable(self.config.pre_word_embeddings, trainable=True, dtype=tf.float32,
				                              name='word_table')
				self.word_embedding = tf.nn.embedding_lookup(self.word_table, self.input_x, name='word_embedding')
				
				if self.config.use_cross_copy:
					self.word_emb = tf.nn.embedding_lookup(self.word_table, self.similar_x, name='word_emb')
				
				if self.config.use_role_embedding:
					self.role_table = tf.Variable(tf.truncated_normal([self.config.role_num + 1, self.config.role_emb],
					                                                  stddev=self.config.init_std), trainable=True,
					                              dtype=tf.float32, name='role_table')
					self.role_embedding = tf.nn.embedding_lookup(self.role_table, self.input_role,
					                                             name='role_embedding')
					self.similar_role_embedding = tf.nn.embedding_lookup(self.role_table, self.similar_role,
					                                                     name='similar_role_embedding')
		
		with tf.variable_scope("utterance_rnn"):
			if self.config.use_role_embedding:
				tiled_role_embedding = tf.multiply(
					tf.ones([batch_size, max_sentence_num, max_sequence_length, self.config.role_emb],
					        dtype=tf.float32), tf.expand_dims(self.role_embedding, axis=2))
				self.word_embedding = tf.concat([self.word_embedding, tiled_role_embedding], axis=-1)
			
			self.word_embedding = tf.reshape(self.word_embedding,
			                                 [-1, max_sequence_length, self.word_embedding.get_shape()[-1].value])
			self.mask = tf.sequence_mask(tf.reshape(self.input_sentences_lens, [-1]), maxlen=max_sequence_length)
			self.mask = tf.cast(tf.expand_dims(self.mask, axis=-1), dtype=tf.float32)
			self.word_embedding = tf.multiply(self.word_embedding, self.mask)
			
			cell_fw = MultiRNNCell(
				[CudnnCompatibleLSTMCell(self.config.rnn_hidden_size) for _ in
				 range(self.config.rnn_layer_num)]
			)
			cell_bw = MultiRNNCell(
				[CudnnCompatibleLSTMCell(self.config.rnn_hidden_size) for _ in
				 range(self.config.rnn_layer_num)]
			)
			(output_fw, output_bw), (output_state_fw, output_state_bw) = tf.nn.bidirectional_dynamic_rnn(
				cell_fw=cell_fw, cell_bw=cell_bw, inputs=self.word_embedding,
				dtype=tf.float32, sequence_length=tf.reshape(self.input_sentences_lens, [-1])
			)
			
			# final_states = tf.concat([output_state_fw[0].h, output_state_bw[0].h], axis=1)
			utterance_memory_embeddings = tf.concat([output_fw, output_bw], axis=2)
			utterance_memory_embeddings = tf.multiply(utterance_memory_embeddings, self.mask)
			utterance_memory_embeddings = tf.nn.dropout(utterance_memory_embeddings, keep_prob=self.dropout_keep_prob,
			                                            name="utterance_memory_embeddings")
			
			self.sample_text_final_state, sen_level_att_score = self.attention_mechanism(utterance_memory_embeddings,
			                                                                             tf.squeeze(self.mask, axis=-1))
			self.sample_text_final_state = tf.reshape(self.sample_text_final_state,
			                                          [batch_size, max_sentence_num, 2 * self.config.rnn_hidden_size])
			sen_level_att_score = tf.reshape(sen_level_att_score, [batch_size, max_sentence_num, max_sequence_length])
			self.sen_level_att_score = sen_level_att_score
		
		if self.config.use_cross_copy:
			with tf.variable_scope("similar_rnn"):
				if self.config.use_role_embedding:
					sim_role_embedding = tf.multiply(
						tf.ones([batch_size, max_similar_num, max_similar_length, self.config.role_emb],
						        dtype=tf.float32), tf.expand_dims(self.similar_role_embedding, axis=2))
					self.word_emb = tf.concat([self.word_emb, sim_role_embedding], axis=-1)
				
				self.word_emb = tf.reshape(self.word_emb, [-1, max_similar_length, self.word_emb.get_shape()[-1].value])
				self.sim_mask = tf.sequence_mask(tf.reshape(self.similar_sentences_lens, [-1]),
				                                 maxlen=max_similar_length)
				self.sim_mask = tf.cast(tf.expand_dims(self.sim_mask, axis=-1), dtype=tf.float32)
				self.word_emb = tf.multiply(self.word_emb, self.sim_mask)
				
				cell_fw = MultiRNNCell(
					[CudnnCompatibleLSTMCell(self.config.rnn_hidden_size) for _ in
					 range(self.config.rnn_layer_num)]
				)
				cell_bw = MultiRNNCell(
					[CudnnCompatibleLSTMCell(self.config.rnn_hidden_size) for _ in
					 range(self.config.rnn_layer_num)]
				)
				
				(output_fw, output_bw), (output_state_fw, output_state_bw) = tf.nn.bidirectional_dynamic_rnn(
					cell_fw=cell_fw, cell_bw=cell_bw, inputs=self.word_emb,
					dtype=tf.float32, sequence_length=tf.reshape(self.similar_sentences_lens, [-1])
				)
				
				# sim_final_states = tf.concat([output_state_fw[0].h, output_state_bw[0].h], axis=1)
				sim_memory_embeddings = tf.concat([output_fw, output_bw], axis=2)
				
				sim_memory_embeddings = tf.multiply(sim_memory_embeddings, self.sim_mask)
				sim_memory_embeddings = tf.nn.dropout(sim_memory_embeddings, keep_prob=self.dropout_keep_prob,
				                                      name="utterance_memory_embeddings")
				
				self.sim_text_final_state, sim_level_att_score = self.attention_mechanism(sim_memory_embeddings,
				                                                                          tf.squeeze(self.sim_mask,
				                                                                                     axis=-1))
				self.sim_text_final_state = tf.reshape(self.sim_text_final_state,
				                                       [batch_size, max_similar_num, 2 * self.config.rnn_hidden_size])
				
				sim_level_att_score = tf.reshape(sim_level_att_score, [batch_size, max_similar_num, max_similar_length])
				self.sim_level_att_score = sim_level_att_score
		
		with tf.variable_scope("utterance_representation"):
			if self.config.use_role_embedding:
				self.final_states = tf.concat([self.role_embedding, self.sample_text_final_state], axis=2)
				if self.config.use_cross_copy:
					self.final_states_sim = tf.concat([self.similar_role_embedding, self.sim_text_final_state], axis=2)
			else:
				self.final_states = self.sample_text_final_state
				if self.config.use_cross_copy:
					self.final_states_sim = self.sim_text_final_state
		
		with tf.variable_scope("dialogue_rnn"):
			self.mask = tf.cast(tf.expand_dims(self.input_sample_mask, axis=-1), dtype=tf.float32)
			self.final_states = tf.multiply(self.mask, self.final_states)
			if self.config.use_cross_copy:
				self.sim_mask = tf.cast(tf.expand_dims(self.similar_sample_mask, axis=-1), dtype=tf.float32)
				self.final_states_sim = tf.multiply(self.sim_mask, self.final_states_sim)
			
			cell_fw = MultiRNNCell(
				[CudnnCompatibleLSTMCell(self.config.rnn_hidden_size) for _ in
				 range(self.config.rnn_layer_num)]
			)
			cell_bw = MultiRNNCell(
				[CudnnCompatibleLSTMCell(self.config.rnn_hidden_size) for _ in
				 range(self.config.rnn_layer_num)]
			)
			
			(outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(
				cell_fw=cell_fw, cell_bw=cell_bw, inputs=self.final_states,
				dtype=tf.float32, sequence_length=self.input_sample_lens,
				swap_memory=True
			)
			outputs = tf.concat(outputs, axis=2)
			
			sample_hidden_states = tf.multiply(outputs, self.mask)
			sample_hidden_states = tf.nn.dropout(sample_hidden_states, keep_prob=self.dropout_keep_prob)
			
			if self.config.use_cross_copy:
				(sim_outputs, (sim_fw_st, sim_bw_st)) = tf.nn.bidirectional_dynamic_rnn(
					cell_fw=cell_fw, cell_bw=cell_bw, inputs=self.final_states_sim,
					dtype=tf.float32, sequence_length=self.similar_sample_lens,
					swap_memory=True
				)
				sim_outputs = tf.concat(sim_outputs, axis=2)
				self.similar_hidden_states = tf.multiply(sim_outputs, self.sim_mask)
		# self.similar_hidden_states = tf.nn.dropout(self.similar_hidden_states, keep_prob=self.dropout_keep_prob)
		
		if self.config.transformer_layers > 0:
			with tf.variable_scope("transformer"):
				attention_mask = create_attention_mask_from_input_mask(from_tensor=sample_hidden_states,
				                                                       to_mask=self.input_sample_mask)
				self.all_encoder_layers = transformer_model(input_tensor=sample_hidden_states,
				                                            attention_mask=attention_mask,
				                                            hidden_size=self.config.rnn_hidden_size * 2,
				                                            num_hidden_layers=self.config.transformer_layers,
				                                            num_attention_heads=self.config.heads,
				                                            intermediate_size=self.config.intermediate_size,
				                                            intermediate_act_fn=gelu,
				                                            hidden_dropout_prob=1.0 - self.dropout_keep_prob,
				                                            initializer_range=self.config.init_std,
				                                            do_return_all_layers=True)
				self.encoder_outputs = self.all_encoder_layers[-1]
				# self.encoder_outputs = tf.nn.dropout(self.encoder_outputs, keep_prob=self.dropout_keep_prob)
			
			if self.config.use_cross_copy:
				with tf.variable_scope("transformer_sim"):
					sim_attention_mask = create_attention_mask_from_input_mask(from_tensor=self.similar_hidden_states,
					                                                           to_mask=self.similar_sample_mask)
					self.sim_encoder_layers = transformer_model(input_tensor=self.similar_hidden_states,
					                                            attention_mask=sim_attention_mask,
					                                            hidden_size=self.config.rnn_hidden_size * 2,
					                                            num_hidden_layers=self.config.transformer_layers,
					                                            num_attention_heads=self.config.heads,
					                                            intermediate_size=self.config.intermediate_size,
					                                            intermediate_act_fn=gelu,
					                                            hidden_dropout_prob=1.0 - self.dropout_keep_prob,
					                                            initializer_range=self.config.init_std,
					                                            do_return_all_layers=True)
					
					self.sim_encoder_outputs = self.sim_encoder_layers[-1]
					# self.sim_encoder_outputs = tf.nn.dropout(self.sim_encoder_outputs, keep_prob=self.dropout_keep_prob)
		
		with tf.variable_scope("Decoder"):
			with tf.variable_scope("decoder_embedding"):
				self.decoder_outputs = tf.placeholder(shape=[None, None], dtype=tf.int32, name="decoder_outputs")
				self.decoder_inputs = tf.placeholder(shape=[None, None], dtype=tf.int32, name="decoder_inputs")
				
				self.decoder_lengths = tf.placeholder(shape=[None], dtype=tf.int32, name="decoder_length")
				self.prev_coverage = tf.placeholder(tf.float32, [None, None], name='prev_coverage')
				self.dec_inp_sample_maks = tf.sequence_mask(self.input_sample_lens, dtype=tf.float32,
				                                            name="dec_inp_sample_maks")
				self.dec_sim_sample_maks = tf.sequence_mask(self.similar_sample_lens, dtype=tf.float32,
				                                            name="dec_sim_sample_maks")
				
				self.decoder_emb_inp = tf.nn.embedding_lookup(self.word_table, self.decoder_inputs,
				                                              name="decoder_embeddings")
				self.projection_layer = tf.layers.Dense(self.config.vocab_size, use_bias=True, name="projection_layer")
				self.projection_layer_pointer = tf.layers.Dense(self.config.pointer_vsize, use_bias=True,
				                                                name="projection_layer_pointer")
				self.transformer_projection_layer = tf.layers.Dense(self.config.embedding_dim, use_bias=True,
				                                                    name="transformer_projection_layer")
				self.decoder_cell = CudnnCompatibleLSTMCell(self.config.rnn_hidden_size * 2)
				
				if config.pointer_generate:
					max_word_index = tf.cond(self.training,
					                         lambda: tf.reduce_max(
						                         [tf.reduce_max(self.input_x), tf.reduce_max(self.decoder_inputs)]),
					                         lambda: tf.reduce_max(self.input_x))
					if self.config.use_cross_copy:
						
						self._max_art_oovs = tf.cond(max_word_index >= self.config.pointer_vsize,
						                             lambda: max_word_index - self.config.pointer_vsize + 1,
						                             lambda: 0)
						
						max_word_index_sim = tf.cond(self.training,
						                             lambda: tf.reduce_max([tf.reduce_max(self.similar_x),
						                                                    tf.reduce_max(self.decoder_inputs)]),
						                             lambda: tf.reduce_max(self.similar_x))
						
						self._max_art_oovs2 = tf.cond(max_word_index_sim >= self.config.pointer_vsize2,
						                              lambda: max_word_index_sim - self.config.pointer_vsize2 + 1,
						                              lambda: 0)
					else:
						self._max_art_oovs = tf.cond(max_word_index >= self.config.pointer_vsize,
						                             lambda: max_word_index - self.config.pointer_vsize + 1,
						                             lambda: 0)
			
			with tf.variable_scope("attention_layer"):
				attention_mechanism = pgn_modeling.MyLuongAttention(2 * self.config.rnn_hidden_size,
				                                                    memory=self.encoder_outputs,
				                                                    memory_sequence_length=self.input_sample_lens,
				                                                    scale=True)
				self.attention = attention_mechanism
				self.decoder_emb_inp = tf.transpose(self.decoder_emb_inp, [1, 0, 2])  # [max_step, batch, emb]
				self.decoder_emb_inp = tensor_noise(self.decoder_emb_inp, self.config.input_noise_std,
				                                    self.training)  # [max_step, batch, emb]  30  32   300
				self.decoder_cell_wrapper = pgn_modeling.MyAttentionWrapper(self.decoder_cell,
				                                                            attention_mechanism,
				                                                            attention_layer_size=self.config.rnn_hidden_size * 2,
				                                                            name="attention_wrapper")
			
			if self.config.use_cross_copy:
				with tf.variable_scope("similar_attention_layer"):
					sim_attention_mechanism = pgn_modeling.MyLuongAttention(2 * self.config.rnn_hidden_size,
					                                                        memory=self.sim_encoder_outputs,
					                                                        memory_sequence_length=self.similar_sample_lens,
					                                                        scale=True)
					
					self.sim_decoder_cell_wrapper = pgn_modeling.MyAttentionWrapper(self.decoder_cell,
					                                                                sim_attention_mechanism,
					                                                                attention_layer_size=self.config.rnn_hidden_size * 2,
					                                                                name="sim_attention_wrapper")
			
			with tf.variable_scope("attention_decoder"):
				train_helper = seq2seq.TrainingHelper(self.decoder_emb_inp,
				                                      tf.ones(shape=(self.batch_size,),
				                                              dtype=tf.int32) * self.config.max_decoder_steps,
				                                      time_major=True)
				
				self.decoder_initial_state = self.decoder_cell_wrapper.zero_state(self.batch_size, tf.float32)
				train_decoder = pgn_modeling.MyBasicDecoder(self.decoder_cell_wrapper,
				                                            train_helper,
				                                            self.decoder_initial_state)
				self.train_outputs, _, _, attn_dists, seq_inputs = pgn_modeling.my_dynamic_decode(train_decoder,
				                                                                                  output_time_major=True,
				                                                                                  swap_memory=True)  # train_outputs(? ,32, 600) (?, 64)  attn_dists(?, ?, ?) seq_inputs(?, ?, 300)
				
				if self.config.use_cross_copy:
					self.sim_decoder_initial_state = self.sim_decoder_cell_wrapper.zero_state(self.batch_size,
					                                                                          tf.float32)
					sim_train_decoder = pgn_modeling.MyBasicDecoder(self.sim_decoder_cell_wrapper,
					                                                train_helper,
					                                                self.sim_decoder_initial_state)
					self.sim_train_outputs, _, _, sim_attn_dists, sim_seq_inputs = pgn_modeling.my_dynamic_decode(
						sim_train_decoder,
						output_time_major=True,
						swap_memory=True)
				
				# self.train_outputs, self.train_state, self.train_sequence_lengths, attn_dists, seq_inputs, \
				# self.sim_train_outputs, _, _, sim_attn_dists, _ = pgn_modeling.new_dynamic_decode(train_decoder,
				#                                                                                   sim_train_decoder,
				#                                                                                   output_time_major=True,
				#                                                                                   swap_memory=True)
				
				if self.config.use_transformer_linear_projection:
					dec = self.transformer_projection_layer(
						tf.nn.dropout(self.train_outputs.rnn_output, keep_prob=self.dropout_keep_prob))
					dec = tf.transpose(dec, [1, 0, 2])  # [32, ?, 300]
					weights = tf.transpose(self.word_table)  # (300, 20001)
					self.seq2seq_logits = tf.einsum('ntd,dk->ntk', dec, weights)  # (N, T2, vocab_size) #(32, ?, 20001)
					
					if self.config.use_cross_copy:
						sim_dec = self.transformer_projection_layer(
							tf.nn.dropout(self.sim_train_outputs.rnn_output, keep_prob=self.dropout_keep_prob))
						sim_dec = tf.transpose(sim_dec, [1, 0, 2])
						sim_weights = tf.transpose(self.word_table)  # (300, 20001)
						self.sim_seq2seq_logits = tf.einsum('ntd,dk->ntk', sim_dec,
						                                    sim_weights)  # (N, T2, vocab_size) #(?, ?, 20001)
				
				else:
					self.seq2seq_logits = self.projection_layer_pointer(
						tf.nn.dropout(self.train_outputs.rnn_output, keep_prob=self.dropout_keep_prob))
					self.seq2seq_logits = tf.transpose(self.seq2seq_logits, [1, 0, 2])  # (?, ?, 20001)
					
					if self.config.use_cross_copy:
						self.sim_seq2seq_logits = self.projection_layer_pointer(
							tf.nn.dropout(self.sim_train_outputs.rnn_output, keep_prob=self.dropout_keep_prob))
						self.sim_seq2seq_logits = tf.transpose(self.sim_seq2seq_logits, [1, 0, 2])  # (32, ?, 20001)
				
				vocab_dists = tf.nn.softmax(self.seq2seq_logits)  # (32, ?, 20001)
				
				if self.config.pointer_generate:
					vocab_dists.set_shape(shape=[None, self.config.max_decoder_steps,
					                             self.config.pointer_vsize])  # tensor [32, 30, 20001]
					if self.config.use_cross_copy:
						vocab_dists.set_shape(shape=[None, self.config.max_decoder_steps,
						                             self.config.pointer_vsize2])  # tensor [32, 30, 20001]
				
				else:
					vocab_dists.set_shape(shape=[None, self.config.max_decoder_steps, self.config.vocab_size])
				
				self.vocab_dists = tf.unstack(vocab_dists, axis=1)  # max_decoder_steps (32, 20001)
				attn_dists.set_shape(shape=[None, None, self.config.max_decoder_steps])
				attn_dists = tf.unstack(attn_dists, axis=2)  # max_decoder_steps (?, ?)
				
				if self.config.use_cross_copy:
					sim_attn_dists.set_shape(shape=[None, None, self.config.max_decoder_steps])
					sim_attn_dists = tf.unstack(sim_attn_dists, axis=2)  # max_decoder_steps (?, ?)
				
				self.attn_dists = []
				self.sim_attn_dists = []
				for dist in attn_dists:
					attn_dist = tf.multiply(tf.expand_dims(dist, axis=-1), self.sen_level_att_score)
					self.attn_dists.append(attn_dist)  # batch, max_sen_num, max_sen_len
				
				if self.config.use_cross_copy:
					for dist in sim_attn_dists:
						sim_attn_dist = tf.multiply(tf.expand_dims(dist, axis=-1), self.sim_level_att_score)
						self.sim_attn_dists.append(sim_attn_dist)  # batch, max_sen_num, max_sen_len
				
				self.p_gen_dense_rnn = tf.layers.Dense(1, use_bias=False)
				self.p_gen_dense_input = tf.layers.Dense(1, use_bias=True)
				
				self.p_gens = tf.nn.sigmoid(
					self.p_gen_dense_rnn(self.train_outputs.rnn_output) + self.p_gen_dense_input(self.decoder_emb_inp))
				# self.train_outputs.rnn_output(?, 64 400) self.decoder_emb_inp(100, 64, 300) self.p_gens(100, 64, 1)
				
				self.p_gens = tf.squeeze(self.p_gens, axis=2)
				self.p_gens.set_shape([self.config.max_decoder_steps, None])
				self.p_gens = tf.unstack(self.p_gens, axis=0)
				
				if self.config.use_cross_copy:
					self.sim_p_gens = tf.nn.sigmoid(
						self.p_gen_dense_rnn(self.sim_train_outputs.rnn_output) + self.p_gen_dense_input(
							self.decoder_emb_inp))
					# self.train_outputs.rnn_output(?, 64 400) self.decoder_emb_inp(100, 64, 300) self.p_gens(100, 64, 1)
					
					self.sim_p_gens = tf.squeeze(self.sim_p_gens, axis=2)
					self.sim_p_gens.set_shape([self.config.max_decoder_steps, None])
					self.sim_p_gens = tf.unstack(self.sim_p_gens, axis=0)
				
				# if self.config.use_cross_copy:
				# 	logits = tf.layers.dense(self.p_gen_dense_rnn(self.train_outputs.rnn_output) +
				# 	                         self.p_gen_dense_rnn(self.sim_train_outputs.rnn_output) +
				# 	                         self.p_gen_dense_input(self.decoder_emb_inp), 3)
				
				# 	log_list = tf.nn.softmax(logits, axis=-1)  # [30, 32, 3]
				# 	self.p_gens, self.context_gen, self.cross_pen = tf.split(log_list, axis=-1, num_or_size_splits=3)  # max_step (32,)
				
				# 	self.p_gens = tf.squeeze(self.p_gens, axis=2)
				# 	self.p_gens.set_shape([self.config.max_decoder_steps, None])
				# 	self.p_gens = tf.unstack(self.p_gens, axis=0)   # max_step (32,)
				
				# 	self.context_gen = tf.squeeze(self.context_gen, axis=2)
				# 	self.context_gen.set_shape([self.config.max_decoder_steps, None])
				# 	self.context_gen = tf.unstack(self.context_gen, axis=0)
				
				# 	self.cross_pen = tf.squeeze(self.cross_pen, axis=2)
				# 	self.cross_pen.set_shape([self.config.max_decoder_steps, None])
				# 	self.cross_pen = tf.unstack(self.cross_pen, axis=0)
				
				
				# if self.config.pointer_generate:
				# 	if self.config.use_cross_copy:
				# 		self.final_dists = self.three_calc_final_dist(self.vocab_dists, self.attn_dists,
				# 		                                              self.sim_attn_dists, self.p_gens,
				# 		                                              self.context_gen, self.cross_pen)
				# 	else:
				# 		self.final_dists = self._calc_final_dist(self.vocab_dists, self.attn_dists, self.p_gens)
				# else:
				# 	self.final_dists = self.vocab_dists
				
				if self.config.pointer_generate:
					self.final_dists = self._calc_final_dist(self.vocab_dists, self.attn_dists, self.p_gens)
				else:
					self.final_dists = self.vocab_dists
					
				self.seq2seq_predicts = tf.argmax(tf.stack(self.final_dists, axis=1), axis=2)
			
			with tf.variable_scope("decoder_loss"):
				if self.config.pointer_generate:
					loss_per_step = []
					batch_nums = tf.range(0, limit=self.batch_size)
					for dec_step, dist in enumerate(self.final_dists):  # dist (32, ?)
						targets = self.decoder_outputs[:, dec_step]  # (32, )
						indices = tf.stack((batch_nums, targets), axis=1)  # (32, 2)
						gold_probs = tf.gather_nd(dist, indices)  # (32, )
						gold_probs = tf.clip_by_value(gold_probs, 1e-6, 1 - 1e-6)  # (32, )
						losses = -tf.log(gold_probs + 1e-20)  # (32, )
						loss_per_step.append(losses)
					
					self.loss_per_step = tf.stack(loss_per_step, axis=1)
					target_weights = tf.sequence_mask(self.decoder_lengths, dtype=tf.float32,
					                                  maxlen=self.config.max_decoder_steps)
					loss = tf.reduce_sum(self.loss_per_step * target_weights, axis=1)
					loss /= tf.cast(self.decoder_lengths, tf.float32)
					self.decoder_loss = tf.reduce_mean(loss)
				
				else:
					target_weights = tf.sequence_mask(self.decoder_lengths, dtype=tf.float32,
					                                  maxlen=self.config.max_decoder_steps)
					self.decoder_loss = seq2seq.sequence_loss(self.seq2seq_logits, self.decoder_outputs, target_weights)
				
				self.regularization_loss = tf.losses.get_regularization_loss()
				self.loss = self.decoder_loss + self.regularization_loss
			
			with tf.variable_scope("infer_decoder"):
				infer_predicts = []
				next_decoder_state = self.decoder_cell_wrapper.zero_state(self.batch_size, dtype=tf.float32)
				next_inputs = tf.nn.embedding_lookup(self.word_table, ids=tf.fill([self.batch_size], self.go_id))
				
				if self.config.use_cross_copy:
					sim_decoder_state = self.sim_decoder_cell_wrapper.zero_state(self.batch_size, dtype=tf.float32)
					sim_inputs = tf.nn.embedding_lookup(self.word_table, ids=tf.fill([self.batch_size], self.go_id))
				
				i = 0
				while i < self.config.max_decoder_steps:
					cell_outputs, next_decoder_state, cell_score = self.decoder_cell_wrapper(next_inputs,
					                                                                         next_decoder_state)
					
					if self.config.use_cross_copy:
						sim_outputs, sim_decoder_state, sim_score = self.sim_decoder_cell_wrapper(sim_inputs,
						                                                                          sim_decoder_state)
					
					if self.config.use_transformer_linear_projection:
						# max_decoder_steps, batch_size, embedding_dim
						dec = self.transformer_projection_layer(cell_outputs)
						weights = tf.transpose(self.word_table)
						infer_seq2seq_logits = tf.matmul(dec, weights)  # (N, vocab_size)
						
						if self.config.use_cross_copy:
							sim_dec = self.transformer_projection_layer(sim_outputs)
							sim_weights = tf.transpose(self.word_table)
							sim_infer_seq2seq_logits = tf.matmul(sim_dec, sim_weights)  # (N, vocab_size)
					
					else:
						infer_seq2seq_logits = self.projection_layer_pointer(cell_outputs)
						
						if self.config.use_cross_copy:
							sim_infer_seq2seq_logits = self.projection_layer_pointer(sim_outputs)
					
					infer_vocab_dists = tf.nn.softmax(infer_seq2seq_logits)
					infer_vocab_dists = tf.expand_dims(infer_vocab_dists, axis=1)
					cell_score = tf.expand_dims(cell_score, axis=2)
					infer_vocab_dists = tf.unstack(infer_vocab_dists, axis=1)
					infer_attn_dists = tf.unstack(cell_score, axis=2)
					
					re_weighted_infer_attn_dists = []
					for dist in infer_attn_dists:
						attn_dist = tf.multiply(tf.expand_dims(dist, axis=-1), self.sen_level_att_score)
						re_weighted_infer_attn_dists.append(attn_dist)  # batch, max_sen_num, max_sen_len
					
					infer_p_gens = tf.nn.sigmoid(
						self.p_gen_dense_rnn(cell_outputs) + self.p_gen_dense_input(next_inputs))
					infer_p_gens = tf.expand_dims(infer_p_gens, axis=0)
					infer_p_gens = tf.squeeze(infer_p_gens, axis=2)
					infer_p_gens.set_shape([1, None])
					infer_p_gens = tf.unstack(infer_p_gens, axis=0)
					
					if self.config.use_cross_copy:
						
						sim_infer_vocab_dists = tf.nn.softmax(sim_infer_seq2seq_logits)
						sim_infer_vocab_dists = tf.expand_dims(sim_infer_vocab_dists, axis=1)
						sim_score = tf.expand_dims(sim_score, axis=2)
						sim_infer_vocab_dists = tf.unstack(sim_infer_vocab_dists, axis=1)
						sim_infer_attn_dists = tf.unstack(sim_score, axis=2)
						
						re_weighted_sim_infer_attn_dists = []
						for sim_dist in sim_infer_attn_dists:
							sim_attn_dist = tf.multiply(tf.expand_dims(sim_dist, axis=-1), self.sim_level_att_score)
							re_weighted_sim_infer_attn_dists.append(sim_attn_dist)  # batch, max_sen_num, max_sen_len
						
						sim_infer_p_gens = tf.nn.sigmoid(
							self.p_gen_dense_rnn(sim_outputs) + self.p_gen_dense_input(sim_inputs))
						sim_infer_p_gens = tf.expand_dims(sim_infer_p_gens, axis=0)
						sim_infer_p_gens = tf.squeeze(sim_infer_p_gens, axis=2)
						sim_infer_p_gens.set_shape([1, None])
						sim_infer_p_gens = tf.unstack(sim_infer_p_gens, axis=0)
					
					# if self.config.use_cross_copy:
					# 	sim_score = tf.expand_dims(sim_score, axis=2)
					# 	sim_infer_attn_dists = tf.unstack(sim_score, axis=2)
					
					# 	re_weighted_sim_infer_attn_dists = []
					# 	for sim_dist in sim_infer_attn_dists:
					# 		sim_attn_dist = tf.multiply(tf.expand_dims(sim_dist, axis=-1), self.sim_level_att_score)
					# 		re_weighted_sim_infer_attn_dists.append(sim_attn_dist)  # batch, max_sen_num, max_sen_len
					
					# 	infer_logits = tf.layers.dense(self.p_gen_dense_rnn(cell_outputs) +
					# 	                               self.p_gen_dense_input(next_inputs) +
					# 	                               self.p_gen_dense_rnn(sim_outputs) +
					# 	                               self.p_gen_dense_input(sim_inputs), 3)
					
					# 	infer_log_list = tf.nn.softmax(infer_logits, axis=-1)
					# 	infer_p_gens, context_infer_p_gens, cross_infer_p_gens = tf.split(infer_log_list, axis=-1,
					# 	                                                                  num_or_size_splits=3)
					
					# 	infer_p_gens = tf.expand_dims(infer_p_gens, axis=0)
					# 	infer_p_gens = tf.squeeze(infer_p_gens, axis=2)
					# 	infer_p_gens.set_shape([1, None])
					# 	infer_p_gens = tf.unstack(infer_p_gens, axis=0)
					
					# 	context_infer_p_gens = tf.expand_dims(context_infer_p_gens, axis=0)
					# 	context_infer_p_gens = tf.squeeze(context_infer_p_gens, axis=2)
					# 	context_infer_p_gens.set_shape([1, None])
					# 	context_infer_p_gens = tf.unstack(context_infer_p_gens, axis=0)
					
					# 	cross_infer_p_gens = tf.expand_dims(cross_infer_p_gens, axis=0)
					# 	cross_infer_p_gens = tf.squeeze(cross_infer_p_gens, axis=2)
					# 	cross_infer_p_gens.set_shape([1, None])
					# 	cross_infer_p_gens = tf.unstack(cross_infer_p_gens, axis=0)
					
					# if self.config.pointer_generate:
					# 	if self.config.use_cross_copy:
					# 		infer_final_dists = self.three_calc_final_dist(infer_vocab_dists,
					# 		                                               re_weighted_infer_attn_dists,
					# 		                                               re_weighted_sim_infer_attn_dists,
					# 		                                               infer_p_gens, context_infer_p_gens,
					# 		                                               cross_infer_p_gens)
					# 	else:
					# 		infer_final_dists1 = self._calc_final_dist(infer_vocab_dists, re_weighted_infer_attn_dists,
					# 		                                          infer_p_gens)
					# 		infer_final_dists2 = self.sim_calc_final_dist(sim_infer_vocab_dists,
					# 		                                             re_weighted_sim_infer_attn_dists,
					# 		                                             sim_infer_p_gens)
					
					# else:
					# 	infer_final_dists = infer_vocab_dists
					
					infer_final_dists = self._calc_final_dist(infer_vocab_dists, re_weighted_infer_attn_dists, infer_p_gens)
					infer_final_dists = self.sim_calc_final_dist(infer_final_dists, re_weighted_sim_infer_attn_dists, sim_infer_p_gens)
					
					infer_final_dists = tf.stack(infer_final_dists, axis=1)
					sample_id = tf.squeeze(tf.argmax(infer_final_dists, axis=2), axis=1)
					next_inputs = tf.nn.embedding_lookup(self.word_table, ids=sample_id)
					infer_predicts.append(sample_id)
					i += 1
				
				self.infer_predicts = tf.stack(infer_predicts, axis=1, name="infer_predicts")
		
		with tf.variable_scope("train_op"):
			tvars = tf.trainable_variables()
			initialized_variable_names = {}
			
			if self.config.fine_tuning:
				(assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(tvars,
				                                                                                  self.config.pre_train_lm_path)
				tf.train.init_from_checkpoint(self.config.pre_train_lm_path, assignment_map)
				print("load bert check point done")
			
			tf.logging.info("**** Trainable Variables ****")
			for var in tvars:
				init_string = ""
				
				if var.name in initialized_variable_names:
					init_string = ", *INIT_FROM_CKPT*"
				
				tf.logging.info("name = %s, shape = %s%s", var.name, var.shape, init_string)
				print("name = %s, shape = %s%s", var.name, var.shape, init_string)
	
	@staticmethod
	def attention_mechanism(inputs, x_mask=None):
		if isinstance(inputs, tuple):
			inputs = tf.concat(inputs, 2)
		_, sequence_length, hidden_size = get_shape_list(inputs)
		
		v = tf.layers.dense(
			inputs, hidden_size,
			kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
			activation=tf.tanh,
			use_bias=True
		)
		att_score = tf.layers.dense(
			v, 1,
			kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
			use_bias=False
		)  # batch_size, sequence_length, 1
		
		att_score = tf.squeeze(att_score, axis=-1) * x_mask + VERY_NEGATIVE_NUMBER * (
				1 - x_mask)  # [batch_size, sentence_length
		att_score = tf.expand_dims(tf.nn.softmax(att_score), axis=-1)  # [batch_size, sentence_length, 1]
		att_pool_vec = tf.matmul(tf.transpose(att_score, [0, 2, 1]), inputs)  # [batch_size,  h]
		att_pool_vec = tf.squeeze(att_pool_vec, axis=1)
		
		return att_pool_vec, att_score
	
	def _calc_final_dist(self, vocab_dists, attn_dists, p_gens):
		
		with tf.variable_scope('final_distribution'):
			vocab_dists = [tf.expand_dims(p_gen, axis=-1) * dist for (p_gen, dist) in zip(p_gens, vocab_dists)]
			attn_dists = [tf.expand_dims(tf.expand_dims((1 - p_gen), axis=-1), axis=-1) * dist for (p_gen, dist) in
			              zip(p_gens, attn_dists)]
			extended_vsize = self.config.pointer_vsize + self._max_art_oovs
			extra_zeros = tf.zeros((self.batch_size, self._max_art_oovs))
			vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in
			                        vocab_dists]
			
			batch_nums = tf.range(0, limit=self.batch_size)  # shape (batch_size)
			batch_nums = tf.expand_dims(batch_nums, 1)  # shape (batch_size, 1)
			# attn_len = tf.shape(self._enc_batch_extend_vocab)[1]  # number of states we attend over
			attn_len = self.max_sequence_length * self.max_sentence_num
			batch_nums = tf.tile(batch_nums, [1, attn_len])  # shape (batch_size, attn_len)
			indices = tf.stack(
				(
					batch_nums,
					tf.reshape(self.input_x, [self.batch_size, self.max_sequence_length * self.max_sentence_num])
				), axis=2
			)  # shape (batch_size, enc_t, 2)
			shape = [self.batch_size, extended_vsize]
			attn_dists_projected = [
				tf.scatter_nd(
					indices,
					tf.reshape(copy_dist, [self.batch_size, self.max_sequence_length * self.max_sentence_num]),
					shape
				) for copy_dist in attn_dists
			]  # list length max_dec_steps (batch_size, extended_vsize)
			final_dists = [vocab_dist + copy_dist for (vocab_dist, copy_dist) in
			               zip(vocab_dists_extended, attn_dists_projected)]
			
			return final_dists
	
	def sim_calc_final_dist(self, vocab_dists, attn_dists, p_gens):
		
		with tf.variable_scope('final_distribution'):
			vocab_dists = [tf.expand_dims(p_gen, axis=-1) * dist for (p_gen, dist) in zip(p_gens, vocab_dists)]
			attn_dists = [tf.expand_dims(tf.expand_dims((1 - p_gen), axis=-1), axis=-1) * dist for (p_gen, dist) in
			              zip(p_gens, attn_dists)]
			
			extended_vsize = self.config.pointer_vsize2 + self._max_art_oovs2
			extra_zeros = tf.zeros((self.batch_size, self._max_art_oovs))
			vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists]
			
			batch_nums = tf.range(0, limit=self.batch_size)  # shape (batch_size)
			batch_nums = tf.expand_dims(batch_nums, 1)  # shape (batch_size, 1)
			
			attn_len_sim = self.max_similar_length * self.max_similar_num
			batch_nums_sim = tf.tile(batch_nums, [1, attn_len_sim])  # shape (batch_size, attn_len_sim)
			
			indices_sim = tf.stack(
				(
					batch_nums_sim,
					tf.reshape(self.similar_x, [self.batch_size, self.max_similar_length * self.max_similar_num])
				), axis=2
			)  # shape (batch_size, enc_t, 2)
			
			shape = [self.batch_size, extended_vsize]
			
			attn_dists_projected_sim = [
				tf.scatter_nd(
					indices_sim,
					tf.reshape(copy_dist, [self.batch_size, self.max_similar_length * self.max_similar_num]),
					shape
				) for copy_dist in attn_dists
			]  # list length max_dec_steps (batch_size, extended_vsize)
			
			final_dists = [vocab_dist + cro_copy_dist for (vocab_dist, cro_copy_dist) in
			               zip(vocab_dists_extended, attn_dists_projected_sim)]
			
			return final_dists
	
	def three_calc_final_dist(self, vocab_dists, attn_dists, sim_attn_dists, p_gens, context_gens, cross_pens):
		with tf.variable_scope('final_distribution'):
			vocab_dists = [tf.expand_dims(p_gen, axis=-1) * dist for (p_gen, dist) in
			               zip(p_gens, vocab_dists)]  # [32, 30, 20001]
			attn_dists = [tf.expand_dims(tf.expand_dims(context_gen, axis=-1), axis=-1) * dist for (context_gen, dist)
			              in zip(context_gens, attn_dists)]
			cro_attn_dists = [tf.expand_dims(tf.expand_dims(cross_pen, axis=-1), axis=-1) * dist for (cross_pen, dist)
			                  in zip(cross_pens, sim_attn_dists)]
			
			extended_vsize = self.config.pointer_vsize + self._max_art_oovs
			extra_zeros = tf.zeros((self.batch_size, self._max_art_oovs))
			vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists]
			
			batch_nums = tf.range(0, limit=self.batch_size)  # shape (batch_size)
			batch_nums = tf.expand_dims(batch_nums, 1)  # shape (batch_size, 1)
			
			attn_len = self.max_sequence_length * self.max_sentence_num
			attn_len_sim = self.max_similar_length * self.max_similar_num
			
			batch_nums_con = tf.tile(batch_nums, [1, attn_len])  # shape (batch_size, attn_len)
			batch_nums_sim = tf.tile(batch_nums, [1, attn_len_sim])  # shape (batch_size, attn_len_sim)
			
			indices = tf.stack(
				(
					batch_nums_con,
					tf.reshape(self.input_x, [self.batch_size, self.max_sequence_length * self.max_sentence_num])
				), axis=2
			)  # shape (batch_size, enc_t, 2)
			
			indices_sim = tf.stack(
				(
					batch_nums_sim,
					tf.reshape(self.similar_x, [self.batch_size, self.max_similar_length * self.max_similar_num])
				), axis=2
			)  # shape (batch_size, enc_t, 2)
			
			shape = [self.batch_size, extended_vsize]
			
			attn_dists_projected = [
				tf.scatter_nd(
					indices,
					tf.reshape(copy_dist, [self.batch_size, self.max_sequence_length * self.max_sentence_num]),
					shape
				) for copy_dist in attn_dists
			]  # list length max_dec_steps (batch_size, extended_vsize)
			
			attn_dists_projected_sim = [
				tf.scatter_nd(
					indices_sim,
					tf.reshape(copy_dist, [self.batch_size, self.max_similar_length * self.max_similar_num]),
					shape
				) for copy_dist in cro_attn_dists
			]  # list length max_dec_steps (batch_size, extended_vsize)
			
			final_dists = [vocab_dist + copy_dist + cro_copy_dist for (vocab_dist, copy_dist, cro_copy_dist) in
			               zip(vocab_dists_extended, attn_dists_projected, attn_dists_projected_sim)]
			
			return final_dists


