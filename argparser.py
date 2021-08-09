from __future__ import print_function
import os, argparse

def model_opts():
	
	parser = argparse.ArgumentParser('Model-Embeddings')
	parser.add_argument("--cuda_device", type=str, default='0,1',  help="The number of GPU used")
	# Load data path
	parser.add_argument("--train_data_file", type=str, default="../data/jdm1_train", help="Data source for training.")
	parser.add_argument("--dev_data_file",   type=str, default="../data/jdm1_val", help="Data source for validation.")
	parser.add_argument("--test_data_file",  type=str, default="../data/jdm1_test", help="Data source for testing.")
	parser.add_argument("--word_emb_file",   type=str, default="../data/jd_voc_w2v", help="Pre train embedding file.")
	parser.add_argument("--vocab_file",      type=str, default="../data/jd_vocab", help="Vocab model file.")
	parser.add_argument("--train_sim_file",  type=str, default="../data/jdm1_train_sim", help="Similar case files")
	parser.add_argument("--dev_sim_file",    type=str, default="../data/jdm1_val_sim", help="Similar case files")
	parser.add_argument("--test_sim_file",   type=str, default="../data/jdm1_test_sim", help="Similar case files")
	parser.add_argument("--data_file",       type=str, default="../data/jdm1_train.pkl", help="Data source for article classes.")
	parser.add_argument("--pre_file",        type=str, default="../data/jdm1_pre_file", help="The file of the predict.")
	parser.add_argument("--gro_file",        type=str, default="../data/jdm1_gro_file", help="The file of the groundtruth.")
	
	# Model select
	parser.add_argument("--use_role_embedding", type=bool, default=True,  help="Use role embedding or not  (default:True)")
	parser.add_argument("--use_cross_copy",     type=bool, default=True, help="Use cross copy or not True or False (default:True)")
	parser.add_argument("--pointer_generate",   type=bool, default=True, help="use PGN or not")
	parser.add_argument("--coverage_mechanism", type=bool, default=True, help="use coverage or not")
	parser.add_argument("--greedy_search",      type=bool, default=False, help="use greedy search or not")
	parser.add_argument("--beam_search",        type=bool, default=True, help="use beam search or not")
	parser.add_argument("--padding_data",       type=bool, default=False, help="use coverage or not")
	parser.add_argument("--intercept",          type=bool, default=False, help="use coverage or not")
	
	# Embedding params
	parser.add_argument("--batch_size",    type=int, default=64, help="Batch Size (default: 128)")
	parser.add_argument("--embedding_dim", type=int, default=300, help="Dimensionality of word embedding (default: 300)")
	parser.add_argument("--vocab_size",    type=int, default=30001, help="Words in total in Vocab  (default: 160000)")
	parser.add_argument("--pointer_vsize", type=int, default=30001, help="Words in total in Pointer Vocab  (default: 160000)")
	parser.add_argument("--pointer_vsize2", type=int, default=30001,  help="Words in total in Pointer Vocab  (default: 160000)")
	parser.add_argument("--emb_dense_size",type=int, default=300, help="Dimensionality of word embedding dense layer (default: 300)")
	
	# transformer used
	parser.add_argument("--transformer_layers", type=int, default=2, help="Transformer layers (default: 4)")
	parser.add_argument("--sen_transformer_layers", type=int, default=1, help="Sentence Level Transformer layers (default: 1)")
	parser.add_argument("--heads", type=int, default=8, help="multi-head attention (default: 4)")
	parser.add_argument("--intermediate_size", type=int, default=1000, help="Intermediate size (default: 1000)")
	parser.add_argument("--use_transformer_linear_projection", type=bool, default=True, help="Use transformer linear projection or not")
	
	# Pre-train parameters
	parser.add_argument("--fine_tuning", type=bool, default=False, help= "fine_tuning from pretrained lm files")
	parser.add_argument("--continue_training", type=bool, default=False, help="continue training from restore, or start from scratch")
	parser.add_argument("--checkpoint_path", type=str, default=" ", help="Checkpoint file path without extension, as list in file 'checkpoints'")
	parser.add_argument("--pre_train_lm_path", type=str, default=" ", help="Checkpoint file path from pre trained language model.'")
	parser.add_argument("--pre_word_embeddings",type=str, default=" ", help="pre_trained_word_embeddings")
	parser.add_argument("--pre_node_embeddings",type=str, default=" ", help="pre_node_embeddings")
	
	# select model parameters
	parser.add_argument("--role_num", type=int, default=5, help="How many roles  (default: 3)")
	parser.add_argument("--role_emb", type=int, default=100, help="Dimensionality of role embedding  (default: 100)")
	parser.add_argument("--num_classes", type=int, default=41, help="Number of classes (default: 41)")
	parser.add_argument("--max_decoder_steps", type=int, default=30, help="max_decoder_steps")
	parser.add_argument("--max_path_length", type=int, default=5, help="Max path sequence length (default: 5)")
	parser.add_argument("--max_sequence_length", type=int, default=20, help="Max sentence sequence length (default: 40)")
	parser.add_argument("--max_sentence_num", type=int, default=15, help="Max sentence sequence length (default: 15)")
	
	# Data sample bound
	parser.add_argument("--lower_bound", type=int, default=3000, help="lower bound frequency for over-sampling (default: 3,000)")
	parser.add_argument("--upper_bound", type=int, default=1000000, help="upper bound frequency for sub-sampling (default: 100,000)")
	parser.add_argument("--over_sample_times", type=int, default=0, help="over_sample_times (default: 1)")
	
	# rnn used
	parser.add_argument("--rnn_hidden_size", type=int, default=300, help="rnn hidden size (default: 300)")
	parser.add_argument("--rnn_layer_num", type=int, default=1, help="rnn layer num (default: 1)")
	parser.add_argument("--rnn_attention_size", type=int, default=400, help="rnn attention dense layer size (default: 300)")
	parser.add_argument("--rnn_output_mlp_size", type=int, default=500, help="rnn output mlp size (default: 500)")
	parser.add_argument("--num_k", type=int, default=15, help="drnn window size (default: 15)")
	parser.add_argument("--ram_gru_size", type=int, default=300, help="recurrent attention gru cell size (default: 300)")
	parser.add_argument("--ram_times", type=int, default=4, help="recurrent attention times (default: 4)")
	parser.add_argument("--ram_hidden_size", type=int, default=300, help="recurrent attention final episode attention hidden size (default: 300)")
	
	# cnn used
	parser.add_argument("--filter_sizes", type=str, default="1,2,3,4", help="Comma-separated filter sizes (default: '1,2,3,4,5')")
	parser.add_argument("--fc1_dense_size", type=int, default=512, help="fc size before output layer (default: 2048)")
	parser.add_argument("--num_filters", type=str, default="64,64,64,64", help="Number of filters per filter size (default: 128)")
	parser.add_argument("--dropout_keep_prob", type=float, default=0.8, help="Dropout keep probability (default: 0.5)")
	parser.add_argument("--l2_lambda", type=float, default=0.0, help="L2 regularization lambda (default: 0.0)")
	parser.add_argument("--init_std", type=float, default=0.01, help="Init std value for variables (default: 0.01)")
	parser.add_argument("--input_noise_std", type=float, default=0.05, help="Input for noise  (default: 0.01)")
	parser.add_argument("--max_grad_norm",type=float, default=10, help="clip gradients to this norm (default: 10)")
	parser.add_argument("--activation_function", type=str, default="relu", help="activation function used,relu swish elu crelu tanh gelu (default: relu) ")
	
	# Training parameters
	parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs (default: 50)")
	parser.add_argument("--evaluate_every", type=int, default=500, help="Evaluate model on valid set after this many steps (default: 1000)")
	parser.add_argument("--checkpoint_every",type=int, default=500, help="Save model after this many steps (default: 1000)")
	parser.add_argument("--num_checkpoints",type=int, default=20, help="Number of checkpoints to store (default: 5)")
	parser.add_argument("--decay_step",type=int, default=500, help="learning rate decay step (default: 20000)")
	parser.add_argument("--decay_rate", type=float, default=0.9, help="learning rate decay rate (default: 0.7)")
	parser.add_argument("--learning_rate", type=float, default=5e-4, help="initial learning rate (default: 1e-3)")
	parser.add_argument("--warm_up_steps_percent", type=float, default=0.05, help="Warm up steps percent (default: 5%)")
	
	args = parser.parse_args()
	args.filter_sizes = list(map(int, args.filter_sizes.split(",")))
	args.num_filters = list(map(int, args.num_filters.split(",")))
	os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
	
	return args
	
