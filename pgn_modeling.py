import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
import six
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

from tensorflow.python.ops import rnn

import tensorflow.contrib.seq2seq as seq2seq
import collections
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _luong_score, _BaseAttentionMechanism, dtypes, layers_core
from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.contrib.seq2seq.python.ops import helper as helper_py
from tensorflow.python.layers import base as layers_base


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
      tensor: A tf.Tensor to check the rank of.
      expected_rank: Python integer or list of integers, expected rank.
      name: Optional name of the tensor for the error message.

    Raises:
      ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
      tensor: A tf.Tensor object to find the shape of.
      expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
      name: Optional name of the tensor for the error message.

    Returns:
      A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as tf.Tensor scalars.
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


# Note: this function is based on tf.contrib.legacy_seq2seq_attention_decoder, which is now outdated.
# In the future, it would make more sense to write variants on the attention mechanism using the new seq2seq library for tensorflow 1.0: https://www.tensorflow.org/api_guides/python/contrib.seq2seq#Attention
def attention_decoder(decoder_inputs, initial_state, encoder_states, enc_padding_mask, cell,
                      initial_state_attention=False, pointer_gen=True, use_coverage=False, prev_coverage=None):
    """
    Args:
      decoder_inputs: A list of 2D Tensors [batch_size x input_size].
      initial_state: 2D Tensor [batch_size x cell.state_size].
      encoder_states: 3D Tensor [batch_size x attn_length x attn_size].
      enc_padding_mask: 2D Tensor [batch_size x attn_length] containing 1s and 0s; indicates which of the encoder locations are padding (0) or a real token (1).
      cell: rnn_cell.RNNCell defining the cell function and size.
      initial_state_attention:
        Note that this attention decoder passes each decoder input through a linear layer with the previous step's context vector to get a modified version of the input. If initial_state_attention is False, on the first decoder step the "previous context vector" is just a zero vector. If initial_state_attention is True, we use initial_state to (re)calculate the previous step's context vector. We set this to False for train/eval mode (because we call attention_decoder once for all decoder steps) and True for decode mode (because we call attention_decoder once for each decoder step).
      pointer_gen: boolean. If True, calculate the generation probability p_gen for each decoder step.
      use_coverage: boolean. If True, use coverage mechanism.
      prev_coverage:
        If not None, a tensor with shape (batch_size, attn_length). The previous step's coverage vector. This is only not None in decode mode when using coverage.

    Returns:
      outputs: A list of the same length as decoder_inputs of 2D Tensors of
        shape [batch_size x cell.output_size]. The output vectors.
      state: The final state of the decoder. A tensor shape [batch_size x cell.state_size].
      attn_dists: A list containing tensors of shape (batch_size,attn_length).
        The attention distributions for each decoder step.
      p_gens: List of length input_size, containing tensors of shape [batch_size, 1]. The values of p_gen for each decoder step. Empty list if pointer_gen=False.
      coverage: Coverage vector on the last step computed. None if use_coverage=False.
    """
    with variable_scope.variable_scope("attention_decoder") as scope:
        batch_size, _, attn_size = get_shape_list(encoder_states)
        # batch_size = encoder_states.get_shape()[0].value  # if this line fails, it's because the batch size isn't defined
        # attn_size = encoder_states.get_shape()[2].value  # if this line fails, it's because the attention length isn't defined

        # Reshape encoder_states (need to insert a dim)
        encoder_states = tf.expand_dims(encoder_states, axis=2)  # now is shape (batch_size, attn_len, 1, attn_size)

        # To calculate attention, we calculate
        #   v^T tanh(W_h h_i + W_s s_t + b_attn)
        # where h_i is an encoder state, and s_t a decoder state.
        # attn_vec_size is the length of the vectors v, b_attn, (W_h h_i) and (W_s s_t).
        # We set it to be equal to the size of the encoder states.
        attention_vec_size = attn_size

        # Get the weight matrix W_h and apply it to each encoder state to get (W_h h_i), the encoder features
        W_h = variable_scope.get_variable("W_h", [1, 1, attn_size, attention_vec_size])
        encoder_features = nn_ops.conv2d(encoder_states, W_h, [1, 1, 1, 1], "SAME")  # shape (batch_size,attn_length,1,attention_vec_size)

        # Get the weight vectors v and w_c (w_c is for coverage)
        v = variable_scope.get_variable("v", [attention_vec_size])
        if use_coverage:
            with variable_scope.variable_scope("coverage"):
                w_c = variable_scope.get_variable("w_c", [1, 1, 1, attention_vec_size])

        if prev_coverage is not None:  # for beam search mode with coverage
            # reshape from (batch_size, attn_length) to (batch_size, attn_len, 1, 1)
            prev_coverage = tf.expand_dims(tf.expand_dims(prev_coverage, 2), 3)

        def attention(decoder_state, coverage=None):
            """Calculate the context vector and attention distribution from the decoder state.

            Args:
              decoder_state: state of the decoder
              coverage: Optional. Previous timestep's coverage vector, shape (batch_size, attn_len, 1, 1).

            Returns:
              context_vector: weighted sum of encoder_states
              attn_dist: attention distribution
              coverage: new coverage vector. shape (batch_size, attn_len, 1, 1)
            """
            with variable_scope.variable_scope("Attention"):
                # Pass the decoder state through a linear layer (this is W_s s_t + b_attn in the paper)
                decoder_features = linear(decoder_state, attention_vec_size, True)  # shape (batch_size, attention_vec_size)
                decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1)  # reshape to (batch_size, 1, 1, attention_vec_size)

                def masked_attention(e):
                    """Take softmax of e then apply enc_padding_mask and re-normalize"""
                    attn_dist = nn_ops.softmax(e)  # take softmax. shape (batch_size, attn_length)
                    attn_dist *= enc_padding_mask  # apply mask
                    masked_sums = tf.reduce_sum(attn_dist, axis=1)  # shape (batch_size)
                    return attn_dist / tf.reshape(masked_sums, [-1, 1])  # re-normalize

                if use_coverage and coverage is not None:  # non-first step of coverage
                    # Multiply coverage vector by w_c to get coverage_features.
                    coverage_features = nn_ops.conv2d(coverage, w_c, [1, 1, 1, 1],
                                                      "SAME")  # c has shape (batch_size, attn_length, 1, attention_vec_size)

                    # Calculate v^T tanh(W_h h_i + W_s s_t + w_c c_i^t + b_attn)
                    e = math_ops.reduce_sum(v * math_ops.tanh(encoder_features + decoder_features + coverage_features),
                                            [2, 3])  # shape (batch_size,attn_length)

                    # Calculate attention distribution
                    attn_dist = masked_attention(e)

                    # Update coverage vector
                    coverage += array_ops.reshape(attn_dist, [batch_size, -1, 1, 1])
                else:
                    # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
                    e = math_ops.reduce_sum(v * math_ops.tanh(encoder_features + decoder_features),
                                            [2, 3])  # calculate e

                    # Calculate attention distribution
                    attn_dist = masked_attention(e)

                    if use_coverage:  # first step of training
                        coverage = tf.expand_dims(tf.expand_dims(attn_dist, 2), 2)  # initialize coverage

                # Calculate the context vector from attn_dist and encoder_states
                context_vector = math_ops.reduce_sum(
                    array_ops.reshape(attn_dist, [batch_size, -1, 1, 1]) * encoder_states,
                    [1, 2])  # shape (batch_size, attn_size).
                context_vector = array_ops.reshape(context_vector, [-1, attn_size])

            return context_vector, attn_dist, coverage

        outputs = []
        attn_dists = []
        p_gens = []
        state = initial_state
        coverage = prev_coverage  # initialize coverage to None or whatever was passed in
        context_vector = array_ops.zeros([batch_size, attn_size])
        context_vector.set_shape([None, attn_size])  # Ensure the second shape of attention vectors is set.
        if initial_state_attention:  # true in decode mode
            # Re-calculate the context vector from the previous step so that we can pass it through a linear layer with this step's input to get a modified version of the input
            context_vector, _, coverage = attention(initial_state, coverage)  # in decode mode, this is what updates the coverage vector
        for i, inp in enumerate(decoder_inputs):
            tf.logging.info("Adding attention_decoder timestep %i of %i", i, len(decoder_inputs))
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()

            # Merge input and previous attentions into one vector x of the same size as inp
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)
            x = linear([inp] + [context_vector], input_size, True)

            # Run the decoder RNN cell. cell_output = decoder state
            cell_output, state = cell(x, state)

            # Run the attention mechanism.
            if i == 0 and initial_state_attention:  # always true in decode mode
                with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                                   reuse=True):  # you need this because you've already run the initial attention(...) call
                    context_vector, attn_dist, _ = attention(state, coverage)  # don't allow coverage to update
            else:
                context_vector, attn_dist, coverage = attention(state, coverage)
            attn_dists.append(attn_dist)

            # Calculate p_gen
            if pointer_gen:
                with tf.variable_scope('calculate_pgen'):
                    p_gen = linear([context_vector, state.c, state.h, x], 1, True)  # Tensor shape (batch_size, 1)
                    p_gen = tf.sigmoid(p_gen)
                    p_gens.append(p_gen)

            # Concatenate the cell_output (= decoder state) and the context vector, and pass them through a linear layer
            # This is V[s_t, h*_t] + b in the paper
            with variable_scope.variable_scope("AttnOutputProjection"):
                output = linear([cell_output] + [context_vector], cell.output_size, True)
            outputs.append(output)

        # If using coverage, reshape it
        if coverage is not None:
            coverage = array_ops.reshape(coverage, [batch_size, -1])

        return outputs, state, attn_dists, p_gens, coverage


def linear(args, output_size, bias, bias_start=0.0, scope=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: VariableScope for the created subgraph; defaults to "Linear".

    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (isinstance(args, (list, tuple)) and not args):
        raise ValueError("`args` must be specified")
    if not isinstance(args, (list, tuple)):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(axis=1, values=args), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable(
            "Bias", [output_size], initializer=tf.constant_initializer(bias_start))
    return res + bias_term


def _compute_attention(attention_mechanism, cell_output, attention_state,
                       attention_layer):
    """Computes the attention and alignments for a given attention_mechanism."""
    alignments, next_attention_state, score = attention_mechanism(
        cell_output, state=attention_state)

    # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
    expanded_alignments = array_ops.expand_dims(alignments, 1)
    # Context is the inner product of alignments and values along the
    # memory time dimension.
    # alignments shape is
    #   [batch_size, 1, memory_time]
    # attention_mechanism.values shape is
    #   [batch_size, memory_time, memory_size]
    # the batched matmul is over memory_time, so the output shape is
    #   [batch_size, 1, memory_size].
    # we then squeeze out the singleton dim.
    context = math_ops.matmul(expanded_alignments, attention_mechanism.values)
    context = array_ops.squeeze(context, [1])

    if attention_layer is not None:
        attention = attention_layer(array_ops.concat([cell_output, context], 1))
    else:
        attention = context

    return attention, alignments, next_attention_state, score


class MyAttentionWrapper(seq2seq.AttentionWrapper):

    def __init__(
            self,
            cell,
            attention_mechanism,
            attention_layer_size=None,
            alignment_history=False,
            cell_input_fn=None,
            output_attention=True,
            initial_cell_state=None,
            name=None,
            attention_layer=None
    ):
        super(MyAttentionWrapper, self).__init__(
            cell,
            attention_mechanism,
            attention_layer_size,
            alignment_history,
            cell_input_fn,
            output_attention,
            initial_cell_state,
            name,
            attention_layer
        )
        self.encoder_shape = attention_mechanism.encoder_shape

    def call(self, inputs, state):
        """Perform a step of attention-wrapped RNN.

        - Step 1: Mix the `inputs` and previous step's `attention` output via
          `cell_input_fn`.
        - Step 2: Call the wrapped `cell` with this input and its previous state.
        - Step 3: Score the cell's output with `attention_mechanism`.
        - Step 4: Calculate the alignments by passing the score through the
          `normalizer`.
        - Step 5: Calculate the context vector as the inner product between the
          alignments and the attention_mechanism's values (memory).
        - Step 6: Calculate the attention output by concatenating the cell output
          and context through the attention layer (a linear layer with
          `attention_layer_size` outputs).

        Args:
          inputs: (Possibly nested tuple of) Tensor, the input at this time step.
          state: An instance of `AttentionWrapperState` containing
            tensors from the previous time step.

        Returns:
          A tuple `(attention_or_cell_output, next_state)`, where:

          - `attention_or_cell_output` depending on `output_attention`.
          - `next_state` is an instance of `AttentionWrapperState`
             containing the state calculated at this time step.

        Raises:
          TypeError: If `state` is not an instance of `AttentionWrapperState`.
        """
        if not isinstance(state, tf.contrib.seq2seq.AttentionWrapperState):
            raise TypeError("Expected state to be instance of AttentionWrapperState. "
                            "Received type %s instead." % type(state))

        # Step 1: Calculate the true inputs to the cell based on the
        # previous attention value.
        cell_inputs = self._cell_input_fn(inputs, state.attention)
        cell_state = state.cell_state
        # print(self._cell(cell_inputs, cell_state))
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

        cell_batch_size = (cell_output.shape[0].value or array_ops.shape(cell_output)[0])
        error_message = (
                "When applying AttentionWrapper %s: " % self.name +
                "Non-matching batch sizes between the memory "
                "(encoder output) and the query (decoder output).  Are you using "
                "the BeamSearchDecoder?  You may need to tile your memory input via "
                "the tf.contrib.seq2seq.tile_batch function with argument "
                "multiple=beam_width.")
        with ops.control_dependencies(self._batch_size_checks(cell_batch_size, error_message)):
            cell_output = array_ops.identity(cell_output, name="checked_cell_output")

        if self._is_multi:
            previous_attention_state = state.attention_state
            previous_alignment_history = state.alignment_history
        else:
            previous_attention_state = [state.attention_state]
            previous_alignment_history = [state.alignment_history]

        all_alignments = []
        all_attentions = []
        all_attention_states = []
        maybe_all_histories = []
        all_attention_scores = []
        
        for i, attention_mechanism in enumerate(self._attention_mechanisms):
            attention, alignments, next_attention_state, score = _compute_attention(
                attention_mechanism, cell_output, previous_attention_state[i], self._attention_layers[i] if self._attention_layers else None)

            alignment_history = previous_alignment_history[i].write(state.time, alignments) if self._alignment_history else ()

            all_attention_states.append(next_attention_state)
            all_alignments.append(alignments)
            all_attentions.append(attention)
            maybe_all_histories.append(alignment_history)
            all_attention_scores.append(score)

        attention = array_ops.concat(all_attentions, 1)
        next_state = tf.contrib.seq2seq.AttentionWrapperState(
            time=state.time + 1,
            cell_state=next_cell_state,
            attention=attention,
            attention_state=self._item_or_tuple(all_attention_states),
            alignments=self._item_or_tuple(all_alignments),
            alignment_history=self._item_or_tuple(maybe_all_histories))

        score = all_attention_scores[0]

        if self._output_attention:
            return attention, next_state, score
        else:
            return cell_output, next_state, score



class MyLuongAttention(_BaseAttentionMechanism):


    def __init__(self,
                 num_units,
                 memory,
                 memory_sequence_length=None,
                 scale=False,
                 probability_fn=None,
                 score_mask_value=None,
                 dtype=None,
                 name="LuongAttention"):

        if probability_fn is None:
            probability_fn = nn_ops.softmax
        if dtype is None:
            dtype = dtypes.float32
        wrapped_probability_fn = lambda score, _: probability_fn(score)
        super(MyLuongAttention, self).__init__(
            query_layer=None,
            memory_layer=layers_core.Dense(
                num_units, name="memory_layer", use_bias=False, dtype=dtype),
            memory=memory,
            probability_fn=wrapped_probability_fn,
            memory_sequence_length=memory_sequence_length,
            score_mask_value=score_mask_value,
            name=name)
        self._num_units = num_units
        self._scale = scale
        self._name = name
        self.encoder_shape = get_shape_list(memory)
        self.memory_sequence_length = memory_sequence_length

    def __call__(self, query, state):

        with variable_scope.variable_scope(None, "luong_attention", [query]):
            score = _luong_score(query, self._keys, self._scale)
        alignments = self._probability_fn(score, state)
        next_state = alignments
        mask = tf.sequence_mask(self.memory_sequence_length, dtype=tf.float32)
        score = score * mask - (1 - mask) * 1e30
        return alignments, next_state, tf.nn.softmax(score)



class BasicDecoderOutput(
    collections.namedtuple("BasicDecoderOutput", ("rnn_output", "sample_id"))):
    pass


class MyBasicDecoder(decoder.Decoder):

    def __init__(self, cell, helper, initial_state, output_layer=None):
        """Initialize BasicDecoder.

        Args:
          cell: An `RNNCell` instance.
          helper: A `Helper` instance.
          initial_state: A (possibly nested tuple of...) tensors and TensorArrays.
            The initial state of the RNNCell.
          output_layer: (Optional) An instance of `tf.layers.Layer`, i.e.,
            `tf.layers.Dense`. Optional layer to apply to the RNN output prior
            to storing the result or sampling.

        Raises:
          TypeError: if `cell`, `helper` or `output_layer` have an incorrect type.
        """
        rnn_cell_impl.assert_like_rnncell("cell", cell)
        if not isinstance(helper, helper_py.Helper):
            raise TypeError("helper must be a Helper, received: %s" % type(helper))
        if (output_layer is not None and not isinstance(output_layer, layers_base.Layer)):
            raise TypeError("output_layer must be a Layer, received: %s" % type(output_layer))
        self._cell = cell
        self._helper = helper
        self._initial_state = initial_state
        self._output_layer = output_layer
        self.encoder_shape = cell.encoder_shape

    @property
    def batch_size(self):
        return self._helper.batch_size

    def _rnn_output_size(self):
        size = self._cell.output_size
        if self._output_layer is None:
            return size
        else:
            # To use layer's compute_output_shape, we need to convert the
            # RNNCell's output_size entries into shapes with an unknown
            # batch size.  We then pass this through the layer's
            # compute_output_shape and read off all but the first (batch)
            # dimensions to get the output size of the rnn with the layer
            # applied to the top.
            output_shape_with_unknown_batch = nest.map_structure(
                lambda s: tensor_shape.TensorShape([None]).concatenate(s),
                size)
            layer_output_shape = self._output_layer.compute_output_shape(
                output_shape_with_unknown_batch)
            return nest.map_structure(lambda s: s[1:], layer_output_shape)

    @property
    def output_size(self):
        # Return the cell output and the id
        return BasicDecoderOutput(
            rnn_output=self._rnn_output_size(),
            sample_id=self._helper.sample_ids_shape)

    @property
    def output_dtype(self):
        # Assume the dtype of the cell is the output_size structure
        # containing the input_state's first component's dtype.
        # Return that structure and the sample_ids_dtype from the helper.
        dtype = nest.flatten(self._initial_state)[0].dtype
        return BasicDecoderOutput(
            nest.map_structure(lambda _: dtype, self._rnn_output_size()),
            self._helper.sample_ids_dtype)

    def initialize(self, name=None):
        """Initialize the decoder.

        Args:
          name: Name scope for any created operations.

        Returns:
          `(finished, first_inputs, initial_state)`.
        """
        return self._helper.initialize() + (self._initial_state,)

    def step(self, time, inputs, state, name=None):
        """Perform a decoding step.

        Args:
          time: scalar `int32` tensor.
          inputs: A (structure of) input tensors.
          state: A (structure of) state tensors and TensorArrays.
          name: Name scope for any created operations.

        Returns:
          `(outputs, next_state, next_inputs, finished)`.
        """
        with ops.name_scope(name, "BasicDecoderStep", (time, inputs, state)):
            cell_outputs, cell_state, cell_score = self._cell(inputs, state)
            if self._output_layer is not None:
                cell_outputs = self._output_layer(cell_outputs)
            
            sample_ids = self._helper.sample(time=time,
                                             outputs=cell_outputs,
                                             state=cell_state)

            (finished, next_inputs, next_state) = self._helper.next_inputs(
	            time=time,
                outputs=cell_outputs,
                state=cell_state,
                sample_ids=sample_ids)
        outputs = BasicDecoderOutput(cell_outputs, sample_ids)

        return (outputs, next_state, next_inputs, finished, cell_score)


_transpose_batch_time = rnn._transpose_batch_time  # pylint: disable=protected-access
_zero_state_tensors = rnn_cell_impl._zero_state_tensors  # pylint: disable=protected-access


def _create_zero_outputs(size, dtype, batch_size):
    """Create a zero outputs Tensor structure."""

    def _create(s, d):
        return _zero_state_tensors(s, batch_size, d)

    return nest.map_structure(_create, size, dtype)


def my_dynamic_decode(decoder,
                      output_time_major=False,
                      impute_finished=False,
                      maximum_iterations=None,
                      parallel_iterations=32,
                      swap_memory=False,
                      scope=None):
    """Perform dynamic decoding with `decoder`.

    Calls initialize() once and step() repeatedly on the Decoder object.

    Args:
      decoder: A `Decoder` instance.
      output_time_major: Python boolean.  Default: `False` (batch major).  If
        `True`, outputs are returned as time major tensors (this mode is faster).
        Otherwise, outputs are returned as batch major tensors (this adds extra
        time to the computation).
      impute_finished: Python boolean.  If `True`, then states for batch
        entries which are marked as finished get copied through and the
        corresponding outputs get zeroed out.  This causes some slowdown at
        each time step, but ensures that the final state and outputs have
        the correct values and that backprop ignores time steps that were
        marked as finished.
      maximum_iterations: `int32` scalar, maximum allowed number of decoding
         steps.  Default is `None` (decode until the decoder is fully done).
      parallel_iterations: Argument passed to `tf.while_loop`.
      swap_memory: Argument passed to `tf.while_loop`.
      scope: Optional variable scope to use.

    Returns:
      `(final_outputs, final_state, final_sequence_lengths)`.

    Raises:
      TypeError: if `decoder` is not an instance of `Decoder`.
      ValueError: if `maximum_iterations` is provided but is not a scalar.
    """
    if not isinstance(decoder, seq2seq.Decoder):
        raise TypeError("Expected decoder to be type Decoder, but saw: %s" %
                        type(decoder))

    with variable_scope.variable_scope(scope, "decoder") as varscope:
        # Determine context types.
        ctxt = ops.get_default_graph()._get_control_flow_context()  # pylint: disable=protected-access
        is_xla = control_flow_util.GetContainingXLAContext(ctxt) is not None
        in_while_loop = (control_flow_util.GetContainingWhileContext(ctxt) is not None)
        # Properly cache variable values inside the while_loop.
        # Don't set a caching device when running in a loop, since it is possible
        # that train steps could be wrapped in a tf.while_loop. In that scenario
        # caching prevents forward computations in loop iterations from re-reading
        # the updated weights.
        if not context.executing_eagerly() and not in_while_loop:
            if varscope.caching_device is None:
                varscope.set_caching_device(lambda op: op.device)

        if maximum_iterations is not None:
            maximum_iterations = ops.convert_to_tensor(maximum_iterations, dtype=dtypes.int32, name="maximum_iterations")
            if maximum_iterations.get_shape().ndims != 0:
                raise ValueError("maximum_iterations must be a scalar")

        initial_finished, initial_inputs, initial_state = decoder.initialize()

        zero_outputs = _create_zero_outputs(decoder.output_size,
                                            decoder.output_dtype,
                                            decoder.batch_size)

        _, input_size = get_shape_list(initial_inputs)
        if is_xla and maximum_iterations is None:
            raise ValueError("maximum_iterations is required for XLA compilation.")
        if maximum_iterations is not None:
            initial_finished = math_ops.logical_or(initial_finished, 0 >= maximum_iterations)
        initial_sequence_lengths = array_ops.zeros_like(initial_finished, dtype=dtypes.int32)
        initial_time = constant_op.constant(0, dtype=dtypes.int32)
        initial_att_score = array_ops.zeros(
            shape=(decoder.batch_size, decoder.encoder_shape[1]),
            dtype=dtypes.float32
        )
        initial_seq_inputs = array_ops.zeros(
            shape=(decoder.batch_size, decoder.batch_size),
            dtype=dtypes.float32
        )

        batch_size, seq_len = decoder.encoder_shape[0], decoder.encoder_shape[1]

        def _shape(batch_size, from_shape):
            if (not isinstance(from_shape, tensor_shape.TensorShape) or
                    from_shape.ndims == 0):
                return tensor_shape.TensorShape(None)
            else:
                batch_size = tensor_util.constant_value(
                    ops.convert_to_tensor(
                        batch_size, name="batch_size"))
                return tensor_shape.TensorShape([batch_size]).concatenate(from_shape)

        dynamic_size = maximum_iterations is None or not is_xla

        def _create_ta(s, d):
            return tensor_array_ops.TensorArray(
                dtype=d,
                size=0 if dynamic_size else maximum_iterations,
                dynamic_size=dynamic_size,
                element_shape=_shape(decoder.batch_size, s))

        initial_outputs_ta = nest.map_structure(_create_ta, decoder.output_size,
                                                decoder.output_dtype)

        def condition(unused_time, unused_outputs_ta, unused_state, unused_inputs,
                      finished, unused_sequence_lengths, unused_att_score, unused_seq_inputs):
            return math_ops.logical_not(math_ops.reduce_all(finished))

        def body(time, outputs_ta, state, inputs, finished, sequence_lengths, att_scores, seq_inputs):
            """Internal while_loop body.

            Args:
              time: scalar int32 tensor.
              outputs_ta: structure of TensorArray.
              state: (structure of) state tensors and TensorArrays.
              inputs: (structure of) input tensors.
              finished: bool tensor (keeping track of what's finished).
              sequence_lengths: int32 tensor (keeping track of time of finish).

            Returns:
              `(time + 1, outputs_ta, next_state, next_inputs, next_finished,
                next_sequence_lengths)`.
              ```
            """
            (next_outputs, decoder_state, next_inputs,
             decoder_finished, att_score) = decoder.step(time, inputs, state)
            if decoder.tracks_own_finished:
                next_finished = decoder_finished
            else:
                next_finished = math_ops.logical_or(decoder_finished, finished)
            next_sequence_lengths = array_ops.where(
                math_ops.logical_not(finished),
                array_ops.fill(array_ops.shape(sequence_lengths), time + 1),
                sequence_lengths)

            nest.assert_same_structure(state, decoder_state)
            nest.assert_same_structure(outputs_ta, next_outputs)
            nest.assert_same_structure(inputs, next_inputs)

            # Zero out output values past finish
            if impute_finished:
                emit = nest.map_structure(
                    lambda out, zero: array_ops.where(finished, zero, out),
                    next_outputs,
                    zero_outputs)
            else:
                emit = next_outputs

            # Copy through states past finish
            def _maybe_copy_state(new, cur):
                # TensorArrays and scalar states get passed through.
                if isinstance(cur, tensor_array_ops.TensorArray):
                    pass_through = True
                else:
                    new.set_shape(cur.shape)
                    pass_through = (new.shape.ndims == 0)
                return new if pass_through else array_ops.where(finished, cur, new)

            if impute_finished:
                next_state = nest.map_structure(
                    _maybe_copy_state, decoder_state, state)
            else:
                next_state = decoder_state

            att_scores = tf.cond(time > 0, lambda: tf.concat([att_scores, att_score], 1), lambda: att_score)
            seq_inputs = tf.cond(time > 0, lambda: tf.concat([seq_inputs, next_inputs], 1), lambda: next_inputs)
            outputs_ta = nest.map_structure(lambda ta, out: ta.write(time, out),
                                            outputs_ta, emit)
            return (time + 1, outputs_ta, next_state, next_inputs, next_finished, next_sequence_lengths, att_scores, seq_inputs)

        res = control_flow_ops.while_loop(
            condition,
            body,
            loop_vars=(
                initial_time,
                initial_outputs_ta,
                initial_state,
                initial_inputs,
                initial_finished,
                initial_sequence_lengths,
                initial_att_score,
                initial_seq_inputs
            ),
            parallel_iterations=parallel_iterations,
            maximum_iterations=maximum_iterations,
            swap_memory=swap_memory)

        final_outputs_ta = res[1]
        final_state = res[2]
        final_sequence_lengths = res[5]
        att_scores = res[6]
        seq_inputs = res[7]

        # batch_size, max_sen_num, decoder_steps
        att_scores = tf.reshape(att_scores, [batch_size, seq_len, -1])
        seq_inputs = tf.reshape(seq_inputs, [batch_size, -1, input_size])

        final_outputs = nest.map_structure(lambda ta: ta.stack(), final_outputs_ta)

        try:
            final_outputs, final_state = decoder.finalize(
                final_outputs, final_state, final_sequence_lengths)
        except NotImplementedError:
            pass

        if not output_time_major:
            final_outputs = nest.map_structure(_transpose_batch_time, final_outputs)

    return final_outputs, final_state, final_sequence_lengths, att_scores, seq_inputs


def new_dynamic_decode(decoder,
                       sim_decoder,
                       output_time_major=False,
                       impute_finished=False,
                       maximum_iterations=None,
                       parallel_iterations=32,
                       swap_memory=False,
                       scope=None):
    """Perform dynamic decoding with `decoder`.

    Calls initialize() once and step() repeatedly on the Decoder object.

    Args:
      decoder: A `Decoder` instance.
      output_time_major: Python boolean.  Default: `False` (batch major).  If
        `True`, outputs are returned as time major tensors (this mode is faster).
        Otherwise, outputs are returned as batch major tensors (this adds extra
        time to the computation).
      impute_finished: Python boolean.  If `True`, then states for batch
        entries which are marked as finished get copied through and the
        corresponding outputs get zeroed out.  This causes some slowdown at
        each time step, but ensures that the final state and outputs have
        the correct values and that backprop ignores time steps that were
        marked as finished.
      maximum_iterations: `int32` scalar, maximum allowed number of decoding
         steps.  Default is `None` (decode until the decoder is fully done).
      parallel_iterations: Argument passed to `tf.while_loop`.
      swap_memory: Argument passed to `tf.while_loop`.
      scope: Optional variable scope to use.

    Returns:
      `(final_outputs, final_state, final_sequence_lengths)`.

    Raises:
      TypeError: if `decoder` is not an instance of `Decoder`.
      ValueError: if `maximum_iterations` is provided but is not a scalar.
    """
    if not isinstance(decoder, seq2seq.Decoder):
        raise TypeError("Expected decoder to be type Decoder, but saw: %s" % type(decoder))

    with variable_scope.variable_scope(scope, "decoder") as varscope:
        # Determine context types.
        ctxt = ops.get_default_graph()._get_control_flow_context()  # pylint: disable=protected-access
        is_xla = control_flow_util.GetContainingXLAContext(ctxt) is not None
        in_while_loop = (control_flow_util.GetContainingWhileContext(ctxt) is not None)
        # Properly cache variable values inside the while_loop.
        # Don't set a caching device when running in a loop, since it is possible
        # that train steps could be wrapped in a tf.while_loop. In that scenario
        # caching prevents forward computations in loop iterations from re-reading
        # the updated weights.
        if not context.executing_eagerly() and not in_while_loop:
            if varscope.caching_device is None:
                varscope.set_caching_device(lambda op: op.device)

        if maximum_iterations is not None:
            maximum_iterations = ops.convert_to_tensor(maximum_iterations, dtype=dtypes.int32, name="maximum_iterations")
            if maximum_iterations.get_shape().ndims != 0:
                raise ValueError("maximum_iterations must be a scalar")

        initial_finished, initial_inputs, initial_state = decoder.initialize()
        sim_initial_finished, sim_initial_inputs, sim_initial_state = sim_decoder.initialize()

        zero_outputs = _create_zero_outputs(decoder.output_size,
                                            decoder.output_dtype,
                                            decoder.batch_size)

        sim_zero_outputs = _create_zero_outputs(sim_decoder.output_size,
                                                sim_decoder.output_dtype,
                                                sim_decoder.batch_size)

        _, input_size = get_shape_list(initial_inputs)
        _, sim_input_size = get_shape_list(sim_initial_inputs)
        
        if is_xla and maximum_iterations is None:
            raise ValueError("maximum_iterations is required for XLA compilation.")
        if maximum_iterations is not None:
            initial_finished = math_ops.logical_or(initial_finished, 0 >= maximum_iterations)
            sim_initial_finished = math_ops.logical_or(sim_initial_finished, 0 >= maximum_iterations)
	        
        initial_sequence_lengths = array_ops.zeros_like(initial_finished, dtype=dtypes.int32)
        sim_initial_sequence_lengths = array_ops.zeros_like(sim_initial_finished, dtype=dtypes.int32)
        
        initial_time = constant_op.constant(0, dtype=dtypes.int32)
        
        initial_att_score = array_ops.zeros(
            shape=(decoder.batch_size, decoder.encoder_shape[1]),
            dtype=dtypes.float32
        )
        sim_initial_att_score = array_ops.zeros(
            shape=(sim_decoder.batch_size, sim_decoder.encoder_shape[1]),
            dtype=dtypes.float32
        )
        
        initial_seq_inputs = array_ops.zeros(
            shape=(decoder.batch_size, decoder.batch_size),
            dtype=dtypes.float32
        )
        sim_initial_seq_inputs = array_ops.zeros(
            shape=(sim_decoder.batch_size, sim_decoder.batch_size),
            dtype=dtypes.float32
        )
        
        batch_size, seq_len = decoder.encoder_shape[0], decoder.encoder_shape[1]
        _, sim_seq_len = sim_decoder.encoder_shape[0], sim_decoder.encoder_shape[1]

        def _shape(batch_size, from_shape):
            if (not isinstance(from_shape, tensor_shape.TensorShape) or from_shape.ndims == 0):
                return tensor_shape.TensorShape(None)
            else:
                batch_size = tensor_util.constant_value(ops.convert_to_tensor( batch_size, name="batch_size"))
                return tensor_shape.TensorShape([batch_size]).concatenate(from_shape)

        dynamic_size = maximum_iterations is None or not is_xla

        def _create_ta(s, d):
            return tensor_array_ops.TensorArray(
                dtype=d,
                size=0 if dynamic_size else maximum_iterations,
                dynamic_size=dynamic_size,
                element_shape=_shape(decoder.batch_size, s))

        initial_outputs_ta = nest.map_structure(_create_ta, decoder.output_size, decoder.output_dtype)
        sim_initial_outputs_ta = nest.map_structure(_create_ta, sim_decoder.output_size, sim_decoder.output_dtype)

        def condition(unused_time, unused_outputs_ta, unused_state, unused_inputs,
                      finished, unused_sequence_lengths, unused_att_score, unused_seq_inputs):
            return math_ops.logical_not(math_ops.reduce_all(finished))

        def body(time, outputs_ta, state, inputs, finished, sequence_lengths, att_scores, seq_inputs):
            """Internal while_loop body.

            Args:
              time: scalar int32 tensor.
              outputs_ta: structure of TensorArray.
              state: (structure of) state tensors and TensorArrays.
              inputs: (structure of) input tensors.
              finished: bool tensor (keeping track of what's finished).
              sequence_lengths: int32 tensor (keeping track of time of finish).

            Returns:
              `(time + 1, outputs_ta, next_state, next_inputs, next_finished,
                next_sequence_lengths)`.
              ```
            """
            (next_outputs, decoder_state, next_inputs, decoder_finished, att_score) = decoder.step(time, inputs, state)
            if decoder.tracks_own_finished:
                next_finished = decoder_finished
            else:
                next_finished = math_ops.logical_or(decoder_finished, finished)
            next_sequence_lengths = array_ops.where(
                math_ops.logical_not(finished),
                array_ops.fill(array_ops.shape(sequence_lengths), time + 1),
                sequence_lengths)

            nest.assert_same_structure(state, decoder_state)
            nest.assert_same_structure(outputs_ta, next_outputs)
            nest.assert_same_structure(inputs, next_inputs)

            # Zero out output values past finish
            if impute_finished:
                emit = nest.map_structure(
                    lambda out, zero: array_ops.where(finished, zero, out),
                    next_outputs,
                    zero_outputs)
            else:
                emit = next_outputs

            # Copy through states past finish
            def _maybe_copy_state(new, cur):
                # TensorArrays and scalar states get passed through.
                if isinstance(cur, tensor_array_ops.TensorArray):
                    pass_through = True
                else:
                    new.set_shape(cur.shape)
                    pass_through = (new.shape.ndims == 0)
                return new if pass_through else array_ops.where(finished, cur, new)

            if impute_finished:
                next_state = nest.map_structure(
                    _maybe_copy_state, decoder_state, state)
            else:
                next_state = decoder_state

            att_scores = tf.cond(time > 0, lambda: tf.concat([att_scores, att_score], 1), lambda: att_score)
            seq_inputs = tf.cond(time > 0, lambda: tf.concat([seq_inputs, next_inputs], 1), lambda: next_inputs)
            outputs_ta = nest.map_structure(lambda ta, out: ta.write(time, out),
                                            outputs_ta, emit)
            return (time + 1, outputs_ta, next_state, next_inputs, next_finished, next_sequence_lengths, att_scores, seq_inputs)

        res = control_flow_ops.while_loop(
            condition,
            body,
            loop_vars=(
                initial_time,
                initial_outputs_ta,
                initial_state,
                initial_inputs,
                initial_finished,
                initial_sequence_lengths,
                initial_att_score,
                initial_seq_inputs
            ),
            parallel_iterations=parallel_iterations,
            maximum_iterations=maximum_iterations,
            swap_memory=swap_memory)
        
        sim_res = control_flow_ops.while_loop(
            condition,
            body,
            loop_vars=(
                initial_time,
                sim_initial_outputs_ta,
                sim_initial_state,
                sim_initial_inputs,
                sim_initial_finished,
                sim_initial_sequence_lengths,
                sim_initial_att_score,
                sim_initial_seq_inputs
            ),
            parallel_iterations=parallel_iterations,
            maximum_iterations=maximum_iterations,
            swap_memory=swap_memory)
        
        final_outputs_ta = res[1]
        final_state = res[2]
        final_sequence_lengths = res[5]
        att_scores = res[6]
        seq_inputs = res[7]
        
        sim_final_outputs_ta = sim_res[1]
        sim_final_state = sim_res[2]
        sim_final_sequence_lengths = sim_res[5]
        sim_att_scores = sim_res[6]
        sim_seq_inputs = sim_res[7]

        # batch_size, max_sen_num, decoder_steps
        att_scores = tf.reshape(att_scores, [batch_size, seq_len, -1])
        seq_inputs = tf.reshape(seq_inputs, [batch_size, -1, input_size])
        final_outputs = nest.map_structure(lambda ta: ta.stack(), final_outputs_ta)

        sim_att_scores = tf.reshape(sim_att_scores, [batch_size, sim_seq_len, -1])
        sim_seq_inputs = tf.reshape(sim_seq_inputs, [batch_size, -1, sim_input_size])
        sim_final_outputs = nest.map_structure(lambda ta: ta.stack(), sim_final_outputs_ta)
        
        try:
            final_outputs, final_state = decoder.finalize(
                final_outputs, final_state, final_sequence_lengths)
        except NotImplementedError:
            pass
        
        try:
            sim_final_outputs, sim_final_state = decoder.finalize(
	            sim_final_outputs, sim_final_state, sim_final_sequence_lengths)
        except NotImplementedError:
            pass
        
        if not output_time_major:
            final_outputs = nest.map_structure(_transpose_batch_time, final_outputs)
            sim_final_outputs = nest.map_structure(_transpose_batch_time, sim_final_outputs)

    return final_outputs, sim_final_outputs, final_state, sim_final_state, final_sequence_lengths, sim_final_sequence_lengths, att_scores, sim_att_scores, seq_inputs, sim_seq_inputs


class LuongAttention(_BaseAttentionMechanism):

  def __init__(self,
               num_units,
               memory,
               memory_sequence_length=None,
               scale=False,
               probability_fn=None,
               score_mask_value=None,
               dtype=None,
               custom_key_value_fn=None,
               name="LuongAttention"):

    # For LuongAttention, we only transform the memory layer; thus
    # num_units **must** match expected the query depth.
    if probability_fn is None:
      probability_fn = nn_ops.softmax
    if dtype is None:
      dtype = dtypes.float32
    wrapped_probability_fn = lambda score, _: probability_fn(score)
    super(LuongAttention, self).__init__(
        query_layer=None,
        memory_layer=layers_core.Dense(
            num_units, name="memory_layer", use_bias=False, dtype=dtype),
        memory=memory,
        probability_fn=wrapped_probability_fn,
        memory_sequence_length=memory_sequence_length,
        score_mask_value=score_mask_value,
        #custom_key_value_fn=custom_key_value_fn,
        name=name)
    self._num_units = num_units
    self._scale = scale
    self._name = name

  def __call__(self, query, state):

    with variable_scope.variable_scope(None, "luong_attention", [query]):
      attention_g = None
      if self._scale:
        attention_g = variable_scope.get_variable(
            "attention_g",
            dtype=query.dtype,
            initializer=init_ops.ones_initializer,
            shape=())
      score = _luong_score(query, self._keys, attention_g)
    alignments = self._probability_fn(score, state)
    next_state = alignments
    return alignments, next_state

