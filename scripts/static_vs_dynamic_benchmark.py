from __future__ import division, print_function

import random

import numpy as np
import tensorflow as tf

sess = tf.InteractiveSession()
import tensorflow_fold as td


input_types = ['A', 'B']
input_size = 256
feature_size = 256
lstm_state_size = 128
sequence_length = 1000
batch_size = 16


def get_input_sequence():
    input_sequence = []
    for _ in range(sequence_length):
        input_sequence.append(
            {
                'type': random.sample(input_types, 1)[0],
                'data': np.random.randn(input_size)
            }
        )

    return input_sequence


def get_target_sequence():
    return list(
        (np.random.randn(sequence_length).reshape(-1, 1) > 0).astype(float)
    )


def get_dynamic_batch():
    input_sequence = [get_input_sequence() for _ in range(batch_size)]
    target_sequence = [get_target_sequence() for _ in range(batch_size)]
    return input_sequence, target_sequence


def get_static_batch(dynamic_batch=None):
    dynamic_batch = dynamic_batch or get_dynamic_batch()
    inputs = {
        'A': np.zeros((batch_size, sequence_length, input_size)),
        'B': np.zeros((batch_size, sequence_length, input_size))
    }
    masks = {
        'A': np.zeros((batch_size, sequence_length, 1)),
        'B': np.zeros((batch_size, sequence_length, 1))
    }

    for n, in_seq in enumerate(dynamic_batch):
        for t, input_ in enumerate(in_seq):
            inputs[input_['type']][n, t, :] = np.array(input_['data'])
            masks[input_['type']][n, t, 0] = 1

    return inputs, masks

    # feed_dict = {
    #     inputs_A: inputs['A'],
    #     mask_A: masks['A'],
    #     inputs_B: inputs['B'],
    #     mask_B: masks['B']
    # }


def get_tffold_dynamic():
    feature_from_A = td.Vector(input_size) >> td.FC(feature_size)
    feature_from_B = td.Vector(input_size) >> td.FC(feature_size)
    feature = td.OneOf(
        key_fn=lambda x: x['type'],
        case_blocks={
            'A': td.GetItem('data') >> feature_from_A,
            'B': td.GetItem('data') >> feature_from_B
        }
    )

    def GetLastState():
        """Composition of blocks that gets last state vector from LSTM output"""
        return td.GetItem(1) >> td.GetItem(1)


    def SequenceLoss(loss_f):
        """ Expects (seq, seq)
        """
        return td.Zip() >> td.Map(td.Function(loss_f)) >> td.Reduce(td.Sum())


    def sigmoid_cross_entropy_from_logits(y_pred, y_true):
        return tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true,
            logits=y_pred
        )

    feature_sequence = td.Map(feature)

    lstm_cell = td.ScopedLayer(
        tf.contrib.rnn.BasicLSTMCell(num_units=lstm_state_size),
        'lstm_cell'
    )
    lstm_state_sequence = (
        feature_sequence >> td.RNN(lstm_cell, name='lstm') >> td.GetItem(0)
    )
    output_sequence = lstm_state_sequence >> td.Map(td.FC(1, activation=None))
    target_sequence = td.Map(td.Vector(1))

    loss = td.Record((output_sequence, target_sequence)) >> SequenceLoss(
        sigmoid_cross_entropy_from_logits
    )

    compiler = td.Compiler.create((loss,))
    [loss_tensor] = compiler.output_tensors

    return compiler, loss_tensor


def get_tf_static():
    inputs_A = tf.placeholder('float32', shape=(None, None, input_size), name='inputs_A')
    inputs_B = tf.placeholder('float32', shape=(None, None, input_size), name='inputs_B')
    mask_A = tf.placeholder('float32', shape=(None, None, 1), name='mask_A')
    mask_B = tf.placeholder('float32', shape=(None, None, 1), name='mask_B')

    inputs = [inputs_A, inputs_B]
    masks = [mask_A, mask_B]

    target = tf.placeholder('float32', shape=(None, None, 1), name='target')

    feature_from_A = tf.layers.dense(inputs_A, feature_size, activation=tf.nn.relu)
    feature_from_B = tf.layers.dense(inputs_B, feature_size, activation=tf.nn.relu)
    feature_from_A_masked = feature_from_A * mask_A
    feature_from_B_masked = feature_from_B * mask_B

    feature_sequences = feature_from_A_masked + feature_from_B_masked

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=lstm_state_size)
    feature_sequences_ = tf.unstack(feature_sequences, sequence_length, axis=1)
    lstm_state_sequence, _last_states = tf.contrib.rnn.static_rnn(
        lstm_cell,
        feature_sequences_,
        dtype=tf.float32
    )
    lstm_state_sequence_tensor = tf.stack(lstm_state_sequence, axis=1)
    output_sequence = tf.layers.dense(lstm_state_sequence_tensor, 1, activation=None)

    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=target,
            logits=output_sequence
        )
    )

    return inputs, masks, target, loss


dyn_in_seq, dyn_target_seq = get_dynamic_batch()
compiler, dynamic_loss = get_tffold_dynamic()
dynamic_feed_dict = compiler.build_feed_dict(
    (zip(dyn_in_seq, dyn_target_seq),)
)


def run_dyn():
    return sess.run(dynamic_loss, dynamic_feed_dict)


static_batch = get_static_batch(dyn_in_seq)
inputs, masks = static_batch

input_tensors, mask_tensors, target_tensor, static_loss = get_tf_static()
static_feed_dict = dict(
    zip(input_tensors + mask_tensors, inputs.values() + masks.values())
)
static_feed_dict[target_tensor] = np.array(dyn_target_seq)


def run_static():
    return sess.run(static_loss, static_feed_dict)


sess.run(tf.global_variables_initializer())
