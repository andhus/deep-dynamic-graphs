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


def get_dynamic_batch():
    return [get_input_sequence() for _ in range(batch_size)]


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

    feature_sequence = td.Map(feature)

    lstm_cell = td.ScopedLayer(
        tf.contrib.rnn.BasicLSTMCell(num_units=lstm_state_size),
        'lstm_cell'
    )
    lstm_output = feature_sequence >> td.RNN(lstm_cell, name='lstm')

    last_state = lstm_output >> GetLastState()

    compiler = td.Compiler.create((last_state,))
    [last_state_tensor] = compiler.output_tensors

    return compiler, last_state_tensor


def get_tf_static():
    inputs_A = tf.placeholder('float32', shape=(None, None, input_size), name='inputs_A')
    inputs_B = tf.placeholder('float32', shape=(None, None, input_size), name='inputs_B')
    mask_A = tf.placeholder('float32', shape=(None, None, 1), name='mask_A')
    mask_B = tf.placeholder('float32', shape=(None, None, 1), name='mask_B')

    inputs = [inputs_A, inputs_B]
    masks = [mask_A, mask_B]

    feature_from_A = tf.layers.dense(inputs_A, feature_size, activation=tf.nn.relu)
    feature_from_B = tf.layers.dense(inputs_B, feature_size, activation=tf.nn.relu)
    feature_from_A_masked = feature_from_A * mask_A
    feature_from_B_masked = feature_from_B * mask_B

    feature_sequences = feature_from_A_masked + feature_from_B_masked

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=lstm_state_size)
    feature_sequences_ = tf.unstack(feature_sequences, sequence_length, 1)
    outputs, states = tf.contrib.rnn.static_rnn(
        lstm_cell,
        feature_sequences_,
        dtype=tf.float32
    )

    last_state_tensor = outputs[-1]

    return inputs, masks, last_state_tensor


dynamic_batch = get_dynamic_batch()
compiler, dynamic_last_state = get_tffold_dynamic()
dynamic_feed_dict = compiler.build_feed_dict((dynamic_batch,))


def run_dyn():
    return sess.run(dynamic_last_state, dynamic_feed_dict)


static_batch = get_static_batch(dynamic_batch)
inputs, masks = static_batch
input_tensors, mask_tensors, static_last_state = get_tf_static()
static_feed_dict = dict(
    zip(input_tensors + mask_tensors, inputs.values() + masks.values())
)


def run_static():
    return sess.run(static_last_state, static_feed_dict)


sess.run(tf.global_variables_initializer())
