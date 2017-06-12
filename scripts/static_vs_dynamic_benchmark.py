from __future__ import division, print_function

import time
import random

import numpy as np
import tensorflow as tf

import tensorflow_fold as td

sess = tf.InteractiveSession()


input_types = ['A', 'B', 'C']
# input_sizes = [256, 256, 256]
input_sizes = [128, 128, 128]
input_type_to_size = dict(zip(input_types, input_sizes))
mlp_h_size = 512
lstm_feature_size = 256
lstm_state_size = 128
sequence_length = 500
batch_size = 16


def get_input_sequence():
    input_sequence = []
    for _ in range(sequence_length):
        type_ = random.sample(input_types, 1)[0]
        data = np.random.randn(input_type_to_size[type_])
        input_sequence.append({'type': type_, 'data': data})

    return input_sequence


def get_target_sequence():
    return list(
        (np.random.randn(sequence_length).reshape(-1, 1) > 0).astype(float)
    )


def get_dynamic_batch():
    input_sequences = [get_input_sequence() for _ in range(batch_size)]
    target_sequences = [get_target_sequence() for _ in range(batch_size)]
    dynamic_batch = zip(input_sequences, target_sequences)

    return dynamic_batch


def get_static_batch(dynamic_batch=None):
    dynamic_batch = dynamic_batch or get_dynamic_batch()
    # unpack input, target pairs:
    [input_sequences, target_sequences] = map(list, zip(*dynamic_batch))
    type_to_input = {
        type_: np.zeros(
            (batch_size, sequence_length, input_type_to_size[type_])
        ) for type_ in input_types
    }
    type_to_mask = {
        type_: np.zeros(
            (batch_size, sequence_length, 1)
        ) for type_ in input_types
    }

    for n, input_sequence in enumerate(input_sequences):
        for t, input_ in enumerate(input_sequence):
            type_to_input[input_['type']][n, t, :] = input_['data']
            type_to_mask[input_['type']][n, t, 0] = 1

    input_batches = [type_to_input[type_] for type_ in input_types]
    mask_batches = [type_to_mask[type_] for type_ in input_types]
    target_batch = np.array(target_sequences)

    return input_batches, mask_batches, target_batch


def get_tffold_dynamic():
    features = [
        (
            td.Vector(input_type_to_size[type_]) >>
            td.FC(mlp_h_size) >>
            td.FC(lstm_feature_size)
        ) for type_ in input_types
    ]
    lstm_feature = td.OneOf(
        key_fn=lambda x: x['type'],
        case_blocks={
            type_: td.GetItem('data') >> feature
            for type_, feature in zip(input_types, features)
        }
    )

    def SequenceLoss(loss_f):
        """ Expects (seq, seq)
        """
        return td.Zip() >> td.Map(td.Function(loss_f)) >> td.Mean()

    def sigmoid_cross_entropy_from_logits(y_pred, y_true):
        return tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true,
            logits=y_pred
        )

    lstm_feature_sequence = td.Map(lstm_feature)

    lstm_cell = td.ScopedLayer(
        tf.contrib.rnn.BasicLSTMCell(num_units=lstm_state_size),
        'lstm_cell'
    )
    lstm_state_sequence = (
        lstm_feature_sequence >>
        td.RNN(lstm_cell, name='lstm') >>
        td.GetItem(0)
    )
    output_sequence = lstm_state_sequence >> td.Map(td.FC(1, activation=None))
    target_sequence = td.Map(td.Vector(1))

    loss = td.Record((output_sequence, target_sequence)) >> SequenceLoss(
        sigmoid_cross_entropy_from_logits
    )

    compiler = td.Compiler.create(loss)
    [loss_per_sample_tensor] = compiler.output_tensors
    loss_tensor = tf.reduce_mean(loss_per_sample_tensor)

    return compiler, loss_tensor


def get_tf_static():
    inputs = [
        tf.placeholder(
            'float32',
            shape=(None, None, input_type_to_size[type_]),
            name='input_{}'.format(type_)
        ) for type_ in input_types
    ]
    masks = [
        tf.placeholder(
            'float32',
            shape=(None, None, 1),
            name='mask_{}'.format(type_)
        ) for type_ in input_types
    ]

    target = tf.placeholder('float32', shape=(None, None, 1), name='target')

    features = [
        tf.layers.dense(
            tf.layers.dense(input_, mlp_h_size, activation=tf.nn.relu),
            lstm_feature_size,
            activation=tf.nn.relu
        ) for input_ in inputs
    ]
    masked_features = [
        feature * mask for feature, mask in zip(features, masks)
    ]

    lstm_feature_sequence = sum(masked_features)

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=lstm_state_size)
    feature_sequences_ = tf.unstack(
        lstm_feature_sequence,
        sequence_length,
        axis=1
    )
    lstm_state_sequence_list, _last_states = tf.contrib.rnn.static_rnn(
        lstm_cell,
        feature_sequences_,
        dtype=tf.float32
    )
    lstm_state_sequence = tf.stack(lstm_state_sequence_list, axis=1)
    output_sequence = tf.layers.dense(lstm_state_sequence, 1, activation=None)

    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=target,
            logits=output_sequence
        )
    )

    return inputs, masks, target, loss


# DYNAMIC
dynamic_batch = get_dynamic_batch()
compiler, dynamic_loss = get_tffold_dynamic()
dynamic_feed_dict = compiler.build_feed_dict(dynamic_batch)
dynamic_train_op = tf.train.AdamOptimizer().minimize(dynamic_loss)


def run_dynamic_fwd(n_batches=1):
    return [
        sess.run(dynamic_loss, dynamic_feed_dict)
        for _ in range(n_batches)
    ]


def run_dynamic_train(n_batches=1):
    return [
        sess.run(dynamic_train_op, dynamic_feed_dict)
        for _ in range(n_batches)
    ]


# STATIC
static_batch = get_static_batch(dynamic_batch)
input_batches, mask_batches, target_batch = static_batch
input_tensors, mask_tensors, target_tensor, static_loss = get_tf_static()
static_feed_dict = dict(
    zip(input_tensors + mask_tensors, input_batches + mask_batches)
)
static_feed_dict[target_tensor] = target_batch
static_train_op = tf.train.AdamOptimizer().minimize(static_loss)


def run_static_fwd(n_batches=1):
    return [
        sess.run(static_loss, static_feed_dict)
        for _ in range(n_batches)
    ]


def run_static_train(n_batches=1):
    return [
        sess.run(static_train_op, static_feed_dict)
        for _ in range(n_batches)
    ]


def benchmark(n_batches=10):
    print('Running benchmark for {} batches'.format(n_batches))

    print('\nfwd pass...')

    start = time.time()
    run_dynamic_fwd(n_batches)
    end = time.time()
    dyn_elapsed = end - start
    print('dynamic fwd: {}s'.format(round(dyn_elapsed, 3)))

    start = time.time()
    run_static_fwd(n_batches)
    end = time.time()
    stat_elapsed = end - start
    print('static fwd: {}s'.format(round(stat_elapsed, 3)))

    print('dynamic/static fwd: {}'.format(dyn_elapsed/stat_elapsed))


    print('\ntraining...')

    start = time.time()
    run_dynamic_train(n_batches)
    end = time.time()
    dyn_elapsed = end - start
    print('dynamic train: {}s'.format(round(dyn_elapsed, 3)))

    start = time.time()
    run_static_train(n_batches)
    end = time.time()
    stat_elapsed = end - start
    print('static train: {}s'.format(round(stat_elapsed, 3)))

    print('dynamic/static train: {}'.format(dyn_elapsed / stat_elapsed))


sess.run(tf.global_variables_initializer())
