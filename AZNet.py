# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet in Tensorflow
Tested in Tensorflow 1.4 and 1.5

@author: Xiang Zhong
"""

import numpy as np
import tensorflow as tf


class PolicyValueNet():
    def __init__(self, model_file=None):
        self.input_states = tf.placeholder(
                tf.float32, shape=[None, 9, 12, 12])
        self.input_states_reshaped = tf.reshape(
                self.input_states, [-1, 12, 12, 9])
        self.conv1 = tf.layers.conv2d(inputs=self.input_states_reshaped,
                                      filters=32, kernel_size=[3, 3],
                                      padding="same", activation=tf.nn.relu)
        self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=64,
                                      kernel_size=[3, 3], padding="same",
                                      activation=tf.nn.relu)
        self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=128,
                                      kernel_size=[3, 3], padding="same",
                                      activation=tf.nn.relu)
        self.action_conv = tf.layers.conv2d(inputs=self.conv3, filters=4,
                                            kernel_size=[1, 1], padding="same",
                                            activation=tf.nn.relu)
        self.action_conv_flat = tf.reshape(
                self.action_conv, [-1, 4 * 12 * 12])
        self.action_fc = tf.layers.dense(inputs=self.action_conv_flat,
                                         units=12*12*2*12*12*2,
                                         activation=tf.nn.log_softmax)
        self.evaluation_conv = tf.layers.conv2d(inputs=self.conv3, filters=2,
                                                kernel_size=[1, 1],
                                                padding="same",
                                                activation=tf.nn.relu)
        self.evaluation_conv_flat = tf.reshape(
                self.evaluation_conv, [-1, 2 * 12 * 12])
        self.evaluation_fc1 = tf.layers.dense(inputs=self.evaluation_conv_flat,
                                              units=64, activation=tf.nn.relu)
        self.evaluation_fc2 = tf.layers.dense(inputs=self.evaluation_fc1,
                                              units=1, activation=tf.nn.tanh)


        self.labels = tf.placeholder(tf.float32, shape=[None, 1])
        self.value_loss = tf.losses.mean_squared_error(self.labels,
                                                       self.evaluation_fc2)
        self.mcts_probs = tf.placeholder(tf.float32, shape=[None, 12*12*2*12*12*2]) ##12*12*2*12*12*2
        self.policy_loss = tf.negative(tf.reduce_mean(
                tf.reduce_sum(tf.multiply(self.mcts_probs, self.action_fc), 1)))
        l2_penalty_beta = 1e-4
        vars = tf.trainable_variables()
        l2_penalty = l2_penalty_beta * tf.add_n(
            [tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()])
        self.loss = self.value_loss + self.policy_loss + l2_penalty

        self.learning_rate = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss)

        self.session = tf.Session()

        self.entropy = tf.negative(tf.reduce_mean(
                tf.reduce_sum(tf.exp(self.action_fc) * self.action_fc, 1)))

        init = tf.global_variables_initializer()
        self.session.run(init)

        self.saver = tf.train.Saver()
        if model_file is not None:
            self.restore_model(model_file)

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        log_act_probs, value = self.session.run(
                [self.action_fc, self.evaluation_fc2],
                feed_dict={self.input_states: state_batch}
                )
        act_probs = np.exp(log_act_probs)
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        winner_batch = np.reshape(winner_batch, (-1, 1))
        """ kokuta - 2018/06/29 
        self.session.run() で不具合発生 tf.placestructure の self.mcts_probs と mcts_probs の要素のサイズが異なります。
        self.mcts_probs の 要素のサイズ : (289,) = (17*17,)
        mcts_probs の 要素のサイズ      : (42,) board の available に起因? 要検討
        """
        loss, entropy, _ = self.session.run(
                [self.loss, self.entropy, self.optimizer],
                feed_dict={self.input_states: state_batch,
                           self.mcts_probs: mcts_probs,
                           self.labels: winner_batch,
                           self.learning_rate: lr})
        return loss, entropy

    def save_model(self, model_path):
        self.saver.save(self.session, model_path)

    def restore_model(self, model_path):
        self.saver.restore(self.session, model_path)
