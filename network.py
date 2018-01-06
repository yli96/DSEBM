#!/usr/bin/python2.7
#-*- coding: utf-8 -*-

import os
import sys
import logging
import argparse

import tensorflow as tf
import metrics

FLAGS = tf.app.flags.FLAGS

class EBM(object):
    """
    Energy based model
    http://proceedings.mlr.press/v48/zhai16.pdf
    """
    def __init__(self):
        self._variables = []

    def model_variables(self):
        return self._variables

    def _get_variable(self, shape, name, initializer=None):
        if initializer is None:
            initializer = tf.truncated_normal_initializer(stddev=0.0002)

        var = tf.get_variable(name=name, shape=shape, dtype=tf.float32, initializer=initializer)

        if var not in self._variables:
            self._variables.append(var)

        return var

    def energy(self, images):
        """
        各imageのEnergyを返す
        :param tf.Tensorf images: preprocessed images float32 [NHWC] 
        :return: 
         tf.Tensor float32 [N]
        """

        with tf.variable_scope("energy"):
            energy = self._energy(images)
        return energy

    def _energy(self, images):
        """
        各imageのEnergyを返す
        :param tf.Tensorf images: preprocessed images float32 [NHWC] 
        :return:
         tf.Tensor float32 [N]
        """
        # NHWC => NCHW
        images = tf.transpose(images, perm=[0, 3, 1, 2])
        features = images

        for idx_layer in range(1):
            layer_name = "layer{}".format(idx_layer)
            with tf.variable_scope(layer_name):
                filter_size = 7
                in_channel = features.shape.as_list()[1]
                out_channel = 64

                filter = self._get_variable(shape=[filter_size, filter_size, in_channel, out_channel], name="W")
                bias = self._get_variable(shape=[out_channel], name="b", initializer=tf.constant_initializer())

                features = tf.nn.conv2d(features, filter, strides=[1,1,1,1], padding="VALID", data_format="NCHW")
                features = tf.nn.bias_add(features, bias, data_format="NCHW")

                # MEMO: sigmoid関数を使うことで損失が大幅に落ちる (損失の大小がモデルのクオリティではない)
                #features = tf.nn.relu(features)
                features = tf.nn.sigmoid(features)

                features = tf.nn.max_pool(features, ksize=[1,1,2,2], strides=[1,1,2,2], padding="VALID", data_format="NCHW")

        with tf.variable_scope("fc"):
            C, H, W = features.shape.as_list()[1:]
            in_channel = C*H*W
            # MEMO: use K_L = 1 in (8) of the paper
            out_channel = 1
            features = tf.reshape(features, shape=[-1, in_channel])

            filter = self._get_variable(shape=[in_channel, out_channel], name="W")
            bias = self._get_variable(shape=[out_channel], name="b", initializer=tf.constant_initializer())

            features = tf.add(tf.matmul(features, filter), bias, name="E2")
            features = tf.squeeze(features)
            E2 = features

        with tf.variable_scope("prior_dist"):
            # TODO: we can analytically get this prior. (mean of input images)
            # MEMO: 画像だったらあんまり意味ないような (せめてピクセル平均にしたい)
            C, H, W = images.shape.as_list()[1:]
            in_channel = C*H*W
            flatten = tf.reshape(images, shape=[-1, in_channel])

            prior = self._get_variable(shape=[in_channel], name="prior", initializer=tf.constant_initializer(0.5))
            #prior = self._get_variable(shape=[1], name="prior", initializer=tf.constant_initializer(0.5))

            # MEMO: 絶対に必要な項 (少なくともmnistでは)
            E1 = tf.multiply(0.5, tf.reduce_sum(tf.square(flatten - prior), axis=1), name="E1")

            tf.summary.histogram("E1", E1)

        energy = tf.subtract(E1, E2, name="Energy")

        return energy

    def reconstruct(self, images):
        """
        reconstructed imagesを返す
        :param tf.Tensor images: preprocessed image float32 [NHWC]
        :rtype: tf.Tensor
        :return: 
          reconstructed images float32 [NHWC]
        """
        energy = self.energy(images)

        tf.summary.histogram("images", images)

        with tf.variable_scope("reconstruct"):
            reconstructed = tf.subtract(images, tf.gradients(energy, images), name="reconst")

        tf.summary.histogram("reconst", reconstructed)

        return reconstructed

    def _preprocess(self, images):
        images = tf.cast(images, dtype=tf.float32)
        return tf.divide(images, 255, name="preprocessed_images")

    def loss(self, images):
        """
        このモデルの損失関数を返す
        
        :param tf.Tensor images: 元画像 [NHWC] float32 
        :return: 
        """
        noise = tf.random_normal(shape=images.shape, stddev=FLAGS.noise)
        noised_images = tf.add(images, noise, name="noised")

        reconstructed = self.reconstruct(noised_images)

        total_loss = tf.reduce_mean(tf.reduce_sum(tf.square(images - reconstructed), axis=[1,2,3]), name="reconst_error")
        metrics.add_to_loss_view(total_loss)

        return total_loss

    def save_saver_dict(self):
        """
        get dict variable passed to Saver for saving variables.
        :return: 
        """
        return {var.op.name : var for var in self.model_variables()}

    def load_saver_dict(self):
        """
        get dict variable passed to Saver for loading variables.
        :return: 
        """
        return {var.op.name : var for var in self.model_variables()}


if __name__ == "__main__":
    from dataset.mnist import MNIST

    dataset = MNIST(is_train=True)
    images, labels = dataset.dummy_inputs()
    model = EBM()

    loss = model.loss(images)

    variables = model.model_variables()

    for cur_var in variables:
        print ("{} : {}".format(cur_var.op.name, cur_var.shape))
