#!/usr/bin/python2.7
#-*- coding: utf-8 -*-

import os
import sys
import logging
import argparse
import time
import shutil

import tensorflow as tf

import config
from dataset.mnist import MNIST
from network import EBM
import metrics

FLAGS = tf.app.flags.FLAGS

def add_summaries_op():
    loss_average_op = metrics.add_loss_summaries()

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    return loss_average_op

def train():
    logger = logging.getLogger(__name__)

    dataset = MNIST(is_train=True, batch_size=FLAGS.batch_size)

    ### Network definition
    images, labels = dataset.dummy_inputs()
    ebm = EBM()
    loss = ebm.loss(images)

    ### Train definition
    global_step = tf.train.create_global_step()

    lr = tf.train.exponential_decay(FLAGS.lr,
                                    global_step,
                                    FLAGS.decay_steps,
                                    FLAGS.decay_rate,
                                    staircase=True)

    opt = tf.train.AdamOptimizer(lr)
    apply_gradient_op = opt.minimize(loss, global_step=global_step)
    loss_average_op = add_summaries_op()

    train_op = tf.group(apply_gradient_op, loss_average_op)

    #### Session setting
    save_dict = ebm.save_saver_dict()
    saver = tf.train.Saver(save_dict)

    saver_hook = tf.train.CheckpointSaverHook(saver=saver, checkpoint_dir=FLAGS.dir_parameter,
                                              save_steps=5000)

    summary_hook = tf.train.SummarySaverHook(
            summary_op=tf.summary.merge_all(),
            output_dir=FLAGS.dir_train,
            save_steps=500)

    nan_hook = tf.train.NanTensorHook(loss_tensor=loss)
    hooks = [nan_hook, summary_hook, saver_hook, tf.train.StopAtStepHook(last_step=FLAGS.max_steps)]

    gpu_options = tf.GPUOptions(allow_growth=True)
    session_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)

    with tf.train.SingularMonitoredSession(config=session_config, hooks=hooks) as sess:
        num_iter = 0

        while not sess.should_stop():
            num_iter += 1

            start_time = time.time()

            cur_images, cur_labels = dataset.next_batch()
            cur_loss, _ = sess.run([loss, train_op],
                                   feed_dict={images:cur_images, labels:cur_labels})

            if num_iter % 100 == 0:
                duration = time.time() - start_time
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration

                sec_per_batch = float(duration)

                logger.info("step = {} loss = {:.2f} ({:.1f} examples/sec; {:.1f} sec/batch)"
                            .format(num_iter, cur_loss, examples_per_sec, sec_per_batch))

def main(argv=None):
    config.print_config()

    if os.path.exists(FLAGS.dir_train):
        shutil.rmtree(FLAGS.dir_train)
    os.makedirs(FLAGS.dir_train)

    if os.path.exists(FLAGS.dir_parameter):
        shutil.rmtree(FLAGS.dir_parameter)
    os.makedirs(FLAGS.dir_parameter)

    train()

if __name__ == "__main__":
    tf.app.run()
