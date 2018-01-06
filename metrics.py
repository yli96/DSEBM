#!/usr/bin/python2.7
#-*- coding: utf-8 -*-

"""
loss management module
"""

import tensorflow as tf

COLLECTION_LOSS_VIEW = "loss_view"
LOSS_AVERAGE_DECAY=0.99

def add_to_loss_view(loss):
    """
    add loss variable to the watch collecition.
    
    :param tf.Tensor loss: loss Tensor   
    :return: 
    """
    tf.add_to_collection(name=COLLECTION_LOSS_VIEW, value=loss)

def add_loss_summaries():
    """
    add loss summaries to a graph.
    This collects losses from values passed to add_to_loss_view.
    
    :rtype: tf.Operator
    :return: 
      loss average op
    """
    loss_averages = tf.train.ExponentialMovingAverage(LOSS_AVERAGE_DECAY, name='avg', zero_debias=True)
    losses = tf.get_collection(COLLECTION_LOSS_VIEW)
    loss_averages_op = loss_averages.apply(losses)

    for l in losses:
        tf.summary.scalar(name=l.op.name, tensor=l)
        tf.summary.scalar(name=l.op.name + "_avg", tensor=loss_averages.average(l))

    return loss_averages_op
