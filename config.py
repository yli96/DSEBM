#!/usr/bin/python2.7
#-*- coding: utf-8 -*-

import logging
import tensorflow as tf
import pprint



###########  INPUT  ####################
DATA_ROOT="./data/"

tf.app.flags.DEFINE_integer('batch_size', 128,
                            """mini-batch size""")

############## OUTPUT ######################
tf.app.flags.DEFINE_string('dir_train', './out/train_log',
                           """train log directory""")

tf.app.flags.DEFINE_string('dir_eval', './out/eval_log',
                           """eval log directory""")

tf.app.flags.DEFINE_string('dir_parameter', './out/parameter',
                           """parameter directory""")

############# MODEL ######################

# noise level of denoise auto encoder
tf.app.flags.DEFINE_float('noise', 0.4,
                           """noise level of denoising auto encoder""")


############ OPTIMIZE ###################

VARIABLE_AVERAGE_DECAY=0.999

tf.app.flags.DEFINE_integer('max_steps', 30000,
                            """max_steps""")

tf.app.flags.DEFINE_float('lr', 2.0e-4,
                            """initial learning rate.""")
tf.app.flags.DEFINE_float('decay_rate', 0.1,
                            """decay rate.""")
tf.app.flags.DEFINE_integer('decay_steps', 10000,
                            """decay_steps""")

############ LOGGING ###################


def print_config():
    logger = logging.getLogger(__name__)
    FLAGS = tf.app.flags.FLAGS
    msg = pprint.pformat(FLAGS.__flags)
    logger.info(msg)

def get_config_line():
    FLAGS = tf.app.flags.FLAGS
    return str(FLAGS.__flags)

logging.basicConfig(datefmt="%d/%Y %I:%M:%S", level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)s) %(message)s'
                    )