# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import logging

import numpy as np
import cv2

import tensorflow as tf

import config
from dataset.mnist_splitter import DIR_MNIST

FLAGS = tf.app.flags.FLAGS

class MNIST(object):

    def __init__(self, is_train, batch_size=64, dir_root=DIR_MNIST):
        """
        In train mode:
          inlier only
          data is shuffled
          Each data is used many times.
        In test mode:
          inlier and outlier
          data is NOT shuffled
          Each data is used one time.
        
        :param boolean is_train: train mode or not
        :param int batch_size: batch size
        """
        self.is_train = is_train
        self.batch_size = batch_size
        self._completed = False

        self._images, self._labels = self._load(dir_root)

        self._num_data = len(self._images)
        self._reset_indices()

    def _load(self, dir_root):
        """
        ディレクトリから画像を読み込む
        :param dir_root: 
        :return:
          (images, labels)
          label == 0 : indicates INLIER
          label == 1 : indicates OUTLIER
        """
        logger = logging.getLogger(__name__)

        if self.is_train:
            dir_train = os.path.join(dir_root, "train")
            list_path_images = glob.glob(dir_train + "/*.png")

            images = []
            for cur_path in list_path_images:
                cur_image = cv2.imread(cur_path, cv2.IMREAD_GRAYSCALE)
                cur_image = cur_image.reshape(cur_image.shape + (1,))
                images.append(cur_image)

            labels = [0] * len(images)

            logger.info("load {} images".format(len(labels)))

            return np.array(images), np.array(labels)
        else:
            dir_test = os.path.join(dir_root, "test")

            ## LOAD INLIER
            dir_inlier = os.path.join(dir_test, "inlier")
            dir_outlier = os.path.join(dir_test, "outlier")

            images = []
            labels = []

            for cur_label, cur_dir in zip([0, 1], [dir_inlier, dir_outlier]):
                list_path_images = glob.glob(cur_dir + "/*.png")

                for cur_path in list_path_images:
                    cur_image = cv2.imread(cur_path, cv2.IMREAD_GRAYSCALE)
                    cur_image = cur_image.reshape(cur_image.shape + (1,))
                    images.append(cur_image)
                    labels.append(cur_label)

                logger.info("load {} images from {}".format(len(list_path_images), cur_dir))

            return np.array(images), np.array(labels)

    def _reset_indices(self):
        self._indices = range(self._num_data)
        self._idx_next = 0

        if self.is_train:
            np.random.shuffle(self._indices)

        return self._images

    def _next_indices(self, batch_size):
        if self.is_train:
            self._completed = False
            indices = []

            # epochをまたぐケース
            if self._idx_next + batch_size >= self._num_data:
                indices += self._indices[self._idx_next:self._num_data]
                batch_size -= self._num_data - self._idx_next
                self._reset_indices()
                self._completed = True

            indices += self._indices[self._idx_next:self._idx_next+batch_size]
            self._idx_next += batch_size
        else:
            if self._completed:
                raise ValueError("Epoch has been finished.")

            if self._idx_next + batch_size >= self._num_data:
                batch_size = self._num_data - self._idx_next
                self._completed = True

            indices = self._indices[self._idx_next:self._idx_next+batch_size]
            self._idx_next += batch_size

        return indices

    @property
    def completed(self):
        """
        In test mode:
          すべてのデータを利用済みかどうか
        In train mode:
          直前のnext_batchでepochをまたいだかどうか
        :return: 
        """
        return self._completed

    def preprocess(self, images):
        """
        preprocess images (commonly train mode and test mode)
        :param np.ndarray images: uint8 [NHWC]
        :return: 
        """
        images = images.astype(dtype=np.float32)

        return images / 255

    def depreprocess(self, images):
        """
        reverse preprocessed images to original images (commonly train mode and test mode)
        :param np.ndarray images: float32 [NHWC]
        :return: 
        """
        images = np.array(images)
        images = images * 255
        images = images.astype(dtype=np.uint8)
        return images

    def next_batch(self):
        """
        :rtype: (np.ndarray, np.ndarray)
        :return: 
          (images, labels)
          images: preprocessed batch images (float32 NCHW)
          labels: batch labels (uint8 [N])
        """
        indices = self._next_indices(self.batch_size)

        images = self._images[indices]
        labels = self._labels[indices]

        images, labels = np.stack(images), np.stack(labels)
        images = self.preprocess(images)

        return images, labels

    def dummy_inputs(self):
        if self.is_train:
            images = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, 28, 28, 1])
            labels = tf.placeholder(dtype=tf.uint8, shape=[self.batch_size])
        else:
            images = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
            labels = tf.placeholder(dtype=tf.uint8, shape=[None])

        return images, labels


if __name__ == "__main__":
    mnist = MNIST(is_train=True)
    mnist = MNIST(is_train=False)
