# -*- coding:utf-8 -*-
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# The original source code is distributed at
#    https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/learn/python/learn/datasets
# This is modified by laket72@gmail.com
# ==============================================================================


"""Functions for downloading and reading MNIST data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import gzip

import numpy as np
from six.moves import xrange    # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base

import config

DIR_MNIST = os.path.join(config.DATA_ROOT, "mnist")

# CVDF mirror of http://yann.lecun.com/exdb/mnist/
SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'

def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f):
    """Extract the images into a 4D uint8 np array [index, y, x, depth].

    Args:
        f: A file object that can be passed into a gzip reader.

    Returns:
        data: A 4D uint8 np array [index, y, x, depth].

    Raises:
        ValueError: If the bytestream does not start with 2051.

    """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                                             (magic, f.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(f, one_hot=False, num_classes=10):
    """Extract the labels into a 1D uint8 np array [index].

    Args:
        f: A file object that can be passed into a gzip reader.
        one_hot: Does one hot encoding for the result.
        num_classes: Number of classes for the one hot encoding.

    Returns:
        labels: a 1D uint8 np array.

    Raises:
        ValueError: If the bystream doesn't start with 2049.
    """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                                             (magic, f.name))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        if one_hot:
            return dense_to_one_hot(labels, num_classes)
        return labels

def download_mnist(train_dir):
    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

    local_file = base.maybe_download(TRAIN_IMAGES, train_dir,
                                     SOURCE_URL + TRAIN_IMAGES)
    with open(local_file, 'rb') as f:
        train_images = extract_images(f)

    local_file = base.maybe_download(TRAIN_LABELS, train_dir,
                                     SOURCE_URL + TRAIN_LABELS)
    with open(local_file, 'rb') as f:
        train_labels = extract_labels(f, one_hot=False)

    local_file = base.maybe_download(TEST_IMAGES, train_dir,
                                     SOURCE_URL + TEST_IMAGES)
    with open(local_file, 'rb') as f:
        test_images = extract_images(f)

    local_file = base.maybe_download(TEST_LABELS, train_dir,
                                     SOURCE_URL + TEST_LABELS)
    with open(local_file, 'rb') as f:
        test_labels = extract_labels(f, one_hot=False)

    return train_images, train_labels, test_images, test_labels

if __name__ == "__main__":
    """
    Generate Dataset
    """
    import os
    import shutil
    import argparse
    import logging

    import cv2
    import config

    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--inlier", dest="inlier", type=int, help="inlier number", required=True)
    parser.add_argument("--rho", dest="outlier_ratio", default=0.3, type=float, help="outlier_ratio [0.3]")
    parser.add_argument("--test_ratio", dest="test_ratio", default=0.5, type=float, help="test data size relative to train data size [0.5]")
    parser.add_argument("-o", dest="dir_out", default=DIR_MNIST, help="output directory")
    args = parser.parse_args()

    inlier, outlier_ratio, test_ratio, dir_out = args.inlier, args.outlier_ratio, args.test_ratio, args.dir_out
    dir_train = os.path.join(dir_out, "train")
    dir_test = os.path.join(dir_out, "test")

    train_images, train_labels, test_images, test_labels = download_mnist(DIR_MNIST)
    images = np.concatenate([train_images, test_images])
    labels = np.concatenate([train_labels, test_labels])

    indices_inlier = np.where(labels == inlier)[0]
    indices_outlier = np.where(labels != inlier)[0]

    num_inlier = len(indices_inlier)
    num_train_inlier = int(num_inlier / (1.0+test_ratio))
    num_test_inlier = num_inlier - num_train_inlier
    # MEMO: outlier_ratioの定義が論文と違う可能性あり (inlierに対する比率ではなく、合計サイズにおける比率かも)
    num_test_outlier = int(num_test_inlier * outlier_ratio)

    np.random.shuffle(indices_inlier)
    np.random.shuffle(indices_outlier)
    indices_train_inlier = indices_inlier[:num_train_inlier]
    indices_test_inlier = indices_inlier[num_train_inlier:]
    indices_test_outlier = indices_outlier[:num_test_outlier]

    logger.info("total inlier({}) size is {}".format(inlier, num_inlier))
    logger.info("train inlier {} test_inlier {} test_outlier {}".format(
            len(indices_train_inlier),
            len(indices_test_inlier),
            len(indices_test_outlier)))


    if os.path.exists(dir_train):
        shutil.rmtree(dir_train)
    os.makedirs(dir_train)

    if os.path.exists(dir_test):
        shutil.rmtree(dir_test)
    os.makedirs(dir_test)

    for cur_count, cur_idx in enumerate(indices_train_inlier):
        cur_path = os.path.join(dir_train, "{:04}.png".format(cur_count))
        cv2.imwrite(cur_path, images[cur_idx])

    dir_test_inlier = os.path.join(dir_test, "inlier")
    os.makedirs(dir_test_inlier)
    for cur_count, cur_idx in enumerate(indices_test_inlier):
        cur_path = os.path.join(dir_test_inlier, "{:04}.png".format(cur_count))
        cv2.imwrite(cur_path, images[cur_idx])

    dir_test_outlier = os.path.join(dir_test, "outlier")
    os.makedirs(dir_test_outlier)
    for cur_count, cur_idx in enumerate(indices_test_outlier):
        cur_path = os.path.join(dir_test_outlier, "{:04}.png".format(cur_count))
        cv2.imwrite(cur_path, images[cur_idx])





