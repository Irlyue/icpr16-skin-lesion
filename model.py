import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from collections import OrderedDict


def model_placeholders(batch_size, input_size):
    height, width = input_size
    images = tf.placeholder(tf.uint8, shape=(batch_size, height, width, 3), name='images')
    labels = tf.placeholder(tf.int32, shape=(batch_size, height, width), name='labels')
    return images, labels


class PatchModel:
    def __init__(self, images):
        with tf.name_scope('patch_model'):
            images = tf.cast(images, tf.float32)
            conv1 = slim.conv2d(images,
                                num_outputs=60,
                                kernel_size=(6, 6),
                                padding='VALID',
                                scope='conv1')
            pool1 = slim.max_pool2d(conv1,
                                    kernel_size=(2, 2),
                                    stride=2,
                                    scope='pool1')
            conv2 = slim.conv2d(pool1,
                                num_outputs=60,
                                kernel_size=(5, 5),
                                padding='VALID',
                                scope='conv2')
            pool2 = slim.max_pool2d(conv2,
                                    kernel_size=(3, 3),
                                    stride=3,
                                    scope='pool2')
            pool2_flatten = slim.flatten(pool2)
            fc3 = slim.fully_connected(pool2_flatten,
                                       num_outputs=500)
            fc4 = slim.fully_connected(fc3,
                                       num_outputs=2,
                                       activation_fn=None)
            prob = tf.nn.sigmoid(fc4, name='probability')

        endpoints = OrderedDict()
        endpoints['images'] = images
        endpoints['conv1'] = conv1
        endpoints['pool1'] = pool1
        endpoints['conv2'] = conv2
        endpoints['pool2'] = pool2
        endpoints['fc3'] = fc3
        endpoints['fc4'] = fc4
        endpoints['prob'] = prob
        self.endpoints = endpoints

    def __repr__(self):
        myself = ''
        myself += '\n'.join('{:<2} {:<10} {!r}{!r}'.format(i, key, op.dtype, op.shape.as_list())
                            for i, (key, op) in enumerate(self.endpoints.items()))
        return myself


if __name__ == '__main__':
    images, _ = model_placeholders(32, input_size=(31, 31))
    model = PatchModel(images)
    print(model)
