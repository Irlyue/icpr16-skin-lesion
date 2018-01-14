import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from collections import OrderedDict


def model_placeholders(batch_size):
    images_local = tf.placeholder(tf.uint8, shape=(batch_size, 31, 31, 3), name='images_local')
    images_global = tf.placeholder(tf.uint8, shape=(batch_size, 201, 201, 3), name='images_global')
    labels = tf.placeholder(tf.int32, shape=(batch_size, 31, 31), name='labels')
    return images_local, images_global, labels


class PatchModel:
    def __init__(self, images_local, images_global):
        with tf.name_scope('patch_model'):
            local_pool2 = self._one_wind(images_local, 'local')
            global_pool2 = self._one_wind(images_global, 'global')
            pool2 = tf.concat([local_pool2, global_pool2], axis=3, name='pool2')
            pool2_flatten = slim.flatten(pool2)
            fc3 = slim.fully_connected(pool2_flatten,
                                       num_outputs=500)
            fc4 = slim.fully_connected(fc3,
                                       num_outputs=2,
                                       activation_fn=None)
            prob = tf.nn.sigmoid(fc4, name='probability')

        endpoints = OrderedDict()
        endpoints['images_local'] = images_local
        endpoints['images_global'] = images_global
        endpoints['pool2'] = pool2
        endpoints['fc3'] = fc3
        endpoints['fc4'] = fc4
        endpoints['prob'] = prob
        self.endpoints = endpoints

    def _one_wind(self, images, type_):
        with tf.variable_scope(type_):
            images = tf.cast(images, tf.float32)
            if type_ == 'global':
                images = tf.image.resize_images(images, size=(31, 31))
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
        return pool2

    def __repr__(self):
        myself = ''
        myself += '\n'.join('{:<2} {:<15} {!r}{!r}'.format(i, key, op.dtype, op.shape.as_list())
                            for i, (key, op) in enumerate(self.endpoints.items()))
        return myself


if __name__ == '__main__':
    a, b, c = model_placeholders(32)
    model = PatchModel(a, b)
    print(model)
