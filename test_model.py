"""
Script usage:
    python test_model.py --n_gpus=0
"""
import model
import my_utils
import argparse
import numpy as np
import tensorflow as tf

logger = my_utils.get_default_logger()
parser = argparse.ArgumentParser()
parser.add_argument('--n_gpus', type=int, default=0)
parser.add_argument('--n_patches', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)


class RunningModel:
    def __init__(self, batch_size, input_size, gpu_count=0):
        with tf.Graph().as_default() as g:
            image_ph, label_ph = model.model_placeholders(batch_size, input_size)
            self.batch_size = batch_size
            self.input_size = input_size
            self.image_ph = image_ph
            self.net = model.PatchModel(image_ph)
            self.sess = tf.Session(graph=g, config=tf.ConfigProto(device_count={'GPU': gpu_count}))
            self.sess.run(tf.global_variables_initializer())

    def pre_run_n_step(self, n=10):
        logger.info('Pre-run %d steps to warm up the CPU...' % n)
        with my_utils.Timer() as timer:
            height, width = self.input_size
            for _ in range(n):
                image = np.random.randint(255, size=(self.batch_size, height, width, 3), dtype=np.uint8)
                self.inference_prob(image)
        logger.info('Done in %.4fs(%.4fsecs per run)' % (timer.eclipsed, timer.eclipsed / n))

    def run_n_step(self, n_patches=100, same_image=True):
        n_runs = n_patches // self.batch_size
        logger.info('Run %d steps for %d patches...' % (n_runs, n_patches))
        height, width = self.input_size
        image = np.random.randint(255, size=(self.batch_size, height, width, 3), dtype=np.uint8)
        with my_utils.Timer() as timer:
            for _ in range(n_runs):
                if not same_image:
                    image = np.random.randint(255, size=(self.batch_size, height, width, 3), dtype=np.uint8)
                self.inference_prob(image)
        logger.info('Done in %.4fs(%.4fsecs per run)' % (timer.eclipsed, timer.eclipsed / n_runs))

    def _build_feed_dict(self, image):
        return {self.image_ph: image}

    def inference(self, image, ops):
        if type(ops[0]) == str:
            ops = [self.net.endpoints[op] for op in ops]
        return self.sess.run(ops, feed_dict=self._build_feed_dict(image))

    def inference_prob(self, image):
        return self.inference(image, ['prob'])[0]


def test_forward_time(gpu_count=0):
    input_size = 31, 31
    mm = RunningModel(FLAGS.batch_size, input_size, gpu_count=gpu_count)
    mm.pre_run_n_step()
    mm.run_n_step(n_patches=FLAGS.n_patches)


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    test_forward_time(FLAGS.n_gpus)
