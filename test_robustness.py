import argparse

import numpy as np
import tensorflow as tf
import os

from modeldef import CylceGanModelDef
from utils import logger, makedirs
from model import CycleGAN
from data_loader import get_data


class FastSaver(tf.train.Saver):
    def save(self, sess, save_path, global_step=None, latest_filename=None,
             meta_graph_suffix="meta", write_meta_graph=True):
        super(FastSaver, self).save(sess, save_path, global_step, latest_filename,
                                    meta_graph_suffix, False)


class NoiseRobustness:
    def __init__(self, model_dir, model_def: CylceGanModelDef):
        self.model_dir = model_dir
        self.model_def = model_def
        self.session = None

    def reconstruction_loss_a(self, images, noise_levels):
        a = self.model_def.image_tensors.a
        ab = self.model_def.image_tensors.ab
        la = self.model_def.loss_tensors.cycle_aba

        return self.reconstruction_loss(images, noise_levels, a, ab, la)

    def reconstruction_loss_b(self, images, noise_levels):
        b = self.model_def.image_tensors.b
        ba = self.model_def.image_tensors.ba
        lb = self.model_def.loss_tensors.cycle_bab

        return self.reconstruction_loss(images, noise_levels, b, ba, lb)

    def reconstruction_loss(self, images, noise_levels, source_image_ph, target_image_t, reconstruction_loss_t):
        session = self.session
        base_loss = np.zeros_like(images)
        noise_loss = np.zeros(shape=(len(images), len(noise_levels)))

        for i, data in enumerate(images):
            fetches = [target_image_t, reconstruction_loss_t]
            source_image_v = np.expand_dims(data, axis=0)

            image_ab, loss = session.run(fetches, feed_dict={source_image_ph: source_image_v})
            noise = np.random.randn(*image_ab.shape)

            base_loss[i] = loss

            for j, noise_level in enumerate(noise_levels):
                perturbed = image_ab + noise * noise_level

                loss = session.run(reconstruction_loss_t, feed_dict={target_image_t: perturbed})
                noise_loss[i, j] = loss

        return base_loss, noise_loss


def main():
    parser = argparse.ArgumentParser(description="Run commands")
    parser.add_argument('-t', '--train', default=True, type=bool,
                        help="Training mode")
    parser.add_argument('--task', type=str, default='apple2orange',
                        help='Task name')
    parser.add_argument('--cycle_loss_coeff', type=float, default=10,
                        help='Cycle Consistency Loss coefficient')
    parser.add_argument('--perturbation_loss_coeff', type=float, default=0.5,
                        help='Perturbation Invariance Loss coefficient')
    parser.add_argument('--instance_normalization', default=True, type=bool,
                        help="Use instance norm instead of batch norm")
    parser.add_argument('--log_step', default=100, type=int,
                        help="Tensorboard log frequency")
    parser.add_argument('--batch_size', default=1, type=int,
                        help="Batch size")
    parser.add_argument('--image_size', default=128, type=int,
                        help="Image size")
    parser.add_argument('--load_model', default='',
                        help='Model path to load (e.g., train_2017-07-07_01-23-45)')

    args, unparsed = parser.parse_known_args()

    logger.info('Read data:')
    train_A, train_B, test_A, test_B = get_data(args.task, args.image_size)

    logger.info('Build graph:')
    model = CycleGAN(args)

    variables_to_save = tf.global_variables()
    init_op = tf.variables_initializer(variables_to_save)
    init_all_op = tf.global_variables_initializer()
    saver = FastSaver(variables_to_save)

    logger.info('Trainable vars:')
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                 tf.get_variable_scope().name)
    for v in var_list:
        logger.info('  %s %s', v.name, v.get_shape())

    if args.load_model != '':
        model_name = args.load_model
    else:
        assert False
    logdir = './logs'
    logdir = os.path.join(logdir, model_name)
    logger.info('Events directory: %s', logdir)
    summary_writer = tf.summary.FileWriter(logdir)

    def init_fn(sess):
        logger.info('Initializing all parameters.')
        sess.run(init_all_op)

    sv = tf.train.Supervisor(is_chief=True,
                             logdir=logdir,
                             saver=saver,
                             summary_op=None,
                             init_op=init_op,
                             init_fn=init_fn,
                             summary_writer=summary_writer,
                             ready_op=tf.report_uninitialized_variables(variables_to_save),
                             global_step=model.global_step,
                             save_model_secs=300,
                             save_summaries_secs=30)

    logger.info("Starting testing session.")
    with sv.managed_session() as sess:
        experiment = NoiseRobustness("", model.get_modeldef())
        experiment.session = sess
        print(experiment.reconstruction_loss_a(test_A, [0.01, 0.25, 0.5]))


if __name__ == "__main__":
    main()
