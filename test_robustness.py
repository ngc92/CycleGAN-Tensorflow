import argparse

import numpy as np
import tensorflow as tf
import os

from modeldef import CycleGanModelDef
from utils import logger, makedirs
from model import CycleGAN
from data_loader import get_data


class NoiseRobustness:
    def __init__(self, model_def: CycleGanModelDef, session):
        self.model_def = model_def
        self.session = session

    def reconstruction_loss_a(self, images, noise_levels):
        a = self.model_def.image_tensors.a
        ab = self.model_def.image_tensors.ab
        aba = self.model_def.image_tensors.aba
        la = self.model_def.loss_tensors.cycle_aba

        return self.reconstruction_loss(images, noise_levels, a, ab, aba, la)

    def reconstruction_loss_b(self, images, noise_levels):
        b = self.model_def.image_tensors.b
        ba = self.model_def.image_tensors.ba
        bab = self.model_def.image_tensors.bab
        lb = self.model_def.loss_tensors.cycle_bab

        return self.reconstruction_loss(images, noise_levels, b, ba, bab, lb)

    def reconstruction_loss(self, images, noise_levels, source_image_ph, target_image_t, 
                            reconstruction_image_t, reconstruction_loss_t):
        session = self.session
        base_loss = np.zeros_like(images)
        noise_loss = np.zeros(shape=(len(images), len(noise_levels)))
        image_difference = np.zeros(shape=(len(images), len(noise_levels)))

        for i, data in enumerate(images):
            fetches = [target_image_t, reconstruction_loss_t, reconstruction_image_t]
            source_image_v = np.expand_dims(data, axis=0)

            image_ab, loss, reconstruction_image_v = \
                session.run(fetches, feed_dict={source_image_ph: source_image_v, "is_train:0": False})
            noise = np.random.randn(*image_ab.shape)

            base_loss[i] = loss

            for j, noise_level in enumerate(noise_levels):
                perturbed = image_ab + noise * noise_level

                loss, reconstructed_with_noise = \
                    session.run((reconstruction_loss_t, reconstruction_image_t),
                                 feed_dict={source_image_ph: source_image_v, target_image_t: perturbed,
                                            "is_train:0": False})
                noise_loss[i, j] = loss
                image_difference[i, j] = np.mean(np.abs(reconstruction_image_v - reconstructed_with_noise))

        return base_loss, noise_loss, image_difference


def main():
    parser = argparse.ArgumentParser(description="Run commands")
    parser.add_argument('--task', type=str, default='apple2orange',
                        help='Task name')
    parser.add_argument('--image_size', default=128, type=int,
                        help="Image size")
    parser.add_argument('--model', default='',
                        help='Model path to load (e.g., train_2017-07-07_01-23-45)')

    args, unparsed = parser.parse_known_args()

    logger.info('Read data:')
    train_A, train_B, test_A, test_B = get_data(args.task, args.image_size)

    logdir = './logs'
    logdir = os.path.join(logdir, args.model)

    with tf.Session() as session:
        latest_checkpoint = tf.train.latest_checkpoint(logdir)  # type: str
        saver = tf.train.import_meta_graph(latest_checkpoint + ".meta")
        saver.restore(session, logdir)

        experiment = NoiseRobustness(CycleGanModelDef.from_json("cyclegan_model.json"), session)
        noise_levels = np.linspace(0.0, 0.05, 20)
        base, noise, idiff = experiment.reconstruction_loss_a(test_A, noise_levels)

        print(np.mean(base))
        result_data = [noise_levels, np.mean(noise, axis=0), np.mean(idiff, axis=0)]
        np.savetxt(os.path.join(logdir, "robustness.txt"), result_data)


if __name__ == "__main__":
    main()
