from typing import Tuple
import src.utils_sprite as utils_sprite
from functools import reduce

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow._api.v1.keras as ker
import numpy as np
from sklearn.utils import shuffle

input_shape = (100,)
output_shape = (32, 32, 3)
output_size = 32 * 32 * 3
batch_size = 32


class GAN:
    def __init__(self,
                 noise_shape: Tuple,
                 img_shape: Tuple,
                 learning_rate: float):
        self.noise_shape = noise_shape
        self.img_shape = img_shape

        self.optimizer = ker.optimizers.Adam(learning_rate)

        # Build and compile the discriminator
        self.dis = self.discriminator()
        self.dis.compile(loss='categorical_crossentropy',
                         optimizer=self.optimizer,
                         metrics=['accuracy'])

        # Build the generator
        self.gen = self.generator()

        generator = ker.models.Model(inputs=self.gen.inputs,
                                     outputs=self.gen.outputs)
        discriminator_frozen = ker.models.Model(inputs=self.dis.inputs,
                                    outputs=self.dis.outputs)
        discriminator_frozen.trainable = False
        # The generator takes noise as input and generates imgs

        self.gan = ker.models.Sequential([generator, discriminator_frozen])
        self.gan.compile(loss='categorical_crossentropy',
                         optimizer=self.optimizer)

    def generator(self):
        img_size = reduce(lambda x, y: x * y, self.img_shape)
        with tf.name_scope('generator'):
            model = ker.Sequential(
                    [ker.layers.Dense(input_shape=self.noise_shape,
                                      units=128,
                                      activation='sigmoid'),
                     ker.layers.Dense(units=128,
                                      kernel_regularizer=ker.regularizers.l1_l2(),
                                      activation='sigmoid'),
                     ker.layers.Dense(units=128,
                                      activation='sigmoid'),
                     ker.layers.Dense(units=img_size,
                                      kernel_regularizer=ker.regularizers.l1_l2(),
                                      activation='sigmoid'),
                     ker.layers.Reshape(target_shape=self.img_shape)
                     ])
            # model.compile(optimizer=ker.optimizers.Adam(lr=learning_rate),
            #               loss='msle')
            model.summary()
        return model

    def discriminator(self):
        with tf.name_scope('discriminator'):
            model = ker.Sequential(
                    [ker.layers.Flatten(input_shape=self.img_shape),
                     ker.layers.Dense(units=32,
                                      activation='sigmoid'),
                     ker.layers.Dense(units=32,
                                      kernel_regularizer=ker.regularizers.l1_l2(),
                                      activation='sigmoid'),
                     ker.layers.Dense(units=32,
                                      kernel_regularizer=ker.regularizers.l1_l2(),
                                      activation='sigmoid'),
                     ker.layers.Dense(units=2,
                                      activation='softmax'),
                     ])
            # model.compile(optimizer=ker.optimizers.Adam(ls=learning_rate),
            #               loss='binary_crossentropy')
            model.summary()

        return model

    def train(self,
              batch_size: int,
              epoch: int):
        sprite_l = np.array(utils_sprite.load_sprites_from_bin(
                utils_sprite.BIN_LOW_RES_FILE)) / 127.5 - 1.

        # (sprite_l, _), (_, _) = ker.datasets.mnist.load_data()

        real_target = np.array([1, 0] * batch_size).reshape(batch_size, 2)
        fake_target = np.array([0, 1] * batch_size).reshape(batch_size, 2)
        all_target = np.concatenate((real_target, fake_target))

        for e in range(epoch):
            noise = self.gen_noise(batch_size)
            fake_images = self.gen.predict(noise)

            real_images = self.batch_img(sprite_l, batch_size)

            all_images = np.concatenate((real_images, fake_images))

            dis_loss = self.dis.train_on_batch(all_images, all_target)


            # train generator

            gen_loss = self.gan.train_on_batch(noise, real_target)

            if e % 100 == 0:
                print(f"{str(e).rjust(6, '0')} [D loss: {dis_loss[0]:.2f}, acc: {100*dis_loss[1]:.2f}] [G loss: {gen_loss:.2f}]")

            if e % 1000 == 0:
                self.sample_images(e)
        pass

    def gen_noise(self, batch_size: int):
        return np.random.uniform(-1, 1, (batch_size, self.noise_shape[0]))

    def batch_img(self, img_l, batch):
        return img_l[np.random.randint(0, len(img_l) - 1, batch), ...] \
                    .reshape([batch, *self.img_shape])

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = self.gen_noise(r * c)
        gen_imgs = self.gen.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, ...].squeeze())
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()


def main(*args, **kwargs) -> GAN:
    gan = GAN(input_shape, output_shape, 0.00001)
    gan.train(batch_size=64, epoch=10000)

    return gan
    pass


if __name__ == '__main__':
    g = main()
