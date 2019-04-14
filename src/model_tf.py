from functools import reduce
from operator import mul

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data")


class GAN:
    def __init__(self, noise_shape, image_shape, learning_rate):
        self.noise_shape = noise_shape
        self.image_shape = image_shape

        self.image_size = reduce(mul, image_shape)

        self.learning_rate = learning_rate

        # ----------------------------------------------------------------------
        tf.reset_default_graph()
        self.noise = tf.placeholder(tf.float32,
                                    shape=[None, *self.noise_shape],
                                    name='noise')
        self.image = tf.placeholder(tf.float32,
                                    shape=[None, self.image_size],
                                    name='image')

        (self.gen_loss, self.dis_loss), (self.gen_trainer, self.dis_trainer), self.dis_accuracy = self._trainer()

        self.init_g = tf.global_variables_initializer()
        self.init_l = tf.local_variables_initializer()
        self.sess = tf.Session()

    def generator(self, noise, reuse=None):
        with tf.variable_scope('gen', reuse=reuse):
            hidden1 = tf.layers.dense(inputs=noise, units=128,
                                      activation=tf.nn.sigmoid,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
            hidden2 = tf.layers.dense(inputs=hidden1, units=128,
                                      activation=tf.nn.leaky_relu,)
                                      # kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))
            output = tf.layers.dense(inputs=hidden2, units=self.image_size,
                                     activation=tf.nn.tanh)

            return output

    def discriminator(self, image, reuse=None):
        with tf.variable_scope('dis', reuse=reuse):
            hidden1 = tf.layers.dense(inputs=image, units=128,
                                      activation=tf.nn.leaky_relu,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
            hidden2 = tf.layers.dense(inputs=hidden1, units=128,
                                      activation=tf.nn.leaky_relu,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
            logits = tf.layers.dense(hidden2, units=1)
            output = tf.sigmoid(logits)

            return output, logits

    @classmethod
    def loss_func(cls, logits_in, labels_in):
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_in,
                                                    labels=labels_in))

    def gen_noise(self, batch_size):
        return np.random.uniform(-1, 1, size=(batch_size, self.noise_shape[0]))

    def _trainer(self):
        gen = self.generator(self.noise)
        dis_output_real, dis_logits_real = self.discriminator(self.image)
        dis_output_fake, dis_logits_fake = self.discriminator(gen, reuse=True)

        dis_accuracy = tf.metrics.accuracy(labels=tf.concat((tf.ones_like(dis_logits_real), tf.zeros_like(dis_logits_fake)), axis=0),
                                           predictions=tf.round(tf.concat((dis_output_real, dis_output_fake), axis=0)))

        dis_real_loss = self.loss_func(dis_logits_real,
                                       tf.ones_like(dis_logits_real))  # Smoothing for generalization
        dis_fake_loss = self.loss_func(dis_logits_fake,
                                       tf.zeros_like(dis_logits_fake))

        dis_loss = dis_real_loss + dis_fake_loss
        dis_loss += tf.losses.get_regularization_loss(scope='dis')

        gen_loss = self.loss_func(dis_logits_fake, tf.ones_like(dis_logits_fake))
        gen_loss += tf.losses.get_regularization_loss(scope='gen')

        train_vars = tf.trainable_variables()
        d_vars = [var for var in train_vars if 'dis' in var.name]
        g_vars = [var for var in train_vars if 'gen' in var.name]

        dis_trainer = tf.train.AdamOptimizer(self.learning_rate) \
                              .minimize(dis_loss, var_list=d_vars)
        gen_trainer = tf.train.AdamOptimizer(self.learning_rate) \
                              .minimize(gen_loss, var_list=g_vars)
        return (gen_loss, dis_loss), (gen_trainer, dis_trainer), dis_accuracy

    def train(self, epochs, batch_size, init=True):
        if init:
            self.sess.run([self.init_g, self.init_l])

        for epoch in range(epochs):
            num_batches = mnist.train.num_examples // batch_size
            for i in range(num_batches):
                batch = mnist.train.next_batch(batch_size)
                batch_images = batch[0].reshape((batch_size, self.image_size))
                batch_images = batch_images * 2 - 1
                batch_noise = self.gen_noise(batch_size)
                d_trainer, d_loss, d_acc = self.sess.run([self.dis_trainer, self.dis_loss, self.dis_accuracy],
                                                          feed_dict={self.image: batch_images,
                                                                     self.noise: batch_noise})
                g_trainer, g_loss = self.sess.run([self.gen_trainer, self.gen_loss],
                                                   feed_dict={self.noise: batch_noise})

            print(f"[{epoch:04d}]: [G_LOSS]={g_loss:0.3f}, [D_LOSS]={d_loss:0.3f}, [D_ACC]={d_acc[0]:0.3f}, {d_acc[1]:0.3f}")

            if epoch % 10 == 0:
                self.sample_images(epoch)

    def predict_gen(self, sample_noise):
        return self.sess.run(self.generator(self.noise, reuse=True),
                             feed_dict={self.noise: sample_noise})\
                        .reshape([-1, *self.image_shape])

    def predict_dis(self, sample_image):
        return self.sess.run(self.discriminator(self.image, reuse=True),
                             feed_dict={self.image: sample_image})

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = self.gen_noise(r * c)
        gen_imgs = self.predict_gen(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, ...].squeeze(), cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    noise_shape = (500,)
    image_shape = (28, 28)
    epochs = 100
    batch_size = 64
    learning_rate = 0.0001
    g = GAN(noise_shape, image_shape, learning_rate)
    g.train(epochs, batch_size)
    g.sample_images(-1)
