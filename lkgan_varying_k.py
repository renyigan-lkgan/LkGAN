import os
import time
import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, \
    LeakyReLU, Conv2DTranspose, Conv2D, Dropout, Flatten, Reshape
import utils
import scipy as sp
import numpy as np

version, trial_number, seed_num = input("version, trial number, seed: ").split()

if int(version) == 1:
    alpha = 0.6
    beta = 0.4
elif int(version) == 2:
    alpha = 1
    beta = 0
else:
    alpha = 0
    beta = 1
if int(version) == 3:
    gamma = 0
else:
    gamma = (alpha + beta)/2.0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(int(seed_num))
tf.random.set_random_seed(int(seed_num))


class GAN(object):
    def __init__(self, version, alpha, beta, gamma, trial_num):
        self.batch_size = 100
        self.n_classes = 10
        self.buffer_size = 1000
        self.training = True

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.k = round(1.0, 1)
        self.k_placeholder = tf.placeholder(tf.float32, shape=[], name='k_placeholder')
        self.trial_num = trial_num
        self.noise_dim = 28 * 28
        self.dropout_constant = 0.6
        self.epsilon = 1e-8  # To ensure the log doesn't blow up to -infinity
        self.predictions = []
        self.fid_scores = []
        self._make_directory('data/')
        self._make_directory('data/sim/')
        self._make_directory('data/sim/v_' + str(self.version))
        self._make_directory('data/sim/v_' + str(self.version))
        self.save_path = 'data/sim/v_' + str(self.version) + '/trial' + str(self.trial_num)
        self._make_directory(self.save_path)

    @staticmethod
    def _make_directory(PATH):
        if not os.path.exists(PATH):
            os.mkdir(PATH)

    def get_data(self):
        with tf.name_scope('data'):
            (train_d, _), _ = tf.keras.datasets.mnist.load_data()
            train_data = (train_d - 127.5) / 127.5
            train_data = tf.data.Dataset.from_tensor_slices(train_data)
            train_data = train_data.shuffle(60000)
            train_data = train_data.batch(self.batch_size)
            self.iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                                            train_data.output_shapes)
            img = self.iterator.get_next()
            self.img = tf.reshape(img, shape=[-1, 28, 28, 1])

            self.train_init = self.iterator.make_initializer(train_data)
            
            train_images = train_d.reshape(train_d.shape[0], 28 * 28).astype('float64')
            train_images = train_images / 255.0
            self.real_mu = train_images.mean(axis=0)
            train_images = np.transpose(train_images)
            self.real_sigma = np.cov(train_images)


    def build_generator(self):
        with tf.name_scope('generator') as scope:
            model = Sequential(name=scope)
            model.add(Dense(7 * 7 * 256, use_bias=False, kernel_initializer=
            RandomNormal(mean=0.0, stddev=0.01), input_shape=(self.noise_dim,)))
            model.add(BatchNormalization())
            model.add(LeakyReLU())

            model.add(Reshape((7, 7, 256)))
            assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

            model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=
            RandomNormal(mean=0.0, stddev=0.01)))
            assert model.output_shape == (None, 7, 7, 128)
            model.add(BatchNormalization())
            model.add(LeakyReLU())

            model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=
            RandomNormal(mean=0.0, stddev=0.01)))
            assert model.output_shape == (None, 14, 14, 64)
            model.add(BatchNormalization())
            model.add(LeakyReLU())

            model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', activation='tanh', use_bias=False,
                                      kernel_initializer=RandomNormal(mean=0.0, stddev=0.01)))
            assert model.output_shape == (None, 28, 28, 1)

            return model

    def build_discriminator(self):
        with tf.name_scope('discriminator') as scope:
            model = Sequential(name=scope)
            model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=
            RandomNormal(mean=0.0, stddev=0.01)))
            model.add(LeakyReLU())
            model.add(Dropout(0.3))

            model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_initializer=
            RandomNormal(mean=0.0, stddev=0.01)))
            model.add(LeakyReLU())
            model.add(Dropout(0.3))

            model.add(Flatten())
            model.add(Dense(1, activation='sigmoid', kernel_initializer=
            RandomNormal(mean=0.0, stddev=0.01)))

            return model

    def dis_loss(self):
        a = tf.math.reduce_mean(tf.math.pow(self.real_output - self.beta, 2.0 * tf.ones_like(self.real_output)))
        b = tf.math.reduce_mean(tf.math.pow(self.fake_output - self.alpha, 2.0 * tf.ones_like(self.real_output)))
        return 1 / 2.0 * (a + b)

    def gen_loss(self):
        return tf.math.reduce_mean(tf.math.pow(tf.math.abs(self.fake_output - self.gamma),
                                               self.k_placeholder * tf.ones_like(self.real_output)))

    
    def optimize(self):
        self.gen_opt = tf.train.AdamOptimizer(2e-4, beta1=0.5, name="generator_optimizer")
        self.gen_opt_minimize = self.gen_opt.minimize(self.gen_loss_value, var_list=self.generator.trainable_variables)
        self.dis_opt = tf.train.AdamOptimizer(2e-4, beta1=0.5, name="discriminator_optimizer")
        self.dis_opt_minimize = self.dis_opt.minimize(self.dis_loss_value,
                                                      var_list=self.discriminator.trainable_variables)

    def build(self):
        self.get_data()
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.fake_output_images = self.generator(tf.random.normal([self.batch_size, self.noise_dim]))
        self.fake_output = self.discriminator(self.fake_output_images) 
        self.real_output = self.discriminator(tf.cast(self.img, dtype=tf.float32))  
        self.gen_loss_value = self.gen_loss()
        self.dis_loss_value = self.dis_loss()
        self.optimize()

    def train_one_epoch(self, sess, init, epoch):
        start_time = time.time()
        sess.run(init)
        self.training = True
        total_loss_gen = 0
        total_loss_dis = 0
        n_batches = 0
        try:
            while True:
                _, disLoss = sess.run([self.dis_opt_minimize, self.dis_loss_value])
                _, genLoss = sess.run([self.gen_opt_minimize, self.gen_loss_value],
                                        feed_dict={self.k_placeholder: self.k})
                total_loss_gen += genLoss
                total_loss_dis += disLoss
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        self.fid_scores.append(self.calculate_fid(sess))
        print('Average generator loss at epoch {0}: {1}'.format(epoch, total_loss_gen / n_batches))
        print('Average discriminator loss at epoch {0}: {1}'.format(epoch, total_loss_dis / n_batches))
        print('Took: {0} seconds'.format(time.time() - start_time))

    def calculate_fid(self, sess):
        fake_images = self.generator(tf.random.normal([60000, self.noise_dim]))
        fake_images = sess.run(fake_images)
        fake_images = fake_images.reshape(60000, 28*28)
        fake_images = (fake_images * 127.5 + 127.5) / 255.0
        fake_mu = fake_images.mean(axis=0)
        fake_sigma = np.cov(np.transpose(fake_images))
        covSqrt = sp.linalg.sqrtm(np.matmul(fake_sigma, self.real_sigma))
        if np.iscomplexobj(covSqrt):
            covSqrt = covSqrt.real
        fidScore = np.linalg.norm(self.real_mu - fake_mu) + np.trace(self.real_sigma + fake_sigma - 2 * covSqrt)
        return fidScore

    def save_generated_images(self, epoch, sess):
        temp = self.generator(tf.random.normal([self.buffer_size, self.noise_dim]))
        temp = sess.run(temp)
        if len(self.predictions) > 0:
            self.predictions.pop(0)
        self.predictions.append(temp)
        np.save(self.save_path + '/predictions' + str(epoch), self.predictions)

    def train(self, n_epochs):
        self._make_directory('checkpoints')
        self._make_directory('checkpoints/sim')
        self._make_directory('checkpoints/sim/v_' + str(self.version))
        self.cpt_PATH = 'checkpoints/sim/v_' + str(self.version) 
        if self.trial_num == 1:
            self._make_directory(self.cpt_PATH)

        go_up = True
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(self.train_init)
            checkpoint = tf.train.Saver(
                {'generator_optimizer': self.gen_opt, 'discriminator_optimizer': self.dis_opt,
                 'generator': self.generator, 'discriminator': self.discriminator, 'iterator': self.iterator},
                max_to_keep=3)
            for epoch in range(n_epochs):
                if self.trial_num == 1:
                    if epoch % 10 == 0:
                        save_path = checkpoint.save(sess, self.cpt_PATH, global_step=epoch)
                        print("Saved checkpoint for step {}: {}".format(int(epoch), save_path))
                if epoch % 10 == 0:
                    self.save_generated_images(epoch, sess)
                print("K value: " + str(self.k))
                if go_up:
                    self.k = round(self.k + 0.1, 1)
                    if self.k == 3.0:
                        go_up = False
                else:
                    self.k = round(self.k - 0.1, 1)
                    if self.k == 1.0:
                        go_up = True
                self.train_one_epoch(sess, self.train_init, epoch)
                np.save(self.save_path + '/scores', self.fid_scores)

model = GAN(int(version), int(alpha), int(beta), int(gamma), int(trial_number))
model.build()
model.train(n_epochs=100)
