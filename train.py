import numpy as np
import sys
import os
import glob
import gc

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['agg.path.chunksize'] = 100000
matplotlib.use("Agg")

from generator import Generator
from discriminator import Discriminator
from voice_analy import Voice_Analy
from complex import to_polar, to_rect

from keras.optimizers import Adam, SGD
from keras.models import Sequential, model_from_json, Model
from keras.layers import Input

from distutils.dir_util import copy_tree

""" GPUがあれば、GPUメモリの使用量を指定 """
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.backend import tensorflow_backend

d = str(device_lib.list_local_devices())
if "GPU" in d:
    # config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))  # リアルタイムで必要な分だけ確保
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.6))  # 空きメモリの何割を使うかを指定
    # config = tf.ConfigProto(device_count={"GPU": 0})  # GPUを使わない(GPUメモリがどうしても足らないとき)
    session = tf.Session(config=config)
    tensorflow_backend.set_session(session)


class Train:
    def __init__(self):
        self.save_path = "save/"
        os.makedirs(self.save_path, exist_ok=True)

        self.data_path = "materials/"

        self.res_path = "result/"
        os.makedirs(self.res_path + "data_bA", exist_ok=True)
        os.makedirs(self.res_path + "data_aB", exist_ok=True)
        os.makedirs(self.res_path + "pic/data_bA", exist_ok=True)
        os.makedirs(self.res_path + "pic/data_aB", exist_ok=True)

        self.fft_len = 1024 // 2
        self.window = np.hamming(self.fft_len)
        self.step = self.fft_len // 4

        self.input_shape = (self.fft_len,)

        self.g_optimizer = Adam(0.0002, 0.5)
        self.d_optimizer = SGD()

        if os.path.exists(self.save_path + "log.txt"):
            """ [Z] """
            self.g_aB_z = model_from_json(open(self.save_path + "g_ab_z.json", "r").read())
            self.g_aB_z.load_weights(self.save_path + "g_ab_w_z.h5")

            self.d_B_z = model_from_json(open(self.save_path + "d_b_z.json", "r").read())
            self.d_B_z.load_weights(self.save_path + "d_b_w_z.h5")
            self.d_B_z.compile(loss="mse",
                               optimizer=self.d_optimizer,
                               metrics=["accuracy"])

            self.g_bA_z = model_from_json(open(self.save_path + "g_ba_z.json", "r").read())
            self.g_bA_z.load_weights(self.save_path + "g_ba_w_z.h5")

            self.d_A_z = model_from_json(open(self.save_path + "d_a_z.json", "r").read())
            self.d_A_z.load_weights(self.save_path + "d_a_w_z.h5")
            self.d_A_z.compile(loss="mse",
                               optimizer=self.d_optimizer,
                               metrics=["accuracy"])

            input_a = Input(shape=self.input_shape)
            input_b = Input(shape=self.input_shape)

            # create fake data
            fake_a = self.g_bA_z(input_b)  # genuine_B -> fake_A
            fake_b = self.g_aB_z(input_a)  # genuine_A -> fake_B

            # reconstruct
            recon_b = self.g_aB_z(fake_a)  # (genuine_B -> ) fake_A -> genuine_B
            recon_a = self.g_bA_z(fake_b)  # (genuine_A -> ) fake_B -> genuine_A
            self.g_aBA_z = Model(inputs=input_a, outputs=recon_a)
            self.g_bAB_z = Model(inputs=input_b, outputs=recon_b)

            # not convert
            nc_a = self.g_bA_z(input_a)  # genuine_A -> genuine_A (gen: B -> A)
            nc_b = self.g_aB_z(input_b)  # genuine_B -> genuine_B (gen: A -> B)
            self.nc_bA_z = Model(inputs=input_a, outputs=nc_a)
            self.nc_aB_z = Model(inputs=input_b, outputs=nc_b)

            self.d_A_z.trainable = False
            self.d_B_z.trainable = False

            # deceive disc
            deceive_A = self.d_A_z(fake_a)
            deceive_B = self.d_B_z(fake_b)
            self.c_aB_z = Model(inputs=input_a, outputs=deceive_B)  # genuine_B -> fake_A -> Genuine(expected value)
            self.c_bA_z = Model(inputs=input_b, outputs=deceive_A)  # genuine_A -> fake_B -> Genuine(expected value)

            self.c_aB_z.compile(loss="mse",
                                optimizer=self.g_optimizer,
                                metrics=["accuracy"])
            self.c_bA_z.compile(loss="mse",
                                optimizer=self.g_optimizer,
                                metrics=["accuracy"])

            self.g_aBA_z.compile(loss="mae",
                                 optimizer=self.g_optimizer)
            self.g_bAB_z.compile(loss="mae",
                                 optimizer=self.g_optimizer)

            self.nc_bA_z.compile(loss="mae",
                                 optimizer=self.g_optimizer)
            self.nc_aB_z.compile(loss="mae",
                                 optimizer=self.g_optimizer)

            # =======================================================================

            """ [theta] """
            self.g_aB_t = model_from_json(open(self.save_path + "g_ab_t.json", "r").read())
            self.g_aB_t.load_weights(self.save_path + "g_ab_w_t.h5")

            self.d_B_t = model_from_json(open(self.save_path + "d_b_t.json", "r").read())
            self.d_B_t.load_weights(self.save_path + "d_b_w_t.h5")
            self.d_B_t.compile(loss="mse",
                               optimizer=self.d_optimizer,
                               metrics=["accuracy"])

            self.g_bA_t = model_from_json(open(self.save_path + "g_ba_t.json", "r").read())
            self.g_bA_t.load_weights(self.save_path + "g_ba_w_t.h5")

            self.d_A_t = model_from_json(open(self.save_path + "d_a_t.json", "r").read())
            self.d_A_t.load_weights(self.save_path + "d_a_w_t.h5")
            self.d_A_t.compile(loss="mse",
                               optimizer=self.d_optimizer,
                               metrics=["accuracy"])

            input_a = Input(shape=self.input_shape)
            input_b = Input(shape=self.input_shape)

            # create fake data
            fake_a = self.g_bA_t(input_b)  # genuine_B -> fake_A
            fake_b = self.g_aB_t(input_a)  # genuine_A -> fake_B

            # reconstruct
            recon_b = self.g_aB_t(fake_a)  # (genuine_B -> ) fake_A -> genuine_B
            recon_a = self.g_bA_t(fake_b)  # (genuine_A -> ) fake_B -> genuine_A
            self.g_aBA_t = Model(inputs=input_a, outputs=recon_a)
            self.g_bAB_t = Model(inputs=input_b, outputs=recon_b)

            # not convert
            nc_a = self.g_bA_t(input_a)  # genuine_A -> genuine_A (gen: B -> A)
            nc_b = self.g_aB_t(input_b)  # genuine_B -> genuine_B (gen: A -> B)
            self.nc_bA_t = Model(inputs=input_a, outputs=nc_a)
            self.nc_aB_t = Model(inputs=input_b, outputs=nc_b)

            self.d_A_t.trainable = False
            self.d_B_t.trainable = False

            # deceive disc
            deceive_A = self.d_A_t(fake_a)
            deceive_B = self.d_B_t(fake_b)
            self.c_aB_t = Model(inputs=input_a, outputs=deceive_B)  # genuine_B -> fake_A -> Genuine(expected value)
            self.c_bA_t = Model(inputs=input_b, outputs=deceive_A)  # genuine_A -> fake_B -> Genuine(expected value)

            self.c_aB_t.compile(loss="mse",
                                optimizer=self.g_optimizer,
                                metrics=["accuracy"])
            self.c_bA_t.compile(loss="mse",
                                optimizer=self.g_optimizer,
                                metrics=["accuracy"])

            self.g_aBA_t.compile(loss="mae",
                                 optimizer=self.g_optimizer)
            self.g_bAB_t.compile(loss="mae",
                                 optimizer=self.g_optimizer)

            self.nc_bA_t.compile(loss="mae",
                                 optimizer=self.g_optimizer)
            self.nc_aB_t.compile(loss="mae",
                                 optimizer=self.g_optimizer)

            # open(self.save_path + "log.txt", "w").write("")  # log.txtをクリア

        else:
            self.gen = Generator(input_shape=self.input_shape)  # 1 + self.fft_len // 2 + 1
            self.disc = Discriminator(input_shape=self.input_shape)  # 1 + self.fft_len // 2 + 1

            """ [Z] """
            # genuine_A -> fake_B
            self.g_aB_z = self.gen.build_generator_z(filters=16)

            # B -> genuine or fake
            self.d_B_z = self.disc.build_discriminator(filters=16)
            self.d_B_z.compile(loss="mse",
                               optimizer=self.d_optimizer,
                               metrics=["accuracy"])

            # genuine_B -> fake_A
            self.g_bA_z = self.gen.build_generator_z(filters=16)

            # A -> genuine or fake
            self.d_A_z = self.disc.build_discriminator(filters=16)
            self.d_A_z.compile(loss="mse",
                               optimizer=self.d_optimizer,
                               metrics=["accuracy"])

            input_a_z = Input(shape=self.input_shape)
            input_b_z = Input(shape=self.input_shape)

            # create fake data
            fake_a_z = self.g_bA_z(input_b_z)  # genuine_B -> fake_A
            fake_b_z = self.g_aB_z(input_a_z)  # genuine_A -> fake_B

            # reconstruct
            recon_b_z = self.g_aB_z(fake_a_z)  # (genuine_B -> ) fake_A -> genuine_B
            recon_a_z = self.g_bA_z(fake_b_z)  # (genuine_A -> ) fake_B -> genuine_A
            self.g_aBA_z = Model(inputs=input_a_z, outputs=recon_a_z)
            self.g_bAB_z = Model(inputs=input_b_z, outputs=recon_b_z)

            # not convert
            nc_a_z = self.g_bA_z(input_a_z)  # genuine_A -> genuine_A (gen: B -> A)
            nc_b_z = self.g_aB_z(input_b_z)  # genuine_B -> genuine_B (gen: A -> B)
            self.nc_bA_z = Model(inputs=input_a_z, outputs=nc_a_z)
            self.nc_aB_z = Model(inputs=input_b_z, outputs=nc_b_z)

            self.d_A_z.trainable = False
            self.d_B_z.trainable = False

            # deceive disc
            deceive_A_z = self.d_A_z(fake_a_z)
            deceive_B_z = self.d_B_z(fake_b_z)
            self.c_aB_z = Model(inputs=input_a_z, outputs=deceive_B_z)  # genuine_B -> fake_A -> Genuine(expected value)
            self.c_bA_z = Model(inputs=input_b_z, outputs=deceive_A_z)  # genuine_A -> fake_B -> Genuine(expected value)

            self.c_aB_z.compile(loss="mse",
                                optimizer=self.g_optimizer,
                                metrics=["accuracy"])
            self.c_bA_z.compile(loss="mse",
                                optimizer=self.g_optimizer,
                                metrics=["accuracy"])

            self.g_aBA_z.compile(loss="mae",
                                 optimizer=self.g_optimizer)
            self.g_bAB_z.compile(loss="mae",
                                 optimizer=self.g_optimizer)

            self.nc_bA_z.compile(loss="mae",
                                 optimizer=self.g_optimizer)
            self.nc_aB_z.compile(loss="mae",
                                 optimizer=self.g_optimizer)

            self.model_save(self.g_aB_z, "g_ab_z.json")
            self.model_save(self.d_B_z, "d_b_z.json")
            self.model_save(self.g_bA_z, "g_ba_z.json")
            self.model_save(self.d_A_z, "d_a_z.json")

            # =========================================================

            """ [theta] """
            # genuine_A -> fake_B
            self.g_aB_t = self.gen.build_generator_t(filters=16)

            # B -> genuine or fake
            self.d_B_t = self.disc.build_discriminator(filters=16)
            self.d_B_t.compile(loss="mse",
                               optimizer=self.d_optimizer,
                               metrics=["accuracy"])

            # genuine_B -> fake_A
            self.g_bA_t = self.gen.build_generator_t(filters=16)

            # A -> genuine or fake
            self.d_A_t = self.disc.build_discriminator(filters=16)
            self.d_A_t.compile(loss="mse",
                               optimizer=self.d_optimizer,
                               metrics=["accuracy"])

            input_a_t = Input(shape=self.input_shape)
            input_b_t = Input(shape=self.input_shape)

            # create fake data
            fake_a_t = self.g_bA_t(input_b_t)  # genuine_B -> fake_A
            fake_b_t = self.g_aB_t(input_a_t)  # genuine_A -> fake_B

            # reconstruct
            recon_b_t = self.g_aB_t(fake_a_t)  # (genuine_B -> ) fake_A -> genuine_B
            recon_a_t = self.g_bA_t(fake_b_t)  # (genuine_A -> ) fake_B -> genuine_A
            self.g_aBA_t = Model(inputs=input_a_t, outputs=recon_a_t)
            self.g_bAB_t = Model(inputs=input_b_t, outputs=recon_b_t)

            # not convert
            nc_a_t = self.g_bA_t(input_a_t)  # genuine_A -> genuine_A (gen: B -> A)
            nc_b_t = self.g_aB_t(input_b_t)  # genuine_B -> genuine_B (gen: A -> B)
            self.nc_bA_t = Model(inputs=input_a_t, outputs=nc_a_t)
            self.nc_aB_t = Model(inputs=input_b_t, outputs=nc_b_t)

            self.d_A_t.trainable = False
            self.d_B_t.trainable = False

            # deceive disc
            deceive_A_t = self.d_A_t(fake_a_t)
            deceive_B_t = self.d_B_t(fake_b_t)
            self.c_aB_t = Model(inputs=input_a_t, outputs=deceive_B_t)  # genuine_B -> fake_A -> Genuine(expected value)
            self.c_bA_t = Model(inputs=input_b_t, outputs=deceive_A_t)  # genuine_A -> fake_B -> Genuine(expected value)

            self.c_aB_t.compile(loss="mse",
                                optimizer=self.g_optimizer,
                                metrics=["accuracy"])
            self.c_bA_t.compile(loss="mse",
                                optimizer=self.g_optimizer,
                                metrics=["accuracy"])

            self.g_aBA_t.compile(loss="mae",
                                 optimizer=self.g_optimizer)
            self.g_bAB_t.compile(loss="mae",
                                 optimizer=self.g_optimizer)

            self.nc_bA_t.compile(loss="mae",
                                 optimizer=self.g_optimizer)
            self.nc_aB_t.compile(loss="mae",
                                 optimizer=self.g_optimizer)

            self.model_save(self.g_aB_t, "g_ab_t.json")
            self.model_save(self.d_B_t, "d_b_t.json")
            self.model_save(self.g_bA_t, "g_ba_t.json")
            self.model_save(self.d_A_t, "d_a_t.json")

    def model_save(self, model, name):
        j = model.to_json()
        open(self.save_path + name, "w").write(j)

    def save_weight(self):
        """ [Z] """
        self.g_aB_z.save_weights(self.save_path + "g_ab_w_z.h5")
        self.d_B_z.save_weights(self.save_path + "d_b_w_z.h5")
        self.g_bA_z.save_weights(self.save_path + "g_ba_w_z.h5")
        self.d_A_z.save_weights(self.save_path + "d_a_w_z.h5")

        """ [theta] """
        self.g_aB_t.save_weights(self.save_path + "g_ab_w_t.h5")
        self.d_B_t.save_weights(self.save_path + "d_b_w_t.h5")
        self.g_bA_t.save_weights(self.save_path + "g_ba_w_t.h5")
        self.d_A_t.save_weights(self.save_path + "d_a_w_t.h5")

    def train(self, epochs=1000, batch_size=20, save_interval=5):
        va = Voice_Analy()

        wave_files_a = glob.glob(self.data_path + "data_A/*.wav")
        wave_files_b = glob.glob(self.data_path + "data_B/*.wav")

        real_label = np.ones((batch_size, 1))
        fake_label = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            for (file_a, file_b) in zip(wave_files_a, wave_files_b):

                fs_a, data_a = va.get_data_from_wave(file_a)
                data_a = np.array(data_a, dtype=np.float32).copy()
                data_a /= 32768.0
                stft_a = va.stft(data_a, self.window, self.step)
                stft_a_z, stft_a_theta = to_polar(stft_a)
                stft_a_z /= 256.0
                stft_a_theta /= np.pi

                fs_b, data_b = va.get_data_from_wave(file_b)
                data_b = np.array(data_b, dtype=np.float32).copy()
                data_b /= 32768.0
                stft_b = va.stft(data_b, self.window, self.step)
                stft_b_z, stft_b_theta = to_polar(stft_b)
                stft_b_z /= 256.0
                stft_b_theta /= np.pi

                num_batches = 50

                print("A:", os.path.basename(file_a))
                print("B:", os.path.basename(file_b))

                for batch in range(num_batches):
                    try:
                        idx_a = np.random.randint(0, stft_a.shape[0], batch_size)
                        input_a_z = stft_a_z[idx_a]
                        input_a_t = stft_a_theta[idx_a]

                        idx_b = np.random.randint(0, stft_b.shape[0], batch_size)
                        input_b_z = stft_b_z[idx_b]
                        input_b_t = stft_b_theta[idx_b]

                        # ======================================================
                        """ [Z] """

                        """ B -> A """
                        fake_a_z = self.g_bA_z.predict(input_b_z)
                        d_loss_real_ba = self.d_A_z.train_on_batch(input_a_z, real_label)
                        d_loss_fake_ba = self.d_A_z.train_on_batch(fake_a_z, fake_label)
                        d_loss_ba_z = 0.5 * np.add(d_loss_real_ba, d_loss_fake_ba)

                        c_loss_ba_z = self.c_bA_z.train_on_batch(input_b_z, real_label)

                        r_loss_ba_z = self.g_bAB_z.train_on_batch(input_b_z, input_b_z)
                        nc_loss_ba_z = self.nc_bA_z.train_on_batch(input_a_z, input_a_z)

                        """ A -> B """
                        fake_b_z = self.g_aB_z.predict(input_a_z)
                        d_loss_real_ab = self.d_B_z.train_on_batch(input_b_z, real_label)
                        d_loss_fake_ab = self.d_B_z.train_on_batch(fake_b_z, fake_label)
                        d_loss_ab_z = 0.5 * np.add(d_loss_real_ab, d_loss_fake_ab)

                        c_loss_ab_z = self.c_aB_z.train_on_batch(input_a_z, real_label)

                        r_loss_ab_z = self.g_aBA_z.train_on_batch(input_a_z, input_a_z)
                        nc_loss_ab_z = self.nc_aB_z.train_on_batch(input_b_z, input_b_z)

                        # ======================================================
                        """ [theta] """

                        """ B -> A """
                        fake_a_t = self.g_bA_t.predict(input_b_t)
                        d_loss_real_ba = self.d_A_t.train_on_batch(input_a_t, real_label)
                        d_loss_fake_ba = self.d_A_t.train_on_batch(fake_a_t, fake_label)
                        d_loss_ba_t = 0.5 * np.add(d_loss_real_ba, d_loss_fake_ba)

                        c_loss_ba_t = self.c_bA_t.train_on_batch(input_b_t, real_label)

                        r_loss_ba_t = self.g_bAB_t.train_on_batch(input_b_t, input_b_t)
                        nc_loss_ba_t = self.nc_bA_t.train_on_batch(input_a_t, input_a_t)

                        """ A -> B """
                        fake_b_t = self.g_aB_t.predict(input_a_t)
                        d_loss_real_ab = self.d_B_t.train_on_batch(input_b_t, real_label)
                        d_loss_fake_ab = self.d_B_t.train_on_batch(fake_b_t, fake_label)
                        d_loss_ab_t = 0.5 * np.add(d_loss_real_ab, d_loss_fake_ab)

                        c_loss_ab_t = self.c_aB_t.train_on_batch(input_a_t, real_label)

                        r_loss_ab_t = self.g_aBA_t.train_on_batch(input_a_t, input_a_t)
                        nc_loss_ab_t = self.nc_aB_t.train_on_batch(input_b_t, input_b_t)

                        # ==============================================================

                        if batch % save_interval == 0:
                            """ B -> A """
                            z = self.g_bA_z.predict(stft_b_z) * 256
                            t = self.g_bA_t.predict(stft_b_theta) * np.pi
                            data_bA = to_rect(z, t, stft_b.shape)
                            istft = va.istft(data_bA, self.window, self.step)
                            res = (istft * 32768.0).astype(np.int16)
                            p = self.res_path + "data_bA/" + os.path.basename(file_b)
                            p = p[:-4] + "_%d.wav" % epoch
                            va.save_wave(p, fs_a, res)

                            plt.figure()
                            plt.subplot(211)
                            plt.plot(data_b)
                            plt.subplot(212)
                            plt.plot(istft)
                            plt.ylim(-1, 1)
                            plt.savefig((self.res_path + "pic/data_bA/{}").format(
                                os.path.basename(file_a)[:-4] + "_%d.png" % epoch))
                            # plt.clf()
                            plt.close()

                            """ A -> B """
                            z = self.g_aB_z.predict(stft_a_z) * 256
                            t = self.g_aB_t.predict(stft_a_theta) * np.pi
                            data_aB = to_rect(z, t, stft_a.shape)
                            istft = va.istft(data_aB, self.window, self.step)
                            res = (istft * 32768.0).astype(np.int16)
                            p = self.res_path + "data_aB/" + os.path.basename(file_a)
                            p = p[:-4] + "_%d.wav" % epoch
                            va.save_wave(p, fs_b, res)

                            plt.figure()
                            plt.subplot(211)
                            plt.plot(data_a)
                            plt.subplot(212)
                            plt.plot(istft)
                            plt.ylim(-1, 1)
                            plt.savefig((self.res_path + "pic/data_aB/{}").format(
                                os.path.basename(file_b)[:-4] + "_%d.png" % epoch))
                            # plt.clf()
                            plt.close()

                            self.save_weight()

                            # メモリ解放
                            del z, t,
                            del data_bA, data_aB
                            del istft, res
                            gc.collect()

                        log = ("epoch:{:4}, batch:{:2}  " +
                               "Z[dl_ba:{:.4f}, da:{:.2f}, cl_ba:{:.4f}, ca:{:.2f}, rl_ba:{:.4f}, nl_ba:{:.4f}] " +
                               "T[dl_ba:{:.4f}, da:{:.2f}, cl_ba:{:.4f}, ca:{:.2f}, rl_ba:{:.4f}, nl_ba:{:.4f}] " +
                               "Z[dl_ab:{:.4f}, da:{:.2f}, cl_ab:{:.4f}, ca:{:.2f}, rl_ab:{:.4f}, nl_ab:{:.4f}] " +
                               "T[dl_ab:{:.4f}, da:{:.2f}, cl_ab:{:.4f}, ca:{:.2f}, rl_ab:{:.4f}, nl_ab:{:.4f}] "). \
                            format(
                            epoch,
                            batch,

                            d_loss_ba_z[0],
                            d_loss_ba_z[1],
                            c_loss_ba_z[0],
                            c_loss_ba_z[1],
                            r_loss_ba_z,
                            nc_loss_ba_z,

                            d_loss_ba_t[0],
                            d_loss_ba_t[1],
                            c_loss_ba_t[0],
                            c_loss_ba_t[1],
                            r_loss_ba_t,
                            nc_loss_ba_t,

                            d_loss_ab_z[0],
                            d_loss_ab_z[1],
                            c_loss_ab_z[0],
                            c_loss_ab_z[1],
                            r_loss_ab_z,
                            nc_loss_ab_z,

                            d_loss_ab_t[0],
                            d_loss_ab_t[1],
                            c_loss_ab_t[0],
                            c_loss_ab_t[1],
                            r_loss_ab_t,
                            nc_loss_ab_t,
                        )
                        open(self.save_path + "log.txt", "a").write(log + "\n")
                        print(log)

                    except:
                        self.save_weight()
                        print(sys.exc_info())
                        sys.exit()

                    if __name__ == "__main__":
                        print("test finished")
                        sys.exit()

                    # メモリ解放
                    del input_a_z, input_a_t
                    del input_b_z, input_b_t
                    del idx_a, idx_b
                    del fake_a_z, fake_a_t
                    del fake_b_z, fake_b_t
                    del d_loss_ab_z, d_loss_ab_t
                    del d_loss_ba_z, d_loss_ba_t
                    del c_loss_ab_z, c_loss_ab_t
                    del c_loss_ba_z, c_loss_ba_t
                    del r_loss_ab_z, r_loss_ab_t
                    del r_loss_ba_z, r_loss_ba_t
                    del nc_loss_ab_z, nc_loss_ab_t
                    del nc_loss_ba_z, nc_loss_ba_t
                    del log
                    gc.collect()

                # メモリ解放
                del data_a, data_b
                del fs_a, fs_b
                del stft_a, stft_a_z, stft_a_theta
                del stft_b, stft_b_z, stft_b_theta
                gc.collect()

                copy_tree("./save", "./backup")
                print("backup finished :  ./save to ./backup")


if __name__ == "__main__":
    t = Train()
    t.train()
