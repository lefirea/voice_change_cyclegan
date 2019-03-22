import numpy as np
import sys

from keras.models import Model, Input
from keras.layers import Conv1D, UpSampling1D, Dropout, BatchNormalization, LeakyReLU, Reshape, Dense, Add
from keras.layers import AveragePooling1D, Activation


class Generator:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.dim = input_shape[0]
        pass

    def build_generator_z(self, filters, summary=False):
        def conv(layer, f, k_size=4, s_size=2):
            c = Conv1D(filters=f,
                       kernel_size=k_size,
                       strides=s_size,
                       padding="same")(layer)
            c = LeakyReLU(alpha=0.2)(c)
            c = BatchNormalization()(c)
            return c

        def deconv(layer, f, drop_rate=0, k_size=4, s_size=1):
            c = UpSampling1D(2)(layer)
            c = Conv1D(filters=f,
                       kernel_size=k_size,
                       strides=s_size,
                       padding="same",
                       activation="relu")(c)
            if drop_rate:
                c = Dropout(drop_rate)(c)
            c = BatchNormalization()(c)
            return c

        def resnet(layer, f, num_fase=3, num_blocks=3):
            x = layer

            for i in range(num_fase):
                for b in range(num_blocks):
                    if b % 3 == 0 and b != 0:
                        x = AveragePooling1D(2)(x)
                        f *= 2

                    if i % 3 == 0:
                        x = Conv1D(filters=f // 2,
                                   kernel_size=5,
                                   padding="same")(x)

                    shortcut = x
                    shortcut = BatchNormalization()(shortcut)

                    x = Conv1D(filters=f,
                               kernel_size=3,
                               padding="same")(x)
                    x = BatchNormalization()(x)
                    x = Activation("relu")(x)

                    x = Conv1D(filters=f // 2,
                               kernel_size=3,
                               padding="same")(x)
                    x = BatchNormalization()(x)

                    x = Add()([x, shortcut])

                    x = Activation("relu")(x)

            return x

        input = Input(shape=self.input_shape)
        reshape = Reshape(target_shape=(-1, 1))(input)
        c = conv(reshape, filters, k_size=5)
        c = conv(c, filters * 2, k_size=5)
        c = conv(c, filters * 4, k_size=5)
        c = conv(c, filters * 8, k_size=5)

        # c = resnet(c, filters * 8, num_fase=3)

        c = deconv(c, filters * 4, k_size=5)
        c = deconv(c, filters * 2, k_size=5)
        c = deconv(c, filters, k_size=5)

        c = UpSampling1D(2)(c)
        output = Conv1D(filters=1,
                        kernel_size=5,
                        strides=1,
                        padding="same",
                        activation="sigmoid")(c)
        output = Reshape(target_shape=(-1,))(output)

        d = Dense(units=self.dim // 2)(input)
        d = Dropout(0.5)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Dense(units=self.dim // 4)(d)
        d = Dropout(0.5)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Dense(units=self.dim, activation="sigmoid")(d)

        output = Add()([output, d])

        model = Model(inputs=input, outputs=output)

        if summary:
            model.summary()

        return model

    def build_generator_t(self, filters, summary=False):
        def conv(layer, f, k_size=4, s_size=2):
            c = Conv1D(filters=f,
                       kernel_size=k_size,
                       strides=s_size,
                       padding="same")(layer)
            c = LeakyReLU(alpha=0.2)(c)
            c = BatchNormalization()(c)
            return c

        def deconv(layer, f, drop_rate=0, k_size=4, s_size=1):
            c = UpSampling1D(2)(layer)
            c = Conv1D(filters=f,
                       kernel_size=k_size,
                       strides=s_size,
                       padding="same",
                       activation="relu")(c)
            if drop_rate:
                c = Dropout(drop_rate)(c)
            c = BatchNormalization()(c)
            return c

        def resnet(layer, f, num_fases=3, num_blocks=3):
            x = layer

            for i in range(num_fases):
                for b in range(num_blocks):
                    if b % 3 == 0 and b != 0:
                        x = AveragePooling1D(2)(x)
                        f *= 2

                    if i % 3 == 0:
                        x = Conv1D(filters=f // 2,
                                   kernel_size=5,
                                   padding="same")(x)

                    shortcut = x
                    shortcut = BatchNormalization()(shortcut)

                    x = Conv1D(filters=f,
                               kernel_size=3,
                               padding="same")(x)
                    x = BatchNormalization()(x)
                    x = Activation("relu")(x)

                    x = Conv1D(filters=f // 2,
                               kernel_size=3,
                               padding="same")(x)
                    x = BatchNormalization()(x)

                    x = Add()([x, shortcut])

                    x = Activation("relu")(x)

            return x

        input = Input(shape=self.input_shape)
        reshape = Reshape(target_shape=(-1, 1))(input)
        c = conv(reshape, filters, k_size=5)
        c = conv(c, filters * 2, k_size=5)
        c = conv(c, filters * 4, k_size=5)
        c = conv(c, filters * 8, k_size=5)

        # c = resnet(c, filters * 8, num_fases=3)

        c = deconv(c, filters * 4, k_size=5)
        c = deconv(c, filters * 2, k_size=5)
        c = deconv(c, filters, k_size=5)

        c = UpSampling1D(2)(c)
        output = Conv1D(filters=1,
                        kernel_size=5,
                        strides=1,
                        padding="same",
                        activation="tanh")(c)
        output = Reshape(target_shape=(-1,))(output)

        d = Dense(units=self.dim // 2)(input)
        d = Dropout(0.5)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Dense(units=self.dim // 4)(d)
        d = Dropout(0.5)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Dense(units=self.dim, activation="tanh")(d)

        output = Add()([output, d])

        model = Model(inputs=input, outputs=output)

        if summary:
            model.summary()

        return model


if __name__ == "__main__":
    gen = Generator(input_shape=(1024 // 2, 2))
    g1 = gen.build_generator_z(filters=16, summary=True)
