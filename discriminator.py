import numpy as np
import sys

from keras.models import Model
from keras.layers import Input, Conv1D, Reshape, LeakyReLU, BatchNormalization, Flatten, Dense, Conv2D, Dropout


class Discriminator:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        pass


    def build_discriminator(self, filters, summary=False):
        def conv(layer, f, k_size=4, s_size=2, dropout=0.0, normalize=True):
            c = Conv1D(filters=f,
                       kernel_size=k_size,
                       strides=s_size,
                       padding="same")(layer)
            c = LeakyReLU(alpha=0.2)(c)
            if dropout:
                c = Dropout(dropout)(c)

            if normalize:
                c = BatchNormalization()(c)
            return c

        input = Input(shape=self.input_shape)

        c = Reshape(target_shape=(-1, 1))(input)
        c = conv(c, filters, normalize=False)
        c = conv(c, filters * 2, k_size=3, dropout=0.5)
        c = conv(c, filters * 4, k_size=3, dropout=0.5)
        c = conv(c, filters * 8, k_size=3, dropout=0.5)
        c = Flatten()(c)
        c = Dense(units=128)(c)
        c = LeakyReLU(0.2)(c)
        output = Dense(units=1, activation="sigmoid")(c)

        model = Model(inputs=input, outputs=output)
        if summary:
            model.summary()

        return model


if __name__ == "__main__":
    disc = Discriminator(input_shape=(1024 // 2, 2))
    disc.build_discriminator(filters=16, summary=True)
