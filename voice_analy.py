import numpy as np
import scipy as sp
from scipy.io import wavfile


class Voice_Analy:
    def __init__(self):
        pass

    def get_data_from_wave(self, path):
        sample_rate, data = wavfile.read(path)
        return sample_rate, data

    def save_wave(self, path, sample_rate, data):
        wavfile.write(path, sample_rate, data.astype(np.int16))

    def stft(self, x, window, step):
        length = len(x)
        N = len(window)
        M = int(sp.ceil(float(length - N + step) / step))
        new_x = np.zeros(N + (M - 1) * step).astype(np.float32)
        new_x[:length] = x

        X = np.zeros([M, N], dtype=np.complex64)
        for m in range(M):
            start = step * m
            X[m, :] = np.fft.fft(new_x[start:start + N] * window)
        return X

    def istft(self, X, window, step):
        M, N = X.shape

        length = (M - 1) * step + N
        x = np.zeros(length, dtype=np.float32)
        wsum = np.zeros(length, dtype=np.float32)

        for m in range(M):
            start = step * m
            x[start:start + N] = x[start:start + N] + np.fft.ifft(X[m, :]).real * window
            wsum[start:start + N] += window ** 2
        pos = (wsum != 0)

        x[pos] /= wsum[pos]
        return x


if __name__ == "__main__":
    va = Voice_Analy()
    va.__init__()
