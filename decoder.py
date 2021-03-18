import numpy as np
import scipy.io.wavfile as wav
from PIL import Image
from scipy import signal

FREQ = 20800
NOAA_LINE_LENGTH = 2080


class Decoder(object):
    def __init__(self, filename):
        self.fs, self.signal = wav.read(filename)
        if self.fs != FREQ:
            self.resample()

        # Keep only one channel if audio is stereo
        if self.signal.ndim > 1:
            self.signal = self.signal[:, 0]

        truncate = FREQ * int(len(self.signal) // FREQ)
        self.signal = self.signal[:truncate]
        self.image = None

    def resample(self, out_filename=None):
        if self.fs != FREQ:
            coef = FREQ / self.fs
            samples = int(coef * len(self.signal))
            self.signal = signal.resample(self.signal, samples)
            self.fs = FREQ
            if out_filename is not None:
                wav.write(out_filename, FREQ, self.signal)

    def decode(self, outfile=None, show_img=False):
        hilbert = signal.hilbert(self.signal)
        filtered = signal.medfilt(np.abs(hilbert), 5)
        reshaped = filtered.reshape(len(filtered) // 5, 5)
        digitized = self._digitize(reshaped[:, 2])
        matrix = self._reshape(digitized)
        self.image = Image.fromarray(matrix)
        if not outfile is None:
            self.image.save(outfile)
        if show_img:
            self.image.show()
        return matrix

    def _digitize(self, signal, plow=0.5, phigh=99.5):
        '''
        Convert signal to numbers between 0 and 255.
        '''
        (low, high) = np.percentile(signal, (plow, phigh))
        delta = high - low
        data = np.round(255 * (signal - low) / delta)
        data[data < 0] = 0
        data[data > 255] = 255
        return data.astype(np.uint8)

    def _reshape(self, signal):
        '''
        Find sync frames and reshape the 1D signal into a 2D image.
        Finds the sync A frame by looking at the maximum values of the cross
        correlation between the signal and a hardcoded sync A frame.
        The expected distance between sync A frames is 2080 samples, but with
        small variations because of Doppler effect.
        '''
        # sync frame to find: seven impulses and some black pixels (some lines
        # have something like 8 black pixels and then white ones)
        syncA = [0, 128, 255, 128] * 7 + [0] * 7

        # list of maximum correlations found: (index, value)
        peaks = [(0, 0)]

        # minimum distance between peaks
        mindistance = 2000

        # need to shift the values down to get meaningful correlation values
        signalshifted = [x - 128 for x in signal]
        syncA = [x - 128 for x in syncA]
        for i in range(len(signal) - len(syncA)):
            corr = np.dot(syncA, signalshifted[i: i + len(syncA)])

            # if previous peak is too far, keep it and add this value to the
            # list as a new peak
            if i - peaks[-1][0] > mindistance:
                peaks.append((i, corr))

            # else if this value is bigger than the previous maximum, set this
            # one
            elif corr > peaks[-1][1]:
                peaks[-1] = (i, corr)

        # create image matrix starting each line on the peaks found
        matrix = []
        for i in range(len(peaks) - 1):
            matrix.append(signal[peaks[i][0]: peaks[i][0] + 2080])

        return np.array(matrix)
