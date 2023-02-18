# utils to use across all files
import spafe
import config as config
import matplotlib.pyplot as plt
import torch
import numpy as np
import logging


# visualizing
# need to import torch functions here and make them compatible with numpy
# plot spectrogram and plot time


def plot_spectrogram(waveform):
    spafe.utils.vis.show_features(waveform, 'title', 'y', 'x')


def get_class(kls):
    """ get class for given string

    Note:

    Args:
        kls: (string) : full class name to be used to find either the class or method
        in the module

    Returns:
        (string): Resulting class.
    """
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


import my_torch.torchio


def get_average_power(signal, use_buffer=False, show_graph=False):
    """ get average power for a signal

    Note: can use buffer to change between calculating power over whole signal or using a buffer

    Args:
        signal: (tensor or ndarray) : time domain tensor of sound
        use_buffer: (boolean) : determines if a buffer should be used or not
        show_graph: (boolean) : draw a graph if true and buffer is true

    Returns:
        (int): average power over signal
    """

    if torch.is_tensor(signal):
        signal = signal.numpy()[0, :]

    if use_buffer:
        lx = 1000  # length
        p = 500  # overlap
        buf = buffer(signal, lx, p)
        average_pow = []
        for b in buf:
            average_pow.append(np.sum(np.square(b)) / lx)
        if show_graph:
            plt.plot(average_pow)
        return np.mean(average_pow)
    else:
        average_power = np.square(np.linalg.norm(signal, ord=2)) / len(signal)
        return average_power


def snr(signal, noise):
    if len(signal) != len(noise):
        print('error lengths diff')
        return
    if torch.is_tensor(signal):
        length = signal.size(1)
        signal_power = torch.square(torch.linalg.norm(signal, ord=2)) / length
        noise_power = torch.square(torch.linalg.norm(noise, ord=2)) / length
        return 10 * torch.log10(signal_power / noise_power)
    else:
        length = len(signal)
        signal_power = np.square(np.linalg.norm(signal, ord=2)) / length
        noise_power = np.square(np.linalg.norm(noise, ord=2)) / length
    return 10 * np.log10(signal_power / noise_power)

def snr_matlab(signal, noise):
    if len(signal) != len(noise):
        print('error lengths diff')
        return
    if torch.is_tensor(signal):
        signal_rss = torch.linalg.norm(signal, ord=2)
        noise_rss = torch.linalg.norm(noise, ord=2)
        return 20 * torch.log10(signal_rss / noise_rss)
    else:
        signal_rss = np.square(np.linalg.norm(signal, ord=2))
        noise_rss = np.square(np.linalg.norm(noise, ord=2))
    return 10 * np.log10(signal_rss / noise_rss)

def normalize(audio):
    std = np.round(np.std(audio)) * 6  # 97%
    new_audio = audio / std
    if np.mean(new_audio) > 0.01:
        raise logging.warning('mean is large')
    return new_audio


def normalize_dbs(sig, rms_level=0):
    """
    Normalize the signal given a certain technique (peak or rms).
    Args:
        - infile    (str) : input filename/path.
        - rms_level (int) : rms level in dB.
    """
    # read input file

    # linear rms level and scaling factor
    r = 10 ** (rms_level / 10.0)
    a = np.sqrt((len(sig) * r ** 2) / np.sum(sig ** 2))

    # normalize
    y = sig * a

    return y


def buffer(x, n, p=0, opt=None):
    """
    Mimic MATLAB routine to generate buffer array

    MATLAB docs here: https://se.mathworks.com/help/signal/ref/buffer.html

    Parameters
    ----------
    x: ndarray
        Signal array
    n: int
        Number of data segments
    p: int
        Number of values to overlap
    opt: str
        Initial condition options. default sets the first `p` values to zero,
        while 'nodelay' begins filling the buffer immediately.

    Returns
    -------
    result : (n,n) ndarray
        Buffer array created from x
    """
    import numpy as np

    if opt not in [None, 'nodelay']:
        raise ValueError('{} not implemented'.format(opt))

    i = 0
    first_iter = True
    while i < len(x):
        if first_iter:
            if opt == 'nodelay':
                # No zeros at array start
                result = x[:n]
                i = n
            else:
                # Start with `p` zeros
                result = np.hstack([np.zeros(p), x[:n - p]])
                i = n - p
            # Make 2D array and pivot
            result = np.expand_dims(result, axis=0).T
            first_iter = False
            continue

        # Create next column, add `p` results from last col if given
        col = x[i:i + (n - p)]
        if p != 0:
            col = np.hstack([result[:, -1][-p:], col])
        i += n - p

        # Append zeros if last row and not length `n`
        if len(col) < n:
            col = np.hstack([col, np.zeros(n - len(col))])

        # Combine result with next row
        result = np.hstack([result, np.expand_dims(col, axis=0).T])

    return result.T


# remove this
def periodic_power(x, lx, p):
    buf = buffer(x, lx, p)
    average_pow = []
    for b in buf:
        average_pow.append(np.sum(np.square(b)) / lx)
    return average_pow
