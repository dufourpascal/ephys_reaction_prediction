import math
import numpy as np
# import matplotlib.pyplot as plt


def nextpow2(i):
    n = 1
    while n < i:
        n *= 2
    return n


def time_domain_wavelet(wav_time, freq, n_cycles):
    """
    Create a time domain complex cosine wavelet given a frequency, the number
    of cycles in that frequency and a time vector of the wavelet.

    :param wav_time: Time vector of the wavelet must be centered around zero,
    so that the cosine peaks at time zero.
    :param freq: [int]
    :param n_cycles: [int]
    :return: wavelet as 1d array of complex numbers
    """
    twos2 = 2 * (n_cycles / (2 * np.pi * freq)) ** 2
    sin_w = np.exp(2 * 1j * np.pi * freq * wav_time)
    gaus_w = np.exp(-wav_time ** 2 / twos2)
    cmw = sin_w * gaus_w
    return cmw


def frequency_domain_wavelet(wav_time, freq, n_cycles, padding, normalized):
    """
    Creates first a complex cosine morlet wavelet in the time domain and then
    transforms it into the frequency domain.

    :param wav_time: Time vector of the wavelet must be centered around zero,
    so that the cosine peaks at time zero.
    :param freq: [int]
    :param n_cycles: [int]
    :param padding: [int] number of zeros to add to increase freq resolution
    :param normalized: [bool] Normalizes the wavelet to its maximum value, so
    that values oscillate between 0 and 1
    :return: Normalized amplitude coefficients as 1d array
    """

    cmw = time_domain_wavelet(wav_time, freq, n_cycles)
    cmwX = np.fft.fft(cmw, padding)
    if normalized:
        cmwX = cmwX / np.max(abs(cmwX))
    return cmwX


# The following function is for batch processing of multiple trials
def cmw_convolution_trials(
    sig,
    out_shape=(42, 100),
    frange=(1, 100),
    srate=1000,
    srange=(5, 15),
    bsl=(-3, -1),
    flogspace=True
):
    """
    Computes the time-frequency series of a given signal by convoluting the
    given signal with a complex morlet wavelet in the frequency domain
    following the convolution theorem.

    :param time_sig: [array] Time vector in seconds where time zero is the
    stimulation onset
    :param sig: [array] 2d array of time x trials
    :param srate: [float or int] Sampling rate of the given signal in Hz
    :param n_freq: [int] Number of frequencies you want to have
    :param frange: [tuple] Minimum and maximum frequency you want to have in Hz
    :param srange: [tuple] Number of cycles for the minimum and maximum freq
    :param bsl: [tuple] Limits of the baseline in seconds. Can be None.
    :param new_time_sig: [array] Optional, Shorter time vector in seconds where
    time zero is the stimulation onset.
    Used to cut the final signal after convolution.
    :param flogspace: [bool] Optional, space the frequencies as log
    :param average_trials: [bool] Optional, the result is the averaged trials.
    :param plot_result: [bool] Optional, plot the results.
    :return: [ndarray, tuple] Frequency-Time series in absolute power and in
    decibels as well as phase, together with a tuple containing
    frequencies and time.
    """

    assert len(sig.shape) <= 2, "The signal must have 2 or less dimensions"
    assert sig.shape[0] % out_shape[1] == 0, (
        f"Signal length ({sig.shape[0]}) not divisible "
        "by output shape {out_shape[1]}")

    # Set parameteres in its correct shape or order
    n_pts = sig.shape[0]
    n_freq = out_shape[0]
    downsampling_step = int(n_pts / out_shape[1])
    frange = np.sort(frange)
    srange = np.sort(srange)

    # define wavelet parameters
    wvlt_time = np.arange(-2, 2 + (1/srate), 1/srate)
    if flogspace:
        logf = [np.log10(f) for f in frange]
        logs = [np.log10(s) for s in srange]
        wvlt_frex = np.logspace(logf[0], logf[1], n_freq)
        wvlt_n_cycl = np.logspace(logs[0], logs[1], n_freq)
    else:
        wvlt_frex = np.arange(frange[0], frange[1], n_freq)
        wvlt_n_cycl = np.arange(srange[0], srange[1], n_freq)

    # define convolution parameters
    n_trials = sig.shape[1]
    n_wvlt = len(wvlt_time)

    tf = np.zeros([n_freq, out_shape[1], n_trials, 2])
    batch_size = 128
    n_batches = math.ceil(n_trials / batch_size)
    for idx_batch in range(n_batches):
        idx_start = idx_batch * batch_size
        idx_end = (idx_batch + 1) * batch_size
        idx_end = min(idx_end, n_trials)
        n_trials_batch = idx_end - idx_start

        n_data = n_pts * n_trials_batch
        n_convolution = n_wvlt + n_data - 1
        n_conv_pow2 = nextpow2(n_convolution)
        half_n_wvlt = n_wvlt//2

        print(f'Processing batch {idx_batch} / {n_batches}...')
        # compute Fourier coefficients of LFP data (no downsampling here!)
        sig_batch = sig[:, idx_start:idx_end]
        lfp_reshape = np.reshape(sig_batch, [n_pts*n_trials_batch], order='F')
        lfp_fft = np.fft.fft(lfp_reshape, n_conv_pow2)

        # initialize time-frequency output matrix

        for fi in range(n_freq):

            # create frequency-domain wavelet
            cmwX_norm = frequency_domain_wavelet(wvlt_time, wvlt_frex[fi],
                                                 wvlt_n_cycl[fi], n_conv_pow2,
                                                 True)

            # second and third steps of convolution using the
            # *convolution theorem* # Multiply amplitude of signal with
            # amplitude of wavelet, then IFFT
            a_s = np.fft.ifft(cmwX_norm * lfp_fft)

            # Cut out the padding of the signal
            a_s = a_s[:n_convolution]
            a_s = a_s[half_n_wvlt: -half_n_wvlt]

            # Reshape signal to time X trials
            a_s = np.reshape(a_s, [n_pts, n_trials_batch], order="F")

            # 1. Convert amplitude spectrum to power spectrum
            powts = abs(a_s) ** 2  # calculate power
            powts = powts.astype(np.float32)
            powts = powts[0::downsampling_step]

            # phase
            phase = np.angle(a_s)
            phase = np.real(phase).astype(np.float32)
            phase = phase[0::downsampling_step]

            tf[fi, :, idx_start:idx_end, 0] = powts  # Raw data, absolute power
            tf[fi, :, idx_start:idx_end, 1] = phase  # Phases of oscillations

    return tf
