import time
import struct
import numpy as np
from scipy.signal import medfilt

from .autocorr import autocorr

class autocorr_maxhold(autocorr):
    """
    Instantiate a control interface for an Auto-Correlation block. This
    provides auto-correlation spectra of post-FFT data.

    :param host: CasperFpga interface for host.
    :type host: casperfpga.CasperFpga

    :param name: Name of block in Simulink hierarchy.
    :type name: str

    :param logger: Logger instance to which log messages should be emitted.
    :type logger: logging.Logger

    :param init_acc_len: Accumulation length initialization value, in spectra.
    :type init_acc_len: int

    :param n_chan_serial_bits: Number of frequency channels processed serially (2^?)
    :type n_chan_serial_bits: int

    :param n_chan_parallel_bits: Number of frequency channels processed in parallel (2^?)
    :type n_chan_parallel_bits: int

    :param n_input_serial_bits: Number of individual signals processed serially (2^?)
    :type n_input_serial_bits: int

    :param n_input: Total number of individual signals processed
    :type n_input: int

    :param n_input_serial_bits: Number of inputs processed serially (2^?)
    :type n_input_serial_bits: int

    :ivar n_input_per_block: Number of signal streams handled by a
        single correlation core.
    """
    _bin_pt = 0
    _maxhold_dtype = '>f4'
    def get_new_maxhold(self, flush_vacc='auto', wait_on_new=True):
        """
        Get a new max hold power spectra.

        :param flush_vacc: If ``True``, throw away a spectra before grabbing a valid
            one. This can be useful if the upstream analog settings may have changed
            during the last integration. If ``False``, return the first spectra
            available. If ``'auto'`` perform a flush if the input multiplexer has
            changed positions.
        :type flush_vacc: Bool or string

        :param wait_on_new: If True, arm and wait for a new accumulation before
            reading RAMs.
        :type wait_on_new: bool

        :return: Float32 array of dimensions [INPUT, FREQUENCY CHANNEL]
            containing max hold data.
        :rtype: numpy.array

        """

        assert flush_vacc in [True, False, 'auto'], "Don't understand value of `flush_vacc`"
        auto_flush = False
        if flush_vacc == True or auto_flush:
            self._debug("Flushing accumulation")
            self._wait_for_acc()
        if wait_on_new:
            self._arm_readout()
            acc_cnt = self._wait_for_acc()
        spec = self._read_bram(offset=2**self._n_chan_parallel_bits,
                dtype = self._maxhold_dtype) / (2**self._bin_pt)
        nsignals, nchans = spec.shape
        return spec

    def get_new_spectra_and_maxhold(self, flush_vacc='auto',
            wait_on_new=True, filter_ksize=None):
        """
        Get a new average power spectra and maxhold, from the same accumulation.

        :param flush_vacc: If ``True``, throw away a spectra before grabbing a valid
            one. This can be useful if the upstream analog settings may have changed
            during the last integration. If ``False``, return the first spectra
            available. If ``'auto'`` perform a flush if the input multiplexer has
            changed positions.
        :type flush_vacc: Bool or string

        :param wait_on_new: If True, arm and wait for a new accumulation before
            reading RAMs.
        :type wait_on_new: bool

        :param filter_ksize: If not None, apply a spectral median filter to the
            average power spectrum with this kernel size. This size should be odd.
        :type filter_ksize: int

        :return: average_spectra, maxhold_spectra.
            Each is a float32 array of dimensions [INPUT, FREQUENCY CHANNEL].
            average spectra has the accumulation length divided out.
        :rtype: numpy.array, numpy.array

        """
        spec = self.get_new_spectra(flush_vacc=flush_vacc, wait_on_new=wait_on_new,
                filter_ksize=filter_ksize)
        maxhold = self.get_new_spectra(flush_vacc=False, wait_on_new=False)

        return spec, maxhold


    def plot_maxhold(self, db=True, show=True, freqrange=None):
        """
        Plot the maxhold spectra of all inputs.
        
        :param db: If True, plot 10log10(power). Else, plot linear.
        :type db: bool

        :param show: If True, call matplotlib's `show` after plotting
        :type show: bool

        :param freqrange: If provided, use these frequencies for the xaxis of plots.
            Should be a list/array  containing [minimum_freq_hz, maximum_freq_hz].
            The axis points are generated with `numpy.linspace(freqrange[0], freqrange[1], ...)
        :type freqs: list

        :return: matplotlib.Figure

        """
        from matplotlib import pyplot as plt
        specs = self.get_new_maxhold()
        if freqrange is not None:
            x = np.linspace(freqrange[0] / 1e6, freqrange[1] / 1e6, len(specs[0]))
            xlabel = 'Frequency [MHz]'
        else:
            x = np.arange(len(specs[0]))
            xlabel = 'Frequency [Channel Number]'
        f, ax = plt.subplots(1,1)
        if db:
            ax.set_ylabel('Power [dB]')
            specs = 10*np.log10(specs)
        else:
            ax.set_ylabel('Power [linear]')
        ax.set_xlabel(xlabel)
        for speci, spec in enumerate(specs):
            ax.plot(x, spec, label="signal_%d" % (speci))
        ax.legend()
        if show:
            plt.show()
        return f

