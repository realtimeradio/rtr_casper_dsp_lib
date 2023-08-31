import struct
import time
import numpy as np
from .block import Block
from ..error_levels import *

class input(Block):
    """
    Instantiate a control interface for an Input block. This block
    allows switching data streams between constant-zeros, digital noise,
    and ADC streams.

    A statistics interface is also provided, providing bit statistics and
    histograms.

    :param host: CasperFpga interface for host.
    :type host: casperfpga.CasperFpga

    :param name: Name of block in Simulink hierarchy.
    :type name: str

    :param n_inputs: Number of independent signals
    :type n_inputs: int

    :param n_acc_len_bits: Number of samples accumulated (2^?)
    :type n_acc_len_bits: int

    :param n_bits: Number of bits per input ADC sample
    :type n_bits: int

    :ivar n_parallel: Number of parallel samples per signal.
    :ivar n_parallel: int

    :param logger: Logger instance to which log messages should be emitted.
    :type logger: logging.Logger

    """
    _USE_NOISE = 0
    _USE_ADC   = 1
    _USE_ZERO  = 2
    _USE_COUNTER = 3
    _INT_TO_POS = {}
    _INT_TO_POS[_USE_NOISE] = 'noise'
    _INT_TO_POS[_USE_ADC]   = 'adc'
    _INT_TO_POS[_USE_ZERO]  = 'zero'
    _INT_TO_POS[_USE_COUNTER] = 'counter'

    def __init__(self, host, name, n_bits, n_inputs, n_parallel, n_acc_len_bits, logger=None, **kwargs):
        super(input, self).__init__(host, name, logger, **kwargs)
        self.n_bits = n_bits
        self.n_inputs = n_inputs
        self.n_parallel = n_parallel
        self.n_acc_len_bits = n_acc_len_bits

    def get_switch_positions(self):
        """
        Get the positions of the input switches.

        :return: List of switch positions. Entry ``n`` contains the position
            of the switch associated with ADC input ``n``. Switch positions
            are "noise" (internal digital noise generators), "adc"
            (digitized ADC stream), or "zero" (constant 0).
        :rtype: list of str

        """
        pos = []
        inputs_left = self.n_inputs
        for regn in range((self.n_inputs + 15) // 16):
            reg_val = self.read_uint('source_sel%d' % regn)
            for i in range(min(inputs_left, 16)):
                # MSBs of control signals are for first input
                v = (reg_val >> (2*(15-i))) & 0b11
                pos += [self._INT_TO_POS[v]]
                inputs_left -= 1
        return pos
                

    def _switch(self, val, stream=None):
        """
        Set the switch position of a single input stream.
        
        :param val: mux select value desired.
        :type val: int

        :param stream: Which stream to switch. If None, switch all.
        :type stream: int or None

        """
        assert (val < 4), "Mux input value not recognized!"
        if stream is not None:
            assert (stream < self.n_inputs), "Can't switch stream >= self.n_inputs" 
            reg = 'source_sel%d' % (stream // 16) # one register per 16 streams
            reg_pos = 15 - (stream % 16) # First input controlled by MSBs
            self.change_reg_bits(reg, val, 2*reg_pos, 2)
        else:
            for stream in range(self.n_inputs):
                self._switch(val, stream)

    def use_noise(self, stream=None):
        """
        Switch input to internal noise source.

        :param stream: Which stream to switch. If None, switch all.
        :type stream: int or None

        """
        self._debug("Stream %s: switching to Noise" % stream)
        self._switch(self._USE_NOISE, stream)

    def use_adc(self, stream=None):
        """
        Switch input to ADC.

        :param stream: Which stream to switch. If None, switch all.
        :type stream: int or None

        """
        self._debug("Stream %s: switching to ADC" % stream)
        self._switch(self._USE_ADC, stream)

    def use_zero(self, stream=None):
        """
        Switch input to zeros.

        :param stream: Which stream to switch. If None, switch all.
        :type stream: int or None

        """
        self._debug("Stream %s: switching to Zeros" % stream)
        self._switch(self._USE_ZERO, stream)

    def use_counter(self, stream=None):
        """
        Switch input to counter.

        :param stream: Which stream to switch. If None, switch all.
        :type stream: int or None

        """
        self._debug("Stream %s: switching to Counter" % stream)
        self._switch(self._USE_COUNTER, stream)

    def get_bit_stats(self, combine_parallel=True):
        """
        Get the mean, RMS, and mean powers of all ADC streams.

        :param combine_parallel: If True, average together parallel streams for
            each input.
        :type combine_parallel: bool

        :return: (means, powers, rmss) tuple. Each member of the tuple is an
            array with ``self.n_inputs`` elements if ``combine_parallel`` is True.
            Else, each member has ``self.n_inputs * self.n_parallel`` elements,
            with parallel streams of a single input contiguous.
        :rval: (numpy.ndarray, numpy.ndarray, numpy.ndarray)

        """
        self.write_int('rms_enable', 1)
        time.sleep(0.01)
        self.write_int('rms_enable', 0)
        n_val = self.n_inputs * self.n_parallel
        x = np.frombuffer(self.read('rms_levels', n_val * 8), dtype='>u8')
        self.write_int('rms_enable', 1)
        # Lower bits are unsigned powers
        power_bits = (2*self.n_bits - 1) + self.n_acc_len_bits
        # MSBs are signed means
        mean_bits = 64 - power_bits
        means    = (x >> power_bits).astype(np.int64)
        means[means >= 2**(mean_bits-1)] -= 2**mean_bits
        means    = means.astype(float) / 2**self.n_acc_len_bits
        powers   = (x & (2**power_bits - 1)) / 2**self.n_acc_len_bits
        if combine_parallel:
            means = means.reshape(self.n_inputs, self.n_parallel).mean(axis=1)
            powers = powers.reshape(self.n_inputs, self.n_parallel).mean(axis=1)
        rmss = np.sqrt(powers)
        return means, powers, rmss

    def initialize(self, read_only=False):
        """
        Initialize the block.

        :param read_only: If True, do nothing. If False, set the input
            multiplexers to ADC data and enable statistic computation.
        :type read_only: bool

        """
        if read_only:
            pass
        else:
            self.use_adc()
            self.write_int('rms_enable', 1)

    def get_status(self):
        """
        Get status and error flag dictionaries.

        Status keys:

            - switch_position<n> (str) : Switch position ('noise', 'adc', 'zero' or 'counter')
              for input stream ``n``, where ``n`` is a two-digit integer starting at 00.
              Any input position other than 'adc' is flagged with "NOTIFY".

            - power<n> (float) : Mean power of input stream ``n``, where ``n`` is a
              two-digit integer starting at 00. In units of (ADC LSBs)**2.

            - rms<n> (float) : RMS of input stream ``n``, where ``n`` is a
              two-digit integer starting at 00. In units of ADC LSBs. Value is
              flagged as a warning if it is >30 or <5.

            - mean<n> (float) : Mean sample value of input stream ``n``, where ``n`` is a
              two-digit integer starting at 00. In units of ADC LSBs. Value
              is flagged as a warning if it is > 2.

        :return: (status_dict, flags_dict) tuple. `status_dict` is a dictionary of
            status key-value pairs. flags_dict is
            a dictionary with all, or a sub-set, of the keys in `status_dict`. The values
            held in this dictionary are as defined in `error_levels.py` and indicate
            that values in the status dictionary are outside normal ranges.

        """
        stats = {}
        flags = {}
        switch_positions = self.get_switch_positions()
        mean, power, rms = self.get_bit_stats()
        for i in range(self.n_inputs):
            stats['switch_position%.2d' % i] = switch_positions[i]
            if switch_positions[i] != 'adc':
                flags['switch_position%.2d' % i] = RTR_NOTIFY
            stats['power%.2d' % i] = power[i]
            stats['rms%.2d' % i]   = rms[i]
            stats['mean%.2d' % i]  = mean[i]
            if rms[i] > 30 or rms[i] < 5:
                flags['rms%.2d' % i] = RTR_WARNING
            if np.abs(mean[i]) > 2:
                flags['mean%.2d' % i] = RTR_WARNING
        return stats, flags

    #def _set_histogram_input(self, stream):
    #    """
    #    Set input of histogram block computation core.

    #    :param stream: Stream number to select.
    #    :type stream: int
    #    """
    #    assert (stream < self.n_inputs), "Can't switch stream >= self.n_inputs" 
    #    self.write_int('bit_stats_input_sel', stream)

    #def get_histogram(self, stream, sum_cores=True):
    #    """
    #    Get a histogram for an ADC stream.
    #    
    #    :param stream: ADC stream from which to get data.
    #    :type stream: int

    #    :param sum_cores: If True, compute one histogram from both pairs of
    #        interleaved ADC cores associated with an analog input.
    #        If False, compute separate histograms.
    #    :type sum_cores: bool

    #    :return: If ``sum_cores`` is True, return ``(vals, hist)``, where ``vals``
    #        is a list of histogram bin centers, and ``hist`` is a list of
    #        histogram data points. If ``sum_cores`` is False, return
    #        ``(vals, hist_a, hist_b)``, where ``hist_a`` and ``hist_b``
    #        are separate histogram data sets for the even-sample and odd-sample
    #        ADC cores, respectively.
    #    """
    #    self._info("Getting histogram for stream %d" % stream)
    #    self._set_histogram_input(stream)
    #    time.sleep(0.1)
    #    v = np.array(struct.unpack('>%dH' % (2*2**self.n_bits), self.read('bit_stats_histogram_output', 2*2*2**self.n_bits)))
    #    a = v[0:2**self.n_bits]
    #    b = v[2**self.n_bits : 2*2**self.n_bits]
    #    a = np.roll(a, 2**(self.n_bits - 1)) # roll so that array counts -128, -127, ..., 0, ..., 126, 127
    #    b = np.roll(b, 2**(self.n_bits - 1)) # roll so that array counts -128, -127, ..., 0, ..., 126, 127
    #    vals = np.arange(-2**(self.n_bits - 1), 2**(self.n_bits - 1))
    #    if sum_cores:
    #        return vals.tolist(), (a+b).tolist()
    #    else:
    #        return vals.tolist(), a.tolist(), b.tolist()

    #def get_all_histograms(self):
    #    """
    #    Get histograms for all signals, summing over all interleaving cores.

    #    :return: (vals, hists). ``vals`` is a list of histogram bin centers.
    #        ``hists`` is an ``[n_stream x 2**n_bits]`` list of histogram
    #        data.
    #    """
    #    out = np.zeros([self.n_inputs, 2**self.n_bits])
    #    for stream in range(self.n_inputs):
    #        x, out[stream,:] = self.get_histogram(stream, sum_cores=True)
    #    return x, out.tolist()

    #def print_histograms(self):
    #    """
    #    Print histogram stats to screen.
    #    """
    #    x, hist = self.get_all_histograms()
    #    hist = np.array(hist)
    #    hist /= 1024.*1024
    #    for vn, v in enumerate(x):
    #        print('%5d:'%v, end=' ')
    #        for an, ant in enumerate(hist):
    #            print('%.3f'%ant[vn], end=' ')
    #        print()

    #def plot_histogram(self, stream):
    #    """
    #    Plot a histogram.

    #    :param stream: ADC stream from which to get data.
    #    :type stream: int
    #    """
    #    
    #    from matplotlib import pyplot as plt
    #    bins, d = self.get_histogram(stream)
    #    plt.bar(np.array(bins)-0.5, d, width=1)
    #    plt.xlim((np.min(bins)*1.1, np.max(bins)*1.1))
    #    plt.show()
