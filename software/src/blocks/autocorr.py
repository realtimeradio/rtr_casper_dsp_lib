import time
import struct
import numpy as np
from scipy.signal import medfilt

from .block import Block
from ..helpers import get_casper_fft_descramble

class autocorr(Block):
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

    :param is_descrambled: If False, apply descrable map to channel ordering on read.
    :type is_descrambled: bool

    :ivar n_input_per_block: Number of signal streams handled by a
        single correlation core.
    """
    _dtype = '>f4'
    def __init__(self, host, name,
                 n_chan_serial_bits,
                 n_chan_parallel_bits,
                 n_input_serial_bits,
                 n_input,
                 init_acc_len=2**15,
                 is_descrambled=False,
                 logger=None,
                 **kwargs,
                ):
        super(autocorr, self).__init__(host, name, logger)
        self._init_acc_len = init_acc_len
        self._n_chan_serial_bits = n_chan_serial_bits
        self._n_chan_parallel_bits = n_chan_parallel_bits
        self._n_chan_bits = n_chan_parallel_bits + n_chan_serial_bits
        self.n_chan = 2**(self._n_chan_bits)
        self.n_input = n_input
        self._n_input_serial_bits = n_input_serial_bits
        self._acc_len = init_acc_len
        self._is_descrambled = is_descrambled == "on"
        self._descramble_order = get_casper_fft_descramble(self._n_chan_bits, self._n_chan_parallel_bits)

        self._n_cores = n_input // (2**self._n_input_serial_bits)
        self.n_input_per_block = n_input // self._n_cores

    def get_acc_cnt(self):
        """
        Get the current accumulation count.

        :return count: Current accumulation count
        :rtype count: int
        """
        return self.read_uint('acc_cnt')
   
    def _wait_for_acc(self):
        """
        Block until a new accumulation completes, then return
        the count index.

        :return count: Current accumulation count
        :rtype count: int
        """
        cnt0 = self.get_acc_cnt()
        cnt1 = self.get_acc_cnt()
        # Counter overflow protection
        if cnt1 < cnt0:
            cnt1 += 2**32
        while cnt1 < ((cnt0+1) % (2**32)):
            time.sleep(0.1)
            cnt1 = self.get_acc_cnt()
        return cnt1

    def _read_bram(self, offset=0, dtype=None):
        """ 
        Read RAM containing autocorrelation spectra for all signals in a core.

        :param offset: If non-zero, offset BRAM read IDs by this amount.
            Only useful if using this method in a subclassed block
            which adds extra brams with other functionality.
        :type offset: int

        :param dtype: If provided, interpret data using this data type rather than
            that of the ``_dtype`` attribute. Should be a numpy parseable type string,
            e.g. '>u4'.
        :type dtype: str

        :return: Array of autocorrelation data, in float32 format. Array
            dimensions are [INPUTS, FREQUENCY CHANNEL].
        :rtype: numpy.array
        """
        dtype = dtype or self._dtype
        dout = np.zeros([self.n_input, self.n_chan], dtype=dtype)
        read_loop_range = range(self._n_cores)
        n_words_per_stream = self.n_input_per_block * self.n_chan // (2**self._n_chan_parallel_bits)
        n_chan_per_stream = self.n_chan // (2**self._n_chan_parallel_bits)
        for core in read_loop_range:
            for stream in range(2**self._n_chan_parallel_bits):
                ram_id = stream + offset
                ramname = f'{core}_dout{ram_id}_bram'
                raw = self.read(ramname, 4*n_words_per_stream)
                x = np.frombuffer(raw, dtype)
                #x = struct.unpack(f'>{n_words_per_stream}f', raw)
                for subsignal in range(self.n_input_per_block):
                    dout[core*self.n_input_per_block + subsignal, stream::2**self._n_chan_parallel_bits] = \
                        x[subsignal*n_chan_per_stream:(subsignal+1)*n_chan_per_stream]
        if not self._is_descrambled:
            for i in range(self.n_input):
                dout[i] = dout[i][self._descramble_order]
        return dout

    def _arm_readout(self):
        """
        Arm readout buffers to capture the next valid accumlation.
        Once this occurs, the buffers will not be overwritten until
        another arm is issued.
        """
        self.write_int('trig', 0)
        self.write_int('trig', 1)
        self.write_int('trig', 0)
    
    def get_new_spectra(self, flush_vacc='auto', wait_on_new=True, filter_ksize=None):
        """
        Get a new average power spectra.

        :param flush_vacc: If ``True``, throw away a spectra before grabbing a valid
            one. This can be useful if the upstream analog settings may have changed
            during the last integration. If ``False``, return the first spectra
            available. If ``'auto'`` perform a flush if the input multiplexer has
            changed positions.
        :type flush_vacc: Bool or string

        :param wait_on_new: If True, arm and wait for a new accumulation before
            reading RAMs.
        :type wait_on_new: bool

        :param filter_ksize: If not None, apply a spectral median filter
            with this kernel size. The kernet size should be odd.
        :type filter_ksize: int

        :return: Float32 array of dimensions [POLARIZATION, FREQUENCY CHANNEL]
            containing autocorrelations with accumulation length divided out.
        :rtype: numpy.array

        """

        assert flush_vacc in [True, False, 'auto'], "Don't understand value of `flush_vacc`"
        assert filter_ksize is None or filter_ksize % 2 == 1, "Filter kernel size should be odd"

        auto_flush = False
        if flush_vacc == True or auto_flush:
            self._debug("Flushing accumulation")
            self._wait_for_acc()
        if wait_on_new:
            self._arm_readout()
            acc_cnt = self._wait_for_acc()
        spec = self._read_bram() / float(self.get_acc_len())
        nsignals, nchans = spec.shape
        if filter_ksize is not None:
            for signal in range(nsignals):
                spec[signal] = medfilt(spec[signal], kernel_size=filter_ksize)

        return spec

    def plot_spectra(self, db=True, show=True, filter_ksize=None, freqrange=None):
        """
        Plot the spectra of all signals in a single signal_block,
        with accumulation length divided out
        
        :param db: If True, plot 10log10(power). Else, plot linear.
        :type db: bool

        :param show: If True, call matplotlib's `show` after plotting
        :type show: bool

        :param filter_ksize: If not None, apply a spectral median filter
            with this kernel size. The kernet size should be odd.
        :type filter_ksize: int

        :param freqrange: If provided, use these frequencies for the xaxis of plots.
            Should be a list/array  containing [minimum_freq_hz, maximum_freq_hz].
            The axis points are generated with `numpy.linspace(freqrange[0], freqrange[1], ...)
        :type freqs: list

        :return: matplotlib.Figure

        """
        assert filter_ksize is None or filter_ksize % 2 == 1, "Filter kernel size should be odd"
        from matplotlib import pyplot as plt
        specs = self.get_new_spectra(filter_ksize=filter_ksize)
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

    def get_acc_len(self):
        """
        Get the currently loaded accumulation length in units of spectra.

        :return: Current accumulation length
        :rtype: int
        """
        self._acc_len = self.read_int('acc_len')
        return self._acc_len

    def set_acc_len(self, acc_len):
        """
        Set the number of spectra to accumulate.

        :param acc_len: Number of spectra to accumulate
        :type acc_len: int
        """
        assert isinstance(acc_len, int), "Cannot set accumulation length to type %r" % type(acc_len)
        self._acc_len = acc_len
        self.write_int('acc_len',acc_len)

    def get_status(self):
        """
        Get status and error flag dictionaries.

        Status keys:

            - acc_len (int) : Currently loaded accumulation length in number of spectra.

        :return: (status_dict, flags_dict) tuple. `status_dict` is a dictionary of
            status key-value pairs. flags_dict is
            a dictionary with all, or a sub-set, of the keys in `status_dict`. The values
            held in this dictionary are as defined in `error_levels.py` and indicate
            that values in the status dictionary are outside normal ranges.
        """
        stats = {
            'acc_len': self.get_acc_len(),
        }
        flags = {}
        return stats, flags

    def initialize(self, read_only=False):
        """
        Initialize the block, setting (or reading) the accumulation length.

        :param read_only: If False, set the accumulation length to the value provided
            when this block was instantiated. If True, use whatever accumulation length
            is currently loaded.
        :type read_only: bool
        """
        if read_only:
            self.get_acc_len()
        else:
            self.set_acc_len(self._init_acc_len)
