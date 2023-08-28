import numpy as np
import struct
from .block import Block
from ..error_levels import *

class tvg(Block):
    """
    Instantiate a control interface for a post-equalization test vector
    generator block.

    :param host: CasperFpga interface for host.
    :type host: casperfpga.CasperFpga

    :param name: Name of block in Simulink hierarchy.
    :type name: str

    :param logger: Logger instance to which log messages should be emitted.
    :type logger: logging.Logger

    :param n_input: Number of independent streams processed by this block
    :type n_input: int

    :param n_chan: Number of frequency channels.
    :type n_chan: int

    :param dtype: Numpy-style data type of test vector words. E.g. '>u4'
    :type dtype: str

    """
    _FORMAT = 'B'
    def __init__(self, host, name, n_input, n_chan, dtype, logger=None, **kwargs):
        super(tvg, self).__init__(host, name, logger=logger, **kwargs)
        self.n_input = n_input
        self.n_chan = n_chan
        self.dtype = dtype
        self._n_byte_per_word = np.dtype(dtype).itemsize
        # number of bytes in a single stream
        self._input_vector_size = n_chan * self._n_byte_per_word

    def tvg_enable(self):
        """
        Enable the test vector generator.
        """
        self.write_int('ctrl', 1)

    def tvg_disable(self):
        """
        Disable the test vector generator
        """
        self.write_int('ctrl', 0)

    def tvg_is_enabled(self):
        """
        Query the current test vector generator state.

        :return: True if the test vector generator is enabled, else False.
        :rtype: bool

        """
        return bool(self.read_int('ctrl'))
    
    def write_stream_tvg(self, stream, test_vector):
        """
        Write a test vector pattern to a single signal stream.
        
        :param stream: Index of input to which test vectors should be loaded.
        :type stream: int

        :param test_vector: `self.n_chan`-element test vector. Values should
            be representable in `self.dtype` format.
        :type test_vector: list or numpy.ndarray

        """
        tv = np.array(test_vector, dtype=self.dtype)
        assert (tv.shape[0] == self.n_chan), "Test vector should have self.n_chan elements!"
        core_name = f'0_{stream}'
        self.write(core_name, tv.tobytes())

    def write_const_per_stream(self):
        """
        Write a constant to all the channels of a stream,
        with stream `i` taking the value `i`.
        """
        for stream in range(self.n_input):
            self.write_stream_tvg(stream, np.ones(self.n_chan)*stream)

    def write_freq_ramp(self):
        """
        Write a frequency ramp to the test vector 
        that is repeated for all ADC inputs. Data are wrapped to fit into
        8 bits. I.e., the test vector value for channel 257 takes the value ``1``.
        """
        ramp = np.arange(self.n_chan)
        ramp = np.array(ramp, dtype=self.dtype)
        for stream in range(self.n_input):
            self.write_stream_tvg(stream, ramp)

    def read_stream_tvg(self, stream):
        """
        Read the test vector loaded to an ADC stream.
        
        :param stream: Index of input from which test vectors should be read.
        :type stream: int

        :param makecomplex: If True, return an array of 4+4 bit complex numbers,
           as interpretted by the correlator. If False, return the raw unsigned 8-bit
           values loaded in FPGA memory.
        :type makecomplex: Bool

        :return: Test vector array
        :rtype: numpy.ndarray

        """
        core_name = f'0_{stream}'
        s = self.read(core_name, self._input_vector_size)
        tvg = np.frombuffer(s, dtype=self.dtype)

        #if makecomplex:
        #    assert self._FORMAT == 'B', "Don't know how to make '%s' format values complex" % self._FORMAT
        #    tvg_r = tvg.view(dtype=np.int8) >> 4
        #    tvg_r[tvg_r > 7] -= 16
        #    tvg_i = tvg.view(dtype=np.int8) & 0xf
        #    tvg_i[tvg_i > 7] -= 16
        #    tvg = tvg_r + 1j*tvg_i

        return tvg

    def get_status(self):
        """
        Get status and error flag dictionaries.

        Status keys:

            - tvg_enabled: Currently state of test vector generator. ``True`` if
              the generator is enabled, else ``False``.

        :return: (status_dict, flags_dict) tuple. `status_dict` is a dictionary of
            status key-value pairs. flags_dict is
            a dictionary with all, or a sub-set, of the keys in `status_dict`. The values
            held in this dictionary are as defined in `error_levels.py` and indicate
            that values in the status dictionary are outside normal ranges.
        """
        stats = {}
        flags = {}
        stats['tvg_enabled'] = self.tvg_is_enabled()
        if stats['tvg_enabled']:
            flags['tvg_enabled'] = RTR_NOTIFY
        return stats, flags

    def initialize(self, read_only=False):
        """
        Initialize the block.

        :param read_only: If True, do nothing. If False, load frequency-ramp
            test vectors, but disable the test vector generator.
        :type read_only: bool

        """
        if read_only:
            pass
        else:
            self.tvg_disable()
            self.write_freq_ramp()
