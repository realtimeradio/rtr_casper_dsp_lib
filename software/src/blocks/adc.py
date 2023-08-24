import logging
import numpy as np
import struct
import socket
import time
from ..error_levels import *
from .block import Block

class adc(Block):
    """
    Instantiate a control interface for an ADC block.

    :param host: CasperFpga interface for host.
    :type host: casperfpga.CasperFpga

    :param name: Name of block in Simulink hierarchy.
    :type name: str

    :param n_input: Number of independent ADC channels
    :type n_input: int

    :param dtype: Numpy dtype of each sample. E.g. '>i2'
    :type dtype: str

    :param logger: Logger instance to which log messages should be emitted.
    :type logger: logging.Logger
    """
    def __init__(self, host, name, n_input, dtype, logger=None, **kwargs):
        super(adc, self).__init__(host, name, logger=logger, **kwargs)
        self.dtype = dtype
        self.n_input = n_input
        self.sample_bytes = np.dtype(dtype).itemsize

    def get_sync_count(self):
        """
        Get the number of sync events since FPGA programming.

        :return: count
        :rtype: int
        """
        return self.read_uint('sync_counter')

    def get_overflow_count(self):
        """
        Get the number of overflow events since FPGA programming.

        :return: count
        :rtype: int
        """
        return self.read_uint('overflow_counter')

    def get_invalid_count(self):
        """
        Get the number of invalid events since FPGA programming.

        :return: count
        :rtype: int
        """
        return self.read_uint('invalid_counter')

    def _trigger_snapshot(self):
        """
        Simultaneously trigger all ADC streams to record a snapshot of data.
        """
        self.write_int('snapshot_arm', 0)
        self.write_int('snapshot_trig', 0)
        self.write_int('snapshot_arm', 1)
        self.write_int('snapshot_trig', 1)
        self.write_int('snapshot_arm', 0)
        self.write_int('snapshot_trig', 0)

    def get_snapshot(self, trigger=True):
        """
        Read a snapshot of data from all ADC channels.
        Return time series for each channel.

        :param trigger: If True, trigger a new simultaneous capture of data from
            all channels. If False, read existing data capture. 
        :type trigger: bool

        :return: Array of captured data with dimensions
            [ADC_CHANNEL, TIME_SAMPLES].
        :rtype: numpy.array
        """
        if trigger:
            self._trigger_snapshot()
        d = []
        for i in range(self.n_input):
            ss_name = f'ss_{i}'
            try:
                ss = self.snapshots[ss_name]
            except KeyError:
                self._error(f'Could not find snapshot {ss_name}')
                raise
            raw = ss.read_raw(arm=False, timeout=0.1)[0]['data']
            d += [np.frombuffer(raw, dtype=self.dtype)]
        return np.array(d)

    def plot_snapshot(self, show=True):
        """
        Plot a new snapshot of data using matplotlib

        :param show: If True, call show() after plotting.
        :type show: bool

        :return: matplotlib figure instance
        :rtype: matploitlib.Figure
        """
        from matplotlib import pyplot as plt

        x = self.get_snapshot(trigger=True)
        fig = plt.figure()
        for i in range(self.n_input):
            plt.plot(x[i], label=i)
        plt.legend()
        if show:
            plt.show()
        return fig

    def plot_spectrum(self, db=False, acc_len=1, show=True):
        """
        Plot a power spectrum of the ADC input stream using a simple FFT.

        :param db: If True, plot in dBs, else linear.
        :type db: bool

        :param show: If True, call show() after plotting.
        :type show: bool

        :param acc_len: Number of snapshots to sum.
        :type acc_len: int

        :return: matplotlib figure instance
        :rtype: matploitlib.Figure
        """
        from matplotlib import pyplot as plt
        x = self.get_snapshot()
        X = np.abs(np.fft.fft(x, axis=1))**2
        if acc_len > 1:
            for i in range(acc_len-1):
                x = self.get_snapshot()
                X += np.abs(np.fft.fft(x, axis=1))**2
        if db:
            X = 10*np.log10(X)
        fig = plt.figure()
        for i in range(X.shape[0]):
            plt.plot(np.fft.fftshift(X[i]), label=i)
        plt.legend()
        plt.xlabel('FFT bin (DC-centered)')
        if db:
            plt.ylabel('Power (dB; Arbitrary Reference)')
        else:
            plt.ylabel('Power (Linear, Arbitrary Reference)')
        if show:
            plt.show()
        return fig
