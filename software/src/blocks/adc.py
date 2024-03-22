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

    :param logger: Logger instance to which log messages should be emitted.
    :type logger: logging.Logger
    """
    def __init__(self, host, name, logger=None, **kwargs):
        super(adc, self).__init__(host, name, logger=logger, **kwargs)

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

