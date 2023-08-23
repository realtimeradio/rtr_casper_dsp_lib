import socket
import time
import datetime

from .block import Block
from ..error_levels import *

import casperfpga.sysmon

class system(Block):
    """
    Instantiate a control interface for top-level FPGA control.

    :param host: CasperFpga interface for host.
    :type host: casperfpga.CasperFpga

    :param name: Name of block in Simulink hierarchy.
    :type name: str

    :param logger: Logger instance to which log messages should be emitted.
    :type logger: logging.Logger

    """
    def __init__(self, host, logger=None, **kwargs):
        # Top-level F-engine sees all registers
        super(system, self).__init__(host, 'sys', logger, **kwargs)
        self.sysmon = casperfpga.sysmon.Sysmon(self.host)

    def get_fpga_clock(self):
        """
        Estimate the FPGA clock, by polling the ``sys_clkcounter`` register.
        
        :return: Estimated FPGA clock in MHz
        :rtype: float

        """
        c0 = self.read_uint('clkcounter')
        t0 = time.time()
        time.sleep(0.1)
        c1 = self.read_uint('clkcounter')
        t1 = time.time()
        # Catch counter wrap
        if c1 < c0:
            c1 += 2**32
        clk_mhz = (c1 - c0) / 1e6 / (t1 - t0)
        return clk_mhz

    def is_programmed(self):
        """
        Lazy check to see if a board is programmed.
        Check for the "version_version" register. If it exists, the board is deemed programmed.
        
        :return: True if programmed, False otherwise.
        :rtype: bool

        """
        return 'version_version' in self.host.listdev()

    def get_status(self):
        """
        Get status and error flag dictionaries.

        Status keys:

            - programmed (bool) : ``True`` if FPGA appears to be running DSP
              firmware. ``False`` otherwise, and flagged as a warning.

            - fpga_clk_mhz (float) : The estimated FPGA clock rate in MHz. This
                is the same as the estimated ADC sampling rate.
              Flagged with an error if not between 190 and 200 MHz

            - host (str) : The host name of this board.

            - sys_mon (str) : ``'reporting'`` if the current firmware has a
              functioning system monitor module. Otherwise ``'not reporting'``,
              flagged as an error.

            - temp (float) : FPGA junction temperature, in degrees C. (Only
              reported is system monitor is available). Flagged as a warning
              if outside the recommended operating conditions. Flagged as an
              error if outside the absolute maximum ratings. See DS892.

            - vccaux (float) : Voltage of the VCCAUX FPGA power rail. (Only
              reported is system monitor is available). Flagged as a warning
              if outside the recommended operating conditions. Flagged as an
              error if outside the absolute maximum ratings. See DS892.

            - vccbram (float) : Voltage of the VCCBRAM FPGA power rail. (Only
              reported is system monitor is available). Flagged as a warning
              if outside the recommended operating conditions. Flagged as an
              error if outside the absolute maximum ratings. See DS892.

            - vccint (float) : Voltage of the VCCINT FPGA power rail. (Only
              reported is system monitor is available). Flagged as a warning
              if outside the recommended operating conditions. Flagged as an
              error if outside the absolute maximum ratings. See DS892.


        :return: (status_dict, flags_dict) tuple. `status_dict` is a dictionary of
            status key-value pairs. flags_dict is
            a dictionary with all, or a sub-set, of the keys in `status_dict`. The values
            held in this dictionary are as defined in `error_levels.py` and indicate
            that values in the status dictionary are outside normal ranges.
        """
        stats = {}
        flags = {}
        stats['programmed'] = self.is_programmed()
        stats['host'] = self.host.host
        fpga_clk_mhz = self.get_fpga_clock()
        stats['fpga_clk_mhz'] = fpga_clk_mhz
        if fpga_clk_mhz > 800. or fpga_clk_mhz < 10.:
            flags['fpga_clk_mhz'] = RTR_ERROR
        try:
            stats.update(self.sysmon.get_all_sensors())
            stats['sys_mon'] = 'reporting'
            flags['sys_mon'] = RTR_OK
        except:
            stats['sys_mon'] = 'not reporting'
            flags['sys_mon'] = RTR_WARNING
        if not stats['programmed']:
            flags['programmed'] = RTR_ERROR
        if 'vccaux' in stats:
            if stats['vccaux'] < 1.746 or stats['vccaux'] > 1.854:
                flags['vccaux'] = RTR_WARNING
            if stats['vccaux'] < -0.5 or stats['vccaux'] > 2.0:
                flags['vccaux'] = RTR_ERROR
        if 'vccbram' in stats:
            if stats['vccbram'] < 0.922 or stats['vccbram'] > 0.979:
                flags['vccbram'] = RTR_WARNING
            if stats['vccbram'] < -0.5 or stats['vccbram'] > 1.1:
                flags['vccbram'] = RTR_ERROR
        if 'vccint' in stats:
            if stats['vccint'] < 0.922 or stats['vccint'] > 0.979:
                flags['vccint'] = RTR_WARNING
            if stats['vccint'] < -0.5 or stats['vccint'] > 1.1:
                flags['vccint'] = RTR_ERROR
        if 'temp' in stats:
            if stats['temp'] < 0 or stats['temp'] > 85:
                flags['temp'] = RTR_WARNING
            if stats['temp'] > 125:
                flags['temp'] = RTR_ERROR
        return stats, flags
