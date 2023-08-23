import socket
import time
import datetime

from .block import Block
from ..error_levels import *
from .. import __version__

class version(Block):
    """
    Instantiate a control interface for the firmware version block.

    :param host: CasperFpga interface for host.
    :type host: casperfpga.CasperFpga

    :param name: Name of block in Simulink hierarchy.
    :type name: str

    :param logger: Logger instance to which log messages should be emitted.
    :type logger: logging.Logger

    """
    def __init__(self, host, name, logger=None, **kwargs):
        # Top-level F-engine sees all registers
        super(version, self).__init__(host, name, logger, **kwargs)

    def get_firmware_version(self):
        """
        Read the firmware version register and return the contents as a string.

        :return version: major_version.minor_version.revision.bugfix
        :rtype str:
        """
        v = self.read_uint('version')
        major  = (v >> 24) & 0xff
        minor  = (v >> 16) & 0xff
        rev    = (v >>  8) & 0xff
        bugfix = (v >>  0) & 0xff
        return "%d.%d.%d.%d" % (major, minor, rev, bugfix)

    def get_build_time(self):
        """
        Read the UNIX time at which the current firmware was built.

        :return build_time: Seconds since the UNIX epoch at which the running
            firmware was built.

        :rtype int:
        """
        t = self.read_uint('timestamp')
        return t

    def get_status(self):
        """
        Get status and error flag dictionaries.

        Status keys:

            - timestamp (str) : The current time, as an ISO format string.

            - sw_version (str) : The version string of the control software
              package. Flagged as warning if the version indicates a build
              against a dirty git repository.

            - fw_version (str): The version string of the currently running
              firmware. Available only if the board is programmed.

            - fw_build_time (int): The build time of the firmware,
              as an ISO format string. Available only if the board 
              is programmed.

        :return: (status_dict, flags_dict) tuple. `status_dict` is a dictionary of
            status key-value pairs. flags_dict is
            a dictionary with all, or a sub-set, of the keys in `status_dict`. The values
            held in this dictionary are as defined in `error_levels.py` and indicate
            that values in the status dictionary are outside normal ranges.
        """
        stats = {}
        flags = {}
        stats['timestamp'] = datetime.datetime.now().isoformat()
        stats['sw_version'] = __version__
        stats['fw_version'] = self.get_firmware_version()
        stats['fw_build_time'] = datetime.datetime.fromtimestamp(self.get_build_time()).isoformat()
        if stats['sw_version'].endswith('dirty'):
            flags['sw_version'] = RTR_WARNING
        return stats, flags
