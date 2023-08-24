import logging
import socket
import inspect
import json
import numpy as np
import struct
import time
import datetime
import os
import yaml
from . import helpers
from . import __version__
from .blocks import *
from .error_levels import *

def build_block(info, cfpga, logger):
    """
    Given an info dictionary, return an appropriate
    Block instance.

    :param info: Info dictionary
    :type info: dict

    :return: Name, Block instance
    """
    assert info.pop('library', 'not_rtr') == 'rtr'
    block_type = info.pop('block_type', '')
    assert block_type.startswith('rtr:')
    block_type = block_type.split(':', maxsplit=1)[1]
    block_type = block_type.replace('-', '_') #TODO: just don't use hyphens in the simulink tags
    block_name = info.pop('block_name', '')
    # strip off model name
    block_name = block_name.split('/', maxsplit=1)[1]
    block_name = block_name.replace('/', '_')
    try:
        blkclass = globals()[block_type]
        logger.info(f'Found class {block_type}')
    except KeyError:
        logger.error(f'No class {block_type} available')
        return None, None
    block = blkclass(cfpga, block_name, logger=logger, **info)
    return block_name, block


class RtrDspSystem():
    """
    A control class for systems built from RTR firmware modules.

    :param cfpga: CasperFpga client instance
    :type cfpga: CasperFpga

    :param fpgfile: The fpgfile the FPGA is (or should be) programmed with.
    :type fpgfile: str

    :param logger: Logger instance to which log messages should be emitted.
    :type logger: logging.Logger

    """
    def __init__(self, cfpga, fpgfile, logger=None):
        self.fpgfile = fpgfile
        #: Underlying CasperFpga control instance
        self._cfpga = cfpga
        self.hostname = cfpga.host #: hostname of host FPGA board
        #: Python Logger instance
        self.logger = logger or helpers.add_default_log_handlers(logging.getLogger(f'{__name__}:{self.hostname}'))

        self.blocks = {}
        try:
            self._scrape_blocks_from_fpg()
        except:
            self.logger.exception("Failed to scrape firmware blocks. "
                                  "Maybe the board needs programming.")

    def _scrape_blocks_from_fpg(self, fpgfile=None):
        # If fpgfile is provided, use it
        if fpgfile is not None:
            self.logger.info(f'Updating fpgfile to {fpgfile}')
            self.fpgfile = fpgfile
        fpgfile = self.fpgfile

        # Rudimentary existence check
        if not os.path.exists(fpgfile):
            self.logger.error(f'fpgfile path {fpgfile} does not exist')
            raise ValueError

        # Try to parse fpg header with casperfpga
        try:
            self._cfpga.get_system_information(self.fpgfile)
        except:
            self.logger.error(f'Failed to read and decode {fpgfile}')
            raise RuntimeError

        # Scrape blocks based on info blocks.
        # Always add a system block
        blocks = {'system': system(self._cfpga, self.logger)}
        for devname, device in self._cfpga.devices.items():
            if not isinstance(device, dict): continue
            tag = device.get('tag', '')
            if not tag  == 'casper:info': continue
            try:
                info = json.loads(device['info'])
            except KeyError:
                # Shouldn't happen unless simulink info block changes
                self.logger.debug(f'Skipping decoding f{devname} because it had no info field')
                continue
            except json.JsonDecodeError:
                # This might be normal if the block didn't use RTR formatting
                continue
            if not info.get('library', 'not_rtr') == 'rtr': continue
            # If we made it to here, the block should be an RTR-supported one
            self.logger.debug(f'Creating block from {info}')
            block_name, block = build_block(info, self._cfpga, self.logger)
            if block_name is None:
                self.logger.warning('Failed to create a block')
                continue
            self.logger.info(f'Created block {block_name}')
            blocks[block_name] = block
        self.blocks = blocks
        for bn, block in self.blocks.items():
            self.__setattr__(bn, block)

    def is_connected(self):
        """
        :return: True if there is a working connection to an FPGA board. False otherwise.
        :rtype: bool
        """
        return self._cfpga.is_connected()

    def initialize(self, read_only=True):
        """
        Call the ```initialize`` methods of all underlying blocks, then
        optionally issue a software global reset.

        :param read_only: If True, call the underlying initialization methods
            in a read_only manner, and skip software reset.
        :type read_only: bool
        """
        for blockname, block in self.blocks.items():
            if read_only:
                self.logger.info("Initializing block (read only): %s" % blockname)
            else:
                self.logger.info("Initializing block (writable): %s" % blockname)
            block.initialize(read_only=read_only)
        if not read_only:
            self.logger.info("Performing software global reset")
            self.sync.arm_sync()
            self.sync.sw_sync()

    def get_status_all(self):
        """
        Call the ``get_status`` methods of all blocks in ``self.blocks``.
        If the FPGA is not programmed with F-engine firmware, will only
        return basic FPGA status.

        :return: (status_dict, flags_dict) tuple.
            Each is a dictionary, keyed by the names of the blocks in
            ``self.blocks``. These dictionaries contain, respectively, the
            status and flags returned by the ``get_status`` calls of
            each of this F-Engine's blocks.
        """
        stats = {}
        flags = {}
        #if not self.blocks['fpga'].is_programmed():
        #    stats['fpga'], flags['fpga'] = self.blocks['fpga'].get_status()
        #else:
        for blockname, block in self.blocks.items():
            try:
                stats[blockname], flags[blockname] = block.get_status()
            except:
                self.logger.info("Failed to poll stats from block %s" % blockname)
        return stats, flags

    def print_status_all(self, use_color=True, ignore_ok=False):
        """
        Print the status returned by ``get_status`` for all blocks in the system.
        If the FPGA is not programmed with F-engine firmware, will only
        print basic FPGA status.

        :param use_color: If True, highlight values with colors based on
            error codes.
        :type use_color: bool

        :param ignore_ok: If True, only print status values which are outside the
           normal range.
        :type ignore_ok: bool

        """
        #if not self.blocks['fpga'].is_programmed():
        #    print('FPGA stats (not programmed with F-engine image):')
        #    self.blocks['fpga'].print_status()
        #else:
        for blockname, block in self.blocks.items():
            print('Block %s stats:' % blockname)
            block.print_status(use_color=use_color, ignore_ok=ignore_ok)

    def deprogram(self):
        """
        Reprogram the FPGA into its default boot image.
        """
        self._cfpga.transport.progdev(0)

    def program(self, fpgfile=None):
        """
        Program an .fpg file to an FPGA. 

        :param fpgfile: The .fpg file to be loaded. Should be a path to a
            valid .fpg file. If None is given, the image currently in flash
            will be loaded.
        :type fpgfile: str

        """
        fpgfile = fpgfile or self.fpgfile

        if not isinstance(fpgfile, str):
            raise TypeError("wrong type for fpgfile")

        # Resolve symlinks
        fpgfile = os.path.realpath(fpgfile)

        if fpgfile and not os.path.exists(fpgfile):
            raise RuntimeError("Path %s doesn't exist" % fpgfile)

        self._cfpga.upload_to_ram_and_program(fpgfile)
        self._scrape_blocks_from_fpg(fpgfile)
