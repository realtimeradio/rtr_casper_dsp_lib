import logging
import logging.handlers
import sys
import time
import redis
import json
import socket
import struct
import numpy as np


logger = logging.getLogger(__name__)
NOTIFY = logging.INFO + 1
logging.addLevelName(NOTIFY, "NOTIFY")

IS_INITIALIZED_ATTR = "_has_default_handlers"

'''
A Redis-based log handler from:
http://charlesleifer.com/blog/using-redis-pub-sub-and-irc-for-error-logging-with-python/
'''
class RedisHandler(logging.Handler):
    def __init__(self, channel, conn, *args, **kwargs):
        logging.Handler.__init__(self, *args, **kwargs)
        self.channel = channel
        self.redis_conn = conn

    def emit(self, record):
        attributes = [
            'name', 'msg', 'levelname', 'levelno', 'pathname', 'filename',
            'module', 'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
            'thread', 'threadName', 'process', 'processName',
        ]
        record_dict = dict((attr, getattr(record, attr)) for attr in attributes)
        record_dict['formatted'] = self.format(record)
        try:
            self.redis_conn.publish(self.channel, json.dumps(record_dict))
        except UnicodeDecodeError:
            self.redis_conn.publish(self.channel, 'UnicodeDecodeError on emit!')
        except redis.RedisError:
            pass

            
def log_notify(log, message=None):
    msg = message or "%s starting on %s" % (log.name, socket.gethostname())
    log.log(NOTIFY, msg)
            
    
def add_default_log_handlers(logger, redishostname=None, fglevel=logging.INFO, bglevel=NOTIFY):
    if getattr(logger, IS_INITIALIZED_ATTR, False):
        return logger
    setattr(logger, IS_INITIALIZED_ATTR, True)

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter('%(asctime)s - %(name)20s - %(levelname)s - %(message)s')

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(fglevel)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    syslog_handler = logging.handlers.SysLogHandler(address='/dev/log')
    syslog_handler.setLevel(bglevel)
    syslog_handler.setFormatter(formatter)
    logger.addHandler(syslog_handler)

    if redishostname is not None:
        redis_host = redis.StrictRedis(redishostname, socket_timeout=1)
        try:
            redis_host.ping()
        except redis.ConnectionError:
            logger.warn("Couldn't connect to redis server at %s"%redishostname)
            return logger

        redis_handler = RedisHandler('log-channel', redis_host)
        redis_handler.setLevel(bglevel)
        redis_handler.setFormatter(formatter)
        logger.addHandler(redis_handler)
    return logger

def get_casper_fft_descramble(n_bit_fft, n_bit_parallel):
    """
    Get the descramble map for a CASPER FFT with
    2**n_bit_fft channels, presenting 2**n_bit_parallel
    on each cycle
    """
    n_fft = 2**n_bit_fft
    n_parallel = 2**n_bit_parallel
    return np.arange(n_fft).reshape(n_fft // n_parallel, n_parallel).transpose().flatten()

def get_casper_fft_scramble(n_bit_fft, n_bit_parallel):
    """
    Get the scramble map for a CASPER FFT with
    2**n_bit_fft channels, presenting 2**n_bit_parallel
    on each cycle
    """
    descramble = get_casper_fft_descramble(n_bit_fft, n_bit_parallel)
    scramble = np.zeros_like(descramble)
    for i,j in enumerate(descramble):
        scramble[j] = i
    return scramble

def bit_reverse(x, n_bit):
    """
    Return the n_bit bit-reversed integer of x
    """
    return int(np.binary_repr(x, width=n_bit)[::-1], 2)
