"""
Copyright 2015, Ilya Kuzovkin
Copyright 2021-2022, Buran Consulting, LLC

Licensed under MIT

Builds on example code by Jack Keegan
https://batchloaf.wordpress.com/2014/01/17/real-time-analysis-of-data-from-biosemi-activetwo-via-tcpip-using-python/
"""
import logging
log = logging.getLogger(__name__)

import socket
import numpy as np


SPEED_MODE = {
    0: 2048,
    1: 4096,
    2: 8192,
    3: 16384,
    4: 2048,
    5: 4096,
    6: 8192,
    7: 16384,
}


def is_set(x, bit):
    # Parentheses are not needed because of operator precedence rules, but help
    # make the code more readable.
    return (x & (1 << bit)) != 0


def decode_trigger(x):
    '''
    Details
    -------
    Bit 00 (LSB) through 15:
        Trigger input 1 through 16. Note that function keys F1 through F6 can embed
        trigger bits in bits 8-15 if desired.
    Bit 16 High when new Epoch is started
    Bit 17 Speed bit 0
    Bit 18 Speed bit 1
    Bit 19 Speed bit 2
    Bit 20 High when CMS is within range
    Bit 21 Speed bit 3
    Bit 22 High when battery is low
    Bit 23 (MSB) High if ActiveTwo MK2
    '''
    speed_mode = \
        (int(is_set(x, 17)) << 0) + \
        (int(is_set(x, 18)) << 1) + \
        (int(is_set(x, 19)) << 2) + \
        (int(is_set(x, 21)) << 3)

    trigger = 0b1111111111111111

    return {
        'trigger': x & trigger,
        'cms_in_range': is_set(x, 20),
        'low_battery': is_set(x, 22),
        'ActiveMK2': is_set(x, 23),
        'speed_mode': speed_mode,
        'new_epoch': is_set(x, 16),
        'fs': SPEED_MODE[speed_mode],
    }


class ActiveTwoClient:
    """
    Client for communicating with Biosemi ActiveTwo
    """

    #: Host where ActiView acquisition software is running
    #: This is the port ActiView listens on
    #: Number of channles
    #: Data packet size (default: 32 channels @ 512Hz)

    def __init__(self, host='127.0.0.1', port=8888, eeg_channels=32,
                 ex_included=False, sensors_included=False,
                 jazz_included=False, aib_included=False,
                 trigger_included=False, socket_timeout=0.25,
                 fs=512):
        """
        Initialize connection and parameters of the signal

        Parameters
        ----------
        host : string
            IP address of ActiView server
        port : int
            Port number ActiView server is listening on
        eeg_channels : float
            Number of EEG channels included
        """
        self.__dict__.update(locals())

        # Calculate number of TCP samples in array.
        if not (256 <= fs <= 16384):
            raise ValueError('Invalid sampling rate supplied')
        decimation_factor = 16384 / fs
        if int(decimation_factor) != decimation_factor:
            raise ValueError('Invalid sampling rate supplied')
        self.tcp_samples = int(128 / decimation_factor)

        # Build a mapping of channel type to a Numpy slice that can be used to
        # segment the data that we read in. I use a little trick to enable
        # n_channel to track the "offset" as we build the slices. At the end,
        # n_channels will tell us how many channels are being read in.
        slices = {}
        n_channels = 0
        if eeg_channels != 0:
            slices['eeg'] = np.s_[n_channels:eeg_channels]
            n_channels += eeg_channels
        if ex_included:
            slices['ex'] = np.s_[n_channels:n_channels+8]
            n_channels += 8
        if sensors_included:
            slices['sensors'] = np.s_[n_channels:n_channels+7]
            n_channels += 7
        if jazz_included:
            slices['jazz'] = np.s_[n_channels:n_channels+9]
            n_channels += 9
        if aib_included:
            slices['aib'] = np.s_[n_channels:n_channels+32]
            n_channels += 32
        if trigger_included:
            slices['trigger'] = np.s_[-1]
            n_channels += 1

        self.slices = slices
        self.n_channels = n_channels
        self.buffer_size = self.n_channels * self.tcp_samples * 3
        m = 'ActiveTwoClient configured with %d channels at %f Hz'
        log.info(m, self.n_channels, self.fs)
        log.info('Expecting %d samples/chan', self.tcp_samples)

    def _read(self, samples):
        signal_buffer = np.zeros((self.n_channels, self.tcp_samples), dtype='int32')
        data = self.sock.recv(self.buffer_size)

        for m in range(self.tcp_samples):
            # extract samples for each channel
            for channel in range(self.n_channels):
                offset = m * 3 * self.n_channels + (channel * 3)

                # The 3 bytes of each sample arrive in reverse order
                sample = \
                    (data[offset+2] << 16) + \
                    (data[offset+1] << 8) + \
                    (data[offset])

                # Store sample to signal buffer
                signal_buffer[channel, m] = sample

        return signal_buffer

    def read(self, duration):
        """
        Read signal from EEG

        Parameters
        ----------
        duration : float
            Duration, in seconds, to read. If duration is too long, then it
            seems the ActiView client will disconnect.

        Returns
        -------
        signal : 2D array (channel x time)
            Signal.
        """
        total_samples = int(round(duration * self.fs))

        # The reader process will run until requested amount of data is collected
        samples = 0
        data = []
        while samples < total_samples:
            try:
                data.append(self._read(samples))
                samples += self.tcp_samples
            except Exception as e:
                break

        if data:
            data = np.concatenate(data, axis=-1)
        else:
            data = np.empty((self.n_channels, 0), dtype='int32')
        return {k: data[s] for k, s in self.slices.items()}

    def connect(self):
        # Open connection. Be sure to set a timeout to make sure that the
        # program does not become unresponsive (on Windows even Ctrl+C can't
        # break a socket that's hung waiting for data).
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        self.sock.settimeout(self.socket_timeout)

    def disconnect(self):
        # Important! Be sure this is called to properly shut down sockets.
        self.sock.shutdown(socket.SHUT_RDWR)
        self.sock.close()
