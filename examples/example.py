import time

from pyactivetwo import ActiveTwoClient
import matplotlib.pyplot as plt


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
    }


def is_set(x, bit):
    # Parentheses are not needed because of operator precedence rules, but help
    # make the code more readable.
    return (x & (1 << bit)) != 0


def test_read():
    # initialize the device
    print('initialized')
    device = ActiveTwoClient(host='127.0.0.1', port=8888,
            eeg_channels=8, sensors_included=True, trigger_included=True,
            tcp_samples=4, fs=16384)

    print('starting')
    for i in range(5):
        t = time.time()
        print('reading')
        rawdata = device.read(duration=1)
        td = time.time() - t
        print(f'{rawdata.shape} samples in {td} sec')
        #time.sleep(1)
        print(decode_trigger(rawdata[-1, 0]))

    t = time.time()
    rawdata = device.read(duration=1)
    td = time.time() - t
    print(f'{rawdata.shape} samples in {td} sec')

    print('halting')
    device.halt()
    print('exiting')


def test_decode():
    value = 11141120
    print(decode_trigger(value))


if __name__ == '__main__':
    print('test read')
    test_read()
    print('exited')
