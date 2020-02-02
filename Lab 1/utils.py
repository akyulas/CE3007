import numpy as np
import scipy.io.wavfile  as wavfile
import winsound

DTMF_FREQ = {'0': [1336, 941],
            '1': [1209, 697],
            '2': [1336, 697],
            '3': [1477, 697],
            '4': [1209, 770],
            '5': [1336, 770],
            '6': [1477, 770],
            '7': [1209, 852],
            '8': [1336, 852],
            '9': [1477, 852],
            '*': [1209, 941],
            '#': [1477, 941]
             }

def fnGenSampledDTMF(seq, Fs, durTone):
    sTime = 0
    eTime = 0 + durTone + 1/Fs
    y = []

    for char in seq:
        n = np.arange(sTime, eTime, 1.0/Fs)
        y1 = 0.5*np.sin(2 * np.pi * DTMF_FREQ[char][0] * n)
        y2 = 0.5*np.sin(2 * np.pi * DTMF_FREQ[char][0] * n)
        y = np.concatenate((y, y1 + y2))

    return [n,y]

# The following function generates a continuous time sinusoid
# given the amplitude A, F (cycles/seconds), Fs=sampling rate, start and endtime
def fnGenSampledSinusoid(A,Freq,Phi,Fs,sTime,eTime):
    # Showing off how to use numerical python library to create arange
    n = np.arange(sTime,eTime,1.0/Fs)
    y = A*np.cos(2 * np.pi * Freq * n + Phi)
    return [n,y]


# The input is a float array (should have dynamic value from -1.00 to +1.00
def fnNormalizeFloatTo16Bit(yFloat):
    y_16bit = [int(s*32767) for s in yFloat]
    return(np.array(y_16bit, dtype='int16'))

# The input is a float array (should have dynamic value from -1.00 to +1.00
def fnNormalize16BitToFloat(y_16bit):
    yFloat = [float(s/32767.0) for s in y_16bit]
    return(np.array(yFloat, dtype='float'))

def save_sound(file_name, sampling_frequency, bits):
    wavfile.write(file_name, sampling_frequency, bits)

def play_sound(file_name):
    winsound.PlaySound(file_name, winsound.SND_FILENAME)
 
