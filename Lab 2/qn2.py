import numpy as np
import matplotlib.pyplot as plt
from utils import convolve, play_sound, fnNormalize16BitToFloat, read_sound, fnNormalizeFloatTo16Bit, save_sound, delta, fnGenSampledSinusoid
from scipy import signal

ipcleanfilename = 'testIp_16bit.wav'

def q2_5_d():
    ipnoisyfilename = 'helloworld_noisy_16bit.wav'
    _, sampleX_16bit = read_sound(ipnoisyfilename)
    x = fnNormalize16BitToFloat(sampleX_16bit)
    B = [1, -0.7653668, 0.99999]
    A = [1, -0.722744, 0.888622]
    y_ifil = signal.lfilter(B, A, x)
    y_ifil_16bit = fnNormalizeFloatTo16Bit(y_ifil)
    y_ifil_name = 'y_ifil.wav'
    save_sound(y_ifil_name, 16000, y_ifil_16bit)
    play_sound(y_ifil_name)
    [f1, t1, Sxx1] = signal.spectrogram(x, 16000, window=('blackmanharris'),nperseg=512,noverlap=int(0.9*512))
    [f2, t2, Sxx2] = signal.spectrogram(y_ifil, 16000, window=('blackmanharris'),nperseg=512,noverlap=int(0.9*512))
    _, ax = plt.subplots(2, 1)
    ax[0].pcolormesh(t1, f1, 10*np.log10(Sxx1))
    ax[1].pcolormesh(t2, f2, 10*np.log10(Sxx2))
    plt.show()

def q2_5_c():
    ipnoisyfilename = 'helloworld_noisy_16bit.wav'
    _, sampleX_16bit = read_sound(ipnoisyfilename)
    x = fnNormalize16BitToFloat(sampleX_16bit)
    y = np.zeros(len(x), dtype=float)
    B = [1, -0.7653668, 0.99999]
    A = [1, -0.722744, 0.888622]
    for n in range(len(x)):
        if n == 0:
            y[n] = 1 * x[n]
        elif n == 1:
            y[n] = 1 * x[n] + (-0.7653668) * x[n - 1] - (-0.722744) * y[n - 1]
        else:
            y[n] = 1 * x[n] + (-0.7653668) * x[n - 1] + 0.99999 * x[n - 2] - (-0.722744) * y[n - 1] - 0.888622 * y[n - 2]
    y_ifil = signal.lfilter(B, A, x)
    for i in range(len(y)):
        if y[i] == y_ifil[i]:
            continue
        else:
            print("=================")
            print(y[i])
            print(y_ifil[i])
            print("=================")

def q2_5_a():
    ipnoisyfilename = 'helloworld_noisy_16bit.wav'
    _, sampleX_16bit = read_sound(ipnoisyfilename)
    x = fnNormalize16BitToFloat(sampleX_16bit)
    [f, t, Sxx] = signal.spectrogram(x, 16000, window=('blackmanharris'),nperseg=512,noverlap=int(0.9*512))
    _, ax = plt.subplots(2, 1)
    t1 = np.arange(0, len(x), 1) * (1 / 16000)
    ax[0].plot(t1, x)
    ax[0].grid()
    ax[1].pcolormesh(t, f, 10*np.log10(Sxx))
    plt.show()

def q2_4_c():
    h1 = np.array([0.06523, 0.14936, 0.21529, 0.2402, 0.21529, 0.14936, 0.06523], dtype='float')
    h2 = np.array([-0.06523, -0.14936, -0.21529, 0.7598, -0.21529, -0.14936, -0.06523], dtype='float')
    _, x1 = fnGenSampledSinusoid(0.1, 700, 0, 16000, 0, 0.05)
    _, x2 = fnGenSampledSinusoid(0.1, 3333, 0, 16000, 0, 0.05)
    x = x1 + x2
    y1 = np.convolve(x, h1)
    t1 = np.arange(0, len(y1), 1) * (1 / 16000)
    y2 = np.convolve(x, h2)
    t2 = np.arange(0, len(y2), 1) * (1 / 16000)
    [f, t, Sxx] = signal.spectrogram(x, 16000, window=('blackmanharris'),nperseg=512,noverlap=int(0.9*512))
    plt.pcolormesh(t, f, 10*np.log10(Sxx))
    plt.show()
    _, ax = plt.subplots(2, 1)
    ax[0].plot(t1, y1)
    ax[0].grid()
    ax[1].plot(t2, y2)
    ax[1].grid()
    plt.show()
    [f1, t1, Sxx1] = signal.spectrogram(y1, 16000, window=('blackmanharris'),nperseg=512,noverlap=int(0.9*512))
    [f2, t2, Sxx2] = signal.spectrogram(y2, 16000, window=('blackmanharris'),nperseg=512,noverlap=int(0.9*512))
    _, ax = plt.subplots(2, 1)
    ax[0].pcolormesh(t1, f1, 10*np.log10(Sxx1))
    ax[1].pcolormesh(t2, f2, 10*np.log10(Sxx2))
    plt.show()
    y1_16bit = fnNormalizeFloatTo16Bit(y1)
    y2_16bit = fnNormalizeFloatTo16Bit(y2)
    y1_file_name = 'y1_16bit.wav'
    y2_file_name = 'y2_16bit.wav'
    save_sound(y1_file_name, 16000, y1_16bit)
    save_sound(y2_file_name, 16000, y2_16bit)
    play_sound(y1_file_name)
    play_sound(y2_file_name)

def q2_4_b():
    h1 = np.array([0.06523, 0.14936, 0.21529, 0.2402, 0.21529, 0.14936, 0.06523], dtype='float')
    h2 = np.array([-0.06523, -0.14936, -0.21529, 0.7598, -0.21529, -0.14936, -0.06523], dtype='float')
    n = np.arange(0, 10)
    x = np.zeros(len(n))
    for i in range(len(n)):
        x[i] = delta(n[i]) - 2 * delta(n[i] - 15)
    results_1_1 = convolve(x, h1)
    results_1_2 = np.convolve(x, h1)
    results_1_3 = signal.lfilter(h1, [1], x)
    print(results_1_1)
    print(results_1_2)
    print(results_1_3)
    results_2_1 = convolve(x, h2)
    results_2_2 = np.convolve(x, h2)
    results_2_3 = signal.lfilter(h2, [1], x)
    print(results_2_1)
    print(results_2_2)
    print(results_2_3)
    

def q2_4_a():
    h1 = np.array([0.06523, 0.14936, 0.21529, 0.2402, 0.21529, 0.14936, 0.06523], dtype='float')
    h2 = np.array([-0.06523, -0.14936, -0.21529, 0.7598, -0.21529, -0.14936, -0.06523], dtype='float')
    _, ax = plt.subplots(2, 1)
    ax[0].stem(h1)
    ax[0].grid()
    ax[1].stem(h2)
    ax[1].grid()
    plt.show()

def q2_3_b():
    impulseH = np.zeros(8000)
    impulseH[1] = 1
    impulseH[4000] = 0.5
    impulseH[7900] = 0.3
    play_sound(ipcleanfilename)
    Fs, sampleX_16bit = read_sound(ipcleanfilename)
    sampleX_float = fnNormalize16BitToFloat(sampleX_16bit)
    y = convolve(sampleX_float, impulseH)
    y_16bit = fnNormalizeFloatTo16Bit(y)
    save_file_name = "t2_16bit.wav"
    save_sound(save_file_name, Fs, y_16bit)
    play_sound(save_file_name)
    


def q2_3_a():
    impulseH = np.zeros(8000)
    impulseH[1] = 1
    impulseH[4000] = 0.5
    impulseH[7900] = 0.3
    plt.stem(impulseH)
    plt.grid()
    plt.show()
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    h = np.array([1, 2, 3, 4, 5, 6])
    y_test = convolve(x, h)
    y = np.convolve(x, h)
    comparison = y == y_test
    assert comparison.all()


def q2_1_b():
    n = np.arange(0, 102)
    x = np.cos(0.1 * np.pi * n)
    h = np.array([0.2, 0.3, -0.5])
    y = np.convolve(x, h)

    _, ax = plt.subplots(2, 1)
    ax[0].stem(x)
    ax[0].grid()
    ax[1].stem(y)
    ax[1].grid()
    plt.show()

def main():
    # q2_1_b()
    # q2_3_a()
    q2_5_d()

if __name__ == "__main__":
    main()