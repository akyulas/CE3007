from utils import fnGenSampledSinusoid, fnNormalize16BitToFloat, fnNormalizeFloatTo16Bit
from utils import fnGenSampledDTMF, save_sound, play_sound
import matplotlib.pyplot as plt
# plotting 3D complex plane
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def q3_5():
    numSamples = 16
    A = 1
    for k in range(4):
        n = np.arange(0, numSamples, 1)
        y1= np.multiply(A, np.exp(1j * 2 * np.pi / numSamples * k * n))
        # plotting in polar, understand what the spokes are 
        plt.figure(2)
        for x in y1:
            plt.polar([0,np.angle(x)],[0,np.abs(x)],marker='o')
        
        plt.title('Polar plot showing phasors at n=0..N')
        plt.show()
       # plotting 3D complex plane
        plt.rcParams['legend.fontsize'] = 10
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        reVal = y1[0:numSamples].real
        imgVal = y1[0:numSamples].imag
        ax.plot(n,reVal, imgVal,  label='complex exponential phasor')
        ax.scatter(n,reVal,imgVal, c='r', marker='o')
        ax.set_xlabel('sample n')
        ax.set_ylabel('real')
        ax.set_zlabel('imag')
        ax.legend()
        plt.show()

def q3_4():
    numSamples = 36
    A=0.95
    for w in [2*np.pi/36, 2*np.pi/18]:
        n = np.arange(0, numSamples, 1)
        y1 = np.multiply(np.power(A, n), np.exp(1j * w * n))
        
        
        # plotting in 2-D, the real and imag in the same figure
        plt.figure(1)
        plt.plot(n, y1[0:numSamples].real,'r--o')
        plt.plot(n, y1[0:numSamples].imag,'g--o')
        plt.xlabel('sample index n'); plt.ylabel('y[n]')
        plt.title('Complex exponential (red=real) (green=imag)')
        plt.grid()
        plt.show()
        
        # plotting in polar, understand what the spokes are
        plt.figure(2)
        for x in y1:
            plt.polar([0,np.angle(x)],[0,np.abs(x)],marker='o')
        
        plt.title('Polar plot showing phasors at n=0..N')
        plt.show()
        
        # plotting 3D complex plane
        plt.rcParams['legend.fontsize'] = 10
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        reVal = y1[0:numSamples].real
        imgVal = y1[0:numSamples].imag
        ax.plot(n,reVal, imgVal,  label='complex exponential phasor')
        ax.scatter(n,reVal,imgVal, c='r', marker='o')
        ax.set_xlabel('sample n')
        ax.set_ylabel('real')
        ax.set_zlabel('imag')
        ax.legend()
        plt.show()

def q3_3():
     A=0.5; F=10; Phi = 0; Fs=60; sTime=0; eTime = 1 + 1/Fs
     n, y1 = fnGenSampledSinusoid(A, F, Phi, Fs, sTime, eTime)
     A = 1; F = 15
     n, y2 = fnGenSampledSinusoid(A, F, Phi, Fs, sTime, eTime)
     y = y1 + y2
     num_of_samples = np.arange(0,len(n),1)
     plt.stem(num_of_samples, y,'g', use_line_collection=True)
     plt.grid()
     plt.show()

def q3_2():
    Fs = 16000
    duration = 1
    _, y = fnGenSampledDTMF("0123456789*#", Fs, duration)
    y_16bit = fnNormalizeFloatTo16Bit(y)
    file_name = 't1_16bit_DTMF.wav'
    save_sound(file_name, Fs, y_16bit)
    play_sound(file_name)

def q3_1_c():
    print("At q3 1c!")
    A=0.5; F=1; Phi = 0; Fs=16; sTime=0; eTime = 0.006 + 1/Fs
    _, axs = plt.subplots(3,1)
    [n,yfloat] = fnGenSampledSinusoid(A, F, Phi, Fs * 1000, sTime, eTime)
    axs[0].plot(n, yfloat)
    axs[0].grid()
    [n,yfloat] = fnGenSampledSinusoid(A, F, Phi, Fs, sTime, eTime)
    axs[1].plot(n, yfloat,'r--o')
    axs[1].grid()
    num_of_samples = np.arange(0,len(n),1)
    axs[2].stem(num_of_samples, yfloat,'g', use_line_collection=True)
    axs[2].grid()
    plt.show()
    print("##########################################################")

def q3_1_b():
    print("At q3 1b!")
    for F in [1000, 17000]:
        print(F)
        A=0.5; Phi = 0; Fs=16000; sTime=0; eTime = 0.006 + 1/Fs
        _, axs = plt.subplots(3,1)
        [n,yfloat] = fnGenSampledSinusoid(A, F, Phi, Fs * 1000, sTime, eTime)
        axs[0].plot(n, yfloat)
        axs[0].grid()
        [n,yfloat] = fnGenSampledSinusoid(A, F, Phi, Fs, sTime, eTime)
        axs[1].plot(n, yfloat,'r--o')
        axs[1].grid()
        num_of_samples = np.arange(0,len(n),1)
        axs[2].stem(num_of_samples, yfloat,'g', use_line_collection=True)
        axs[2].grid()
        plt.show()
        print("##########################################################")


def q3_1_a():
    print("At q3 1a!")
    A=0.5; F=1000; Phi = 0; Fs=16000; sTime=0; eTime = 0.4
    [_,yfloat] = fnGenSampledSinusoid(A, F, Phi, Fs, sTime, eTime)
    y_16bit = fnNormalizeFloatTo16Bit(yfloat)
    file_name_16_bit = "t1_16bit_" + str(F) + ".wav"
    file_name_32_bit = "t1_float_" + str(F) + ".wav"
    save_sound(file_name_16_bit, Fs, y_16bit)
    save_sound(file_name_32_bit, Fs, yfloat)
    print(F)
    play_sound(file_name_16_bit)
    for F in range(2000, 34000, 2000):
        [_,yfloat] = fnGenSampledSinusoid(A, F, Phi, Fs, sTime, eTime)
        y_16bit = fnNormalizeFloatTo16Bit(yfloat)
        file_name_16_bit = "t1_16bit_" + str(F) + ".wav"
        file_name_32_bit = "t1_float_" + str(F) + ".wav"
        save_sound(file_name_16_bit, Fs, y_16bit)
        save_sound(file_name_32_bit, Fs, yfloat)
        print(F)
        play_sound(file_name_16_bit)
    print("##########################################################")

def main():
    q3_1_a()
    q3_1_b()
    q3_2()
    q3_3()
    q3_4()
    q3_5()

if __name__ == "__main__":
    main()
    