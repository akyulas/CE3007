from utils import myDTFS, myDFT, myIDTFS, myIDFT, myDFTConvolve
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import numpy as np
import cmath
from scipy import signal

def q2_5():
    x = [1, 2, 0, 0, 0, 0, 0, 0]
    h = [1, 2, 3, 0, 0, 0, 0, 0]
    y = myDFTConvolve(x, h)
    expected_result = signal.fftconvolve(x, h)
    print(np.allclose(list(y), list(expected_result)))

def q2_4():
    for N in [12, 24,48,96]:
        ipx = [1, 1, 1, 1, 1, 1, 1]
        ipx = np.pad(ipx, (0, N - len(ipx)), 'constant')
        DTFS_result = myDTFS(ipx, N)
        DTFS_magnitude = [abs(result) for result in DTFS_result]
        DTFS_phase = [cmath.phase(result) for result in DTFS_result]
        _, ax = plt.subplots(2, 1)
        ax[0].stem(DTFS_magnitude)
        ax[0].grid()
        ax[1].stem(DTFS_phase)
        ax[1].grid()
        plt.show()

def q2_3():
    N = 32
    k=1
    W = np.zeros(shape=(N),dtype=complex)
    for n in np.arange(0,N):
        W[n] = np.exp(-1j*(2*np.pi/N)*k*n)

    W_angle = np.angle(W)
    # the lenbth is 1, we are only interested in the angle of each phasor

    plt.figure()
    plt.title('Each row shows the k-th harmonic, from n=0..N-1 index')
    Q = plt.quiver( np.cos(W_angle),np.sin(W_angle),  units='width')
    titleStr = 'Fourier complex vectors N='+str(N)
    plt.title(titleStr)
    plt.ylabel('k-values')
    plt.xlabel('n-values')
    plt.grid()
    plt.show()

def q2_2_d():
    ipx_1 = [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ipx_2 = [10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    len_of_sample = len(ipx_1)
    DTFS_result_1 = myDTFS(ipx_1, len_of_sample)
    DTFS_magnitude_1 = [abs(result) for result in DTFS_result_1]
    DTFS_phase_1 = [cmath.phase(result) for result in DTFS_result_1]
    DTFS_result_2 = myDTFS(ipx_2, len_of_sample)
    DTFS_magnitude_2 = [abs(result) for result in DTFS_result_2]
    DTFS_phase_2 = [cmath.phase(result) for result in DTFS_result_2]
    _, ax = plt.subplots(2, 1)
    ax[0].stem(DTFS_magnitude_1)
    ax[0].grid()
    ax[1].stem(DTFS_phase_1)
    ax[1].grid()
    plt.show()
    _, ax = plt.subplots(2, 1)
    ax[0].stem(DTFS_magnitude_2)
    ax[0].grid()
    ax[1].stem(DTFS_phase_2)
    ax[1].grid()
    plt.show()

def q2_2_c():
    ipx = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    len_of_sample = len(ipx)
    DTFS_result = myDTFS(ipx, len_of_sample)
    IDTFS_result = myIDTFS(DTFS_result)
    DFT_result = myDFT(ipx, len_of_sample)
    IDFT_result = myIDFT(DFT_result)
    FFT_result = fft(ipx, len_of_sample)
    print(np.allclose(np.array(DTFS_result) * len_of_sample, FFT_result))
    print(np.allclose(DFT_result, FFT_result))
    print(np.allclose(ipx, IDTFS_result))
    print(np.allclose(ipx, IDFT_result))

def q2_2_b():
    ipx = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    len_of_sample = len(ipx)
    DTFS_result = myDTFS(ipx, len_of_sample)
    DTFS_magnitude = [abs(result) for result in DTFS_result]
    DTFS_phase = [cmath.phase(result) for result in DTFS_result]
    DFT_result = myDFT(ipx, len_of_sample)
    DFT_magnitude = [abs(result) for result in DFT_result]
    DFT_phase = [cmath.phase(result) for result in DFT_result]
    _, ax = plt.subplots(2, 1)
    ax[0].stem(DTFS_magnitude)
    ax[0].grid()
    ax[1].stem(DTFS_phase)
    ax[1].grid()
    plt.show()
    _, ax = plt.subplots(2, 1)
    ax[0].stem(DFT_magnitude)
    ax[0].grid()
    ax[1].stem(DFT_phase)
    ax[1].grid()
    plt.show()

def main():
    q2_4()

if __name__ == "__main__":
    main()