import numpy as np
import math

def myDTFS(ipx, N):
    X = [0] * N
    for k in range(len(X)):
        number_of_samples = len(ipx)
        result = 0
        for n in range(number_of_samples):
            intermediate_result = 2 * np.pi / number_of_samples * k * n
            result += ipx[n] * (math.cos(intermediate_result) - 1j * math.sin(intermediate_result))
        X[k] = 1 / N * result
    return X

def myIDTFS(Xdtfs):
    len_of_Xdtfs = len(Xdtfs)
    x = [0] * len_of_Xdtfs
    for n in range(len(x)):
        result = 0
        for k in range(len_of_Xdtfs):
            intermediate_result = 2 * np.pi / len_of_Xdtfs * k * n
            result += Xdtfs[k] * (math.cos(intermediate_result) + 1j * math.sin(intermediate_result))
        x[n] = result
    return x

def myDFT(ipx, N):
    X = [0] * N
    for k in range(len(X)):
        number_of_samples = len(ipx)
        result = 0
        for n in range(number_of_samples):
            intermediate_result = 2 * np.pi / number_of_samples * k * n
            result += ipx[n] * (math.cos(intermediate_result) - 1j * math.sin(intermediate_result))
        X[k] = result
    return X

def myIDFT(Xdtf):
    len_of_Xdtf = len(Xdtf)
    x = [0] * len_of_Xdtf
    for n in range(len(x)):
        result = 0
        for k in range(len_of_Xdtf):
            intermediate_result = 2 * np.pi / len_of_Xdtf * k * n
            result += Xdtf[k] * (math.cos(intermediate_result) + 1j * math.sin(intermediate_result))
        x[n] = result / len_of_Xdtf
    return x

def myDFTConvolve(ipX, impulseH):
    len_of_original_ipX = len(ipX)
    len_of_original_impulseH = len(impulseH)
    if (len(ipX) < len(impulseH)):
        ipX = np.pad(ipX, (0, len(impulseH) - len(ipX)), 'constant')
    elif (len(ipX) > len(impulseH)):
        impulseH = np.pad(impulseH, (0, len(ipX) - len(impulseH)), 'constant')
    X = myDFT(ipX, len(ipX))
    H = myDFT(impulseH, len(impulseH))
    Y = np.multiply(X,H)
    y = myIDFT(Y)
    y = np.pad(y, (0, len_of_original_ipX + len_of_original_impulseH - 1 - len(y)), 'constant')
    return y