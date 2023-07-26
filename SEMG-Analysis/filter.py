####### -------------------------------------构建滤波器------------------------------------
from scipy import signal
 
# 构建高通滤波函数
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype = "high", analog = False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

# 构建谐波滤波函数
def harmonic_filter(data, fs, f0, num_harmonics):
    b, a = signal.butter(4, (f0-1,f0+1), btype='bandstop', fs=fs)
    y = signal.filtfilt(b, a, data)
    for i in range(2, num_harmonics + 1):
        f = i * f0
        b, a = signal.butter(4, (f0-1,f0+1), btype='bandstop', fs=fs)
        y = signal.filtfilt(b, a, y)
    return y


########## ----------------------------提取特征新的尝试-------------------------------------------
import numpy as np
from scipy import signal
from scipy.stats import kurtosis, skew
from scipy.signal import welch
from pyeeg import svd_entropy # 奇异谱熵

###### -------------------------- 提取时域特征新尝试--------------------------------------------
def maxim(data):
    return np.max(data,axis=1)  # 最大值
def minm(data):
    return np.min(data,axis=1)  # 最小值
def mean(data):
    return np.mean(data,axis=1) # 平均值
def peak(data):
    ma=maxim(data)
    mi=minm(data)
    return ma - mi  # 峰-峰值
def abv(data):
    return np.mean(np.abs(data),axis=1)  # 绝对值的平均值(整流平均值)
def var(data):
    return np.var(data,axis=1)  # 方差
def std(data):
    return np.std(data,axis=1)  # 标准差
def ku(data):
    return kurtosis(data,axis=1)  # 峭度
def sk(data):
    return skew(data,axis=1)  # 偏度
def rms(data):
    return np.sqrt(np.mean(np.square(data),axis=1))  # 均方根
def S(data):
    rm = rms(data) 
    av= abv(data)
    return rm / av  # 波形因子
def C(data):
    pk=peak(data)
    rm = rms(data)
    return pk / rm  # 峰值因子
def I(data):
    pk=peak(data)
    av= abv(data)
    return pk / av  # 脉冲因子
def K(data):
    pk=peak(data)
    return pk / np.mean(np.power(np.abs(data), 0.5),axis=1)**2  # 裕度因子


###### -------------------------- 提取频率特征新尝试--------------------------------------------
def FC(data): # 重心频率
    f, p = signal.periodogram(data) #得到频率和振幅
    return np.sum(p*f,axis=1) / np.sum(p,axis=1)
def MSF(data):  # 均方频率
    f, p = signal.periodogram(data)
    return np.sum(f**2 * p,axis=1) / np.sum(p,axis=1)
def RMSF(data):  # 均方根频率
    msf=MSF(data)
    return np.sqrt(msf)
def VF(data):    # 频率方差
    fc=FC(data)
    fc = np.expand_dims(fc, axis=1) #扩展形状
    f, p = signal.periodogram(data)
    return np.sum((f - fc)**2 * p,axis=1) / np.sum(p,axis=1)
def RVF(data): # 频率标准差
    vf=VF(data)
    return np.sqrt(vf)


###### ------------------------------------ 计算信息熵 --------------------------------------------------
# 功率谱熵 测试通过
def PSE(data):
    freq, psd = welch(data, axis=1)
    psd_norm = psd / np.sum(psd, axis=1, keepdims=True)  # 归一化
    entropy = -np.sum(psd_norm * np.log2(psd_norm), axis=1)
    return entropy

# 能量熵 测试通过
def EE(data):
    energy = np.sum(data ** 2, axis=1)
    energy_norm = (data ** 2).T / energy  # 归一化
    entropy = -np.sum(energy_norm * np.log2(energy_norm), axis=0)
    return entropy                                      

# 奇异谱熵 有bug？还没放进字典
def qiyipu_entropy(data):
    window_size = round(len(data) / 2)
    svd_entro = svd_entropy(data, window_size)
    return svd_entro


# 创建一个python字典，将函数名和函数对应起来
FEATURE_FUNCTIONS = {
    "maxim": maxim,
    "minm": minm,
    "mean": mean,
    "peak": peak,
    "abv": abv,
    "var": var,
    "std": std,
    "ku": ku,
    "sk": sk,
    "rms": rms,
    "S": S,
    "C": C,
    "I": I,
    "K": K,
    "FC": FC,
    "MSF": MSF,
    "RMSF": RMSF,
    "VF": VF,
    "RVF": RVF,
    "PSE": PSE,
    "EE": EE,
}

# 定义连接特征的函数
def concatenate_features_new(data, *feature_names):
    features = [FEATURE_FUNCTIONS[name](data) for name in feature_names]
    return np.vstack(features)

def concatenate_features_new2(data, feature_names):
    features = [FEATURE_FUNCTIONS[name](data) for name in feature_names]
    return np.vstack(features)

########## ----------------------------提取特征-------------------------------------------
###### 数据预处理
def check_data(data): ### 代码测试通过
    np.random.seed(0)
    data = data.T
    length, num = data.shape # 获取数据尺寸，数据长度/数据组数
    if length == 1:      # 如果数据是一维的，将数据转置，再次获取数据的长度和组数
        data = data.T
        length, num = data.shape
    return data,num


####### ------------------------------------绘制滤波器-------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def draw_filter(filter_type, cutoff, fs, order=2):
    """
    Plots the frequency response of a filter given its type, cutoff frequency, sampling rate, and order.

    Parameters:
    filter_type (str): Type of filter, either 'butterworth', 'chebyshev1', 'chebyshev2', or 'elliptic'.
    cutoff (float): Cutoff frequency of the filter.
    fs (float): Sampling rate of the filter.
    order (int): Order of the filter, default is 2.

    Returns:
    None.
    """
    # Design the filter
    if filter_type == 'butterworth':
        b, a = signal.butter(N=order, Wn=cutoff, btype='highpass', fs=fs)
    elif filter_type == 'chebyshev1':
        b, a = signal.cheby1(N=order, rp=0.5, Wn=cutoff, btype='highpass', fs=fs)
    elif filter_type == 'chebyshev2':
        b, a = signal.cheby2(N=order, rs=30, Wn=cutoff, btype='highpass', fs=fs)
    elif filter_type == 'elliptic':
        b, a = signal.ellip(N=order, rp=0.5, rs=30, Wn=cutoff, btype='highpass', fs=fs)
    else:
        raise ValueError("Invalid filter type. Choose 'butterworth', 'chebyshev1', 'chebyshev2', or 'elliptic'.")

    # Compute the frequency response
    w, h = signal.freqz(b, a)

    # Plot the frequency response
    plt.semilogx(w, 20 * np.log10(abs(h)))
    plt.title(f'{filter_type.capitalize()} filter frequency response')
    plt.xlabel('Frequency [radians / second]')
    plt.ylabel('Amplitude [dB]')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.axvline(cutoff, color='green') # cutoff frequency
    plt.show()