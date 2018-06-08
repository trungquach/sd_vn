import glob
import pandas as pd
import re
import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

# merage vm2012 log format data
# return a [samples,features] array
def merageVm2012Data(expression):
    Accs = []
    for f in sorted(glob.glob(expression), key=numericalSort):
        df = pd.read_csv(f, engine='python')
        accsPerFile = readVm2012Data(df)
        Accs.append(accsPerFile)
    Accs = np.array(Accs)
    Accs = Accs.reshape(Accs.shape[0] * Accs.shape[1])
    return Accs

# read vm2012 log format data
# read acceleration, log export datetime, log data span, frequency
def readVm2012Data(df):
    # get data
    accs = np.array(list(map(np.float32, df.ix[22:, 0])))

    return accs

def numericalSort(value):

    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])

    return parts

def spliteAcc2fft(accs, splite_n, fs, timeserial=None, MAX_FREQ=None, is_draw_wav=False, is_draw_fft=False, figure=""):
    freqs = np.array(np.fft.fftfreq(splite_n, d=1.0 / fs))
    if MAX_FREQ == None:
        MAX_FREQ = np.max(freqs)
    indexes = np.where((freqs >= 0) & (freqs <= MAX_FREQ))[0]
    hammingWindow = np.hamming(splite_n)
    datas = []
    start = 0

    while start + splite_n < len(accs) - 1:
        windowedData = hammingWindow * accs[start:start + splite_n]

        if is_draw_wav == True:
            fig = plt.figure(figure + " acc " + str(splite_n / fs))
            plt.style.use("ggplot")
            Y = accs[start:start + splite_n]
            X = [timeserial + (start + i) / fs / 3600 / 24 for i in np.arange(len(windowedData))]
            plt.plot(X, Y, c="b")
            plt.ylim([-25, 25])
            fig.show()
            is_draw_wav = False

        X = np.fft.fft(windowedData)  # FFT
        amplitudeSpectrum = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in X]  # 振幅スペクトル
        amplitudes = np.array(amplitudeSpectrum)[indexes]
        datas.append(amplitudes)

        if is_draw_fft == True:
            fig = plt.figure(figure + " acc fft " + str(splite_n / fs))
            plt.style.use("ggplot")
            X = freqs[indexes]
            Y = amplitudes
            plt.plot(X, Y, c="b")
            plt.ylim([0, 5250])
            fig.show()
            is_draw_fft = False

        start += splite_n
    datas = np.array(datas)

    return datas

# structure: elements in datas must follow order: normalDatas, bearingAlDatas, gearAlDatas
def split_train_test(datas, labels, frac=0.8):
    len_normal = np.where(labels == 0)[0].shape[0]

    split_id = int(0.8 * len_normal)
    train_datas, train_labels = datas[:split_id, :], labels[:split_id]
    test_datas, test_labels = datas[split_id:, :], labels[split_id:]

    print("Training dataset: ", train_datas.shape)
    print("Testing dataset: ", test_datas.shape)

    return train_datas, test_datas, train_labels, test_labels


# global val
dataset_path = './data/dataset_2018_04_26/'
normal_position1_data_path = dataset_path + 'normal_position1/'
normal_position2_data_path = dataset_path + 'normal_position2/'
bearing_position1_data_path = dataset_path + 'bearing_position1/'
bearing_position2_data_path = dataset_path + 'bearing_position2/'
gear_position1_data_path = dataset_path + 'gear_position1/'
gear_position2_data_path = dataset_path + 'gear_position2/'
csv_file_regular_expression = '*.wdat'
freq = 100000

if __name__ == "__main__":
    # set Japanese Font
    fp = FontProperties(fname=r'/System/Library/Fonts/ヒラギノ明朝 ProN.ttc', size=9)

    accs = merageVm2012Data(normal_position1_data_path + '/' + csv_file_regular_expression)
    print("Normal position 1 : " , accs.shape)

    accs = merageVm2012Data(bearing_position1_data_path + '/' + csv_file_regular_expression)
    print("Bearing position 1 : " , accs.shape)

    accs = merageVm2012Data(gear_position1_data_path + '/' + csv_file_regular_expression)
    print("Gear position 1 : " , accs.shape)

    accs = merageVm2012Data(normal_position2_data_path + '/' + csv_file_regular_expression)
    print("Normal position 2 : " , accs.shape)

    accs = merageVm2012Data(bearing_position2_data_path + '/' + csv_file_regular_expression)
    print("Bearing position 2 : " , accs.shape)

    accs = merageVm2012Data(gear_position2_data_path + '/' + csv_file_regular_expression)
    print("Gear position 2 : " , accs.shape)
