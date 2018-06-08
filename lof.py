import glob
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from matplotlib.font_manager import FontProperties
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score
from sklearn.externals import joblib

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

def spliteAcc2fft(accs, splite_n, freq, MAX_FREQ=None):
    freqs = np.array(np.fft.fftfreq(splite_n, d=1.0/ freq))
    if MAX_FREQ == None:
        MAX_FREQ = np.max(freqs)
    indexes = np.where((freqs >= 0) & (freqs <= MAX_FREQ))[0]
    hammingWindow = np.hamming(splite_n)
    datas = []
    start = 0

    while start + splite_n < len(accs) - 1:
        windowDatas = hammingWindow * accs[start: start + splite_n]
        X = np.fft.fft(windowDatas)
        amplitudeSpectrum = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in X]
        amplitudes = np.array(amplitudeSpectrum)[indexes]
        datas.append(amplitudes)

        start+= splite_n
    datas = np.array(datas)
    return datas


def create_lof_model(neighbors):
    return LocalOutlierFactor(n_neighbors=neighbors)


# global val
dataset_path = 'data/dataset_2018_04_26/'
normal_position1_data_path = dataset_path + 'normal_position1/'
normal_position2_data_path = dataset_path + 'normal_position2/'
bearing_position1_data_path = dataset_path + 'bearing_position1/'
bearing_position2_data_path = dataset_path + 'bearing_position2/'
gear_position1_data_path = dataset_path + 'gear_position1/'
gear_position2_data_path = dataset_path + 'gear_position2/'
csv_file_regular_expression = '*.wdat'
freq = 100000
N = 1024 * 2
use_data_test = True

def _init_data(input, is_use_data_test=True, rate = 0.7):
    if is_use_data_test:
        no_row_test_normal = (int)(input[0].shape[0] * rate)
        datas_train = np.r_(input[0][:no_row_test_normal], input[1], input[2])
        datas_test = input[0][-no_row_test_normal:]

        ground_truth = np.zeros(datas_test.shape[0], dtype=int)
        ground_truth[-(input[1].shape[0] + input[2].shape[0]):] = 1
    else:
        datas_train = np.r_(input)
        ground_truth = np.zeros(datas_train.shape[0], dtype=int)
        ground_truth[-(input[1].shape[0] + input[2].shape[0]):] = 1



    return (datas_train,ground_truth, datas_test)

if __name__ == "__main__":
    # set Japanese Font
    fp = FontProperties(fname=r'/System/Library/Fonts/ヒラギノ明朝 ProN.ttc', size=9)

    try:
        accs_normal1 = merageVm2012Data(normal_position1_data_path + '/' + csv_file_regular_expression)
        print("Normal position 1 : ", accs_normal1.size)
        accs_bearing1 = merageVm2012Data(bearing_position1_data_path + '/' + csv_file_regular_expression)
        print("Bearing position 1 : ", accs_bearing1.size)
        accs_gear1 = merageVm2012Data(gear_position1_data_path + '/' + csv_file_regular_expression)
        print("Gear position 1 : ", accs_gear1.size)
        accs_norma2 = merageVm2012Data(normal_position2_data_path + '/' + csv_file_regular_expression)
        print("Normal position 2 : ", accs_norma2.size)
        accs_bearing2 = merageVm2012Data(bearing_position2_data_path + '/' + csv_file_regular_expression)
        print("Bearing position 2 : ", accs_bearing2.size)
        accs_gear2 = merageVm2012Data(gear_position2_data_path + '/' + csv_file_regular_expression)
        print("Gear position 2 : ", accs_gear2.size)
        plt.figure()
    except Exception as e:
        print(e.__str__())

    # model for position 1
    best_roc_auc = 0.0
    for n in range(1 , 10):
        N = n * 1024
        normal1_datas = spliteAcc2fft(accs_normal1, N, freq)
        bearing1_datas = spliteAcc2fft(accs_bearing1, N, freq)
        gear1_datas = spliteAcc2fft(accs_gear1, N, freq)

        datas_train,ground_truth, datas_test =  _init_data([normal1_datas, bearing1_datas, gear1_datas], is_use_data_test = use_data_test, rate = 0.8)

        lof_model = create_lof_model(datas_train.shape[0]//3).fit(datas_train)

        ground_truth = np.zeros(datas_train.shape[0], dtype=int)
        ground_truth[-(bearing1_datas.shape[0] + gear1_datas.shape[0]):] = 1

        if use_data_test:
            y_score = -lof_model._decision_function(datas_test)
        else:
            y_score = -lof_model.negative_outlier_factor_
        # Compute ROC curve and ROC area for each class
        fpr, tpr, thresholds = roc_curve(ground_truth, y_score)
        roc_auc = auc(fpr, tpr)
        #compute f1_score
        y_pred = np.zeros(datas_train.shape[0], dtype=int)
        y_pred[lof_model.negative_outlier_factor_ <= lof_model.threshold_] = 1
        f1 = f1_score(ground_truth, y_pred)

        #select best model with best roc_auc
        if best_roc_auc < roc_auc:
            best_roc_auc = roc_auc
            best_model = lof_model
        plt.plot(fpr, tpr, lw=2, label='N = %d (area = %0.2f) F1 = %0.2f' % (N, roc_auc, f1))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for position 1')
    plt.legend(loc="best")

    # # save best model to disk
    # filename = 'finalized_model_1.sav'
    # joblib.dump(best_model, filename)


    # # model for position 2
    # plt.figure()
    # for n in range(1 , 10):
    #     N = n * 1024
    #     normal2_datas = spliteAcc2fft(accs_norma2, N, freq)
    #     bearing2_datas = spliteAcc2fft(accs_bearing2, N, freq)
    #     gear2_datas = spliteAcc2fft(accs_gear2, N, freq)
    #
    #     datas_train = np.r_[normal2_datas, bearing2_datas, gear2_datas]
    #
    #     lof_model = create_lof_model(datas_train.shape[0]//3).fit(datas_train)
    #
    #     ground_truth = np.zeros(datas_train.shape[0], dtype=int)
    #     ground_truth[-(bearing2_datas.shape[0] + gear2_datas.shape[0]):] = 1
    #
    #     # Compute ROC curve and ROC area for each class
    #     fpr, tpr, thresholds = roc_curve(ground_truth, -lof_model.negative_outlier_factor_)
    #     roc_auc = auc(fpr, tpr)
    #     f1 = f1_score(ground_truth, -lof_model.negative_outlier_factor_, average=None)
    #
    #     # select best model with best roc_auc
    #     if best_roc_auc < roc_auc:
    #         best_roc_auc = roc_auc
    #         best_model = lof_model
    #
    #     plt.plot(fpr, tpr, lw=2, label='N = %d (area = %0.2f) F1 = %0.2f' % (N, roc_auc, f1))
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC for position 2')
    # plt.legend(loc="best")
    plt.show(block=True)

    # # save best model to disk
    # filename = 'finalized_model_2.sav'
    # joblib.dump(best_model, filename)
