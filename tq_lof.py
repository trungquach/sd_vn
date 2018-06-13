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
import utils

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


def _init_data(input, is_use_data_test=True, rate=0.7):
    if is_use_data_test:
        no_row_test_normal = (int)(input[0].shape[0] * rate)
        datas_train = np.r_[input[0][:no_row_test_normal], input[1], input[2]]
        datas_test = input[0][-no_row_test_normal:]

        ground_truth = np.zeros(datas_test.shape[0], dtype=int)
        ground_truth[-(input[1].shape[0] + input[2].shape[0]):] = 1
    else:
        datas_train = np.r_[input]
        ground_truth = np.zeros(datas_train.shape[0], dtype=int)
        ground_truth[-(input[1].shape[0] + input[2].shape[0]):] = 1

    return (datas_train, ground_truth, datas_test)

def get_best_threshold_roc(fpr,tpr,thresholds):
    # the best one is the point which closest to the top left corner.
    top_left_point = (0.0,1.0) #(FPR, TPR)

    dists = np.array([np.sqrt((0.0 - _fpr)**2 + (1 - _tpr)**2 ) for (_fpr,_tpr) in zip(fpr,tpr)] )
    max_idx = np.argsort(dists)[0]

    return thresholds[max_idx]

def build_model(accs_normal1,accs_bearing1,accs_gear1):

    N = 1024 * 2
    normal1_datas = utils.spliteAcc2fft(accs_normal1, N, freq)
    bearing1_datas = utils.spliteAcc2fft(accs_bearing1, N, freq)
    gear1_datas = utils.spliteAcc2fft(accs_gear1, N, freq)
    n_sample_out = 200
    normal_datas_in, normal_datas_out = normal1_datas[n_sample_out:], normal1_datas[:n_sample_out]
    bearing_datas_in, bearing_datas_out = bearing1_datas[n_sample_out:], bearing1_datas[:n_sample_out]
    gear_datas_in, gear_datas_out = gear1_datas[n_sample_out:], gear1_datas[:n_sample_out]

    datas = np.r_[normal_datas_in, bearing_datas_in, gear_datas_in]
    labels = np.r_[np.zeros(normal_datas_in.shape[0]),  # 0 for inlier, 1 for outlier
                   np.ones(bearing_datas_in.shape[0]),
                   np.ones(gear_datas_in.shape[0])]

    train_datas, test_datas, train_labels, test_labels = utils.split_train_test(datas=datas, labels=labels,
                                                                                frac=0.8)
    for n_neighbor in [20,40,60,100]:
        for n_contamination in [0.05, 0.1]:
            lof_model = LocalOutlierFactor(n_neighbors=n_neighbor, contamination=n_contamination)
            lof_model.fit(train_datas)  # create_lof_model(train_datas.shape[0] // 3).fit(train_datas)
            y_score = -lof_model._decision_function(test_datas)
            # Compute ROC curve and ROC area for each class
            fpr, tpr, thresholds = roc_curve(test_labels, y_score)
            threshold = get_best_threshold_roc(fpr=fpr, tpr=tpr, thresholds=thresholds)
            roc_auc = auc(fpr, tpr)

            # y_score_test = -lof_model._decision_function(test_datas)
            y_pred = np.zeros(test_labels.shape[0])
            y_pred[y_score >= threshold] = 1
            f1 = f1_score(test_labels, y_pred)
            # select best model with best roc_auc
            if f1 > best_test_score:
                best_test_score = f1
                best_model = lof_model
                best_threshold = threshold

            print('n_neighbor: %d, n_contamination: %f, roc_auc score: %.3f, f1 score: %.3f'
                  % (n_neighbor, n_contamination, roc_auc, f1))


    # # save best model to disk
    # filename = 'finalized_model_1.sav'
    # joblib.dump(best_model, filename)

    print('[Test phase] START ')
    out_test_datas = np.vstack([normal_datas_out, bearing_datas_out, gear_datas_out])
    out_test_labels = np.hstack([np.zeros(normal_datas_out.shape[0]),  # 0 for inlier, 1 for outlier
                                 np.ones(bearing_datas_out.shape[0]),
                                 np.ones(gear_datas_out.shape[0])])
    # y_score = -best_model.negative_outlier_factor_
    y_score_test = -best_model._decision_function(out_test_datas)
    fpr, tpr, thresholds = roc_curve(out_test_labels, y_score_test)
    roc_auc = auc(fpr, tpr)


    y_pred = np.zeros(out_test_labels.shape[0])
    y_pred[y_score_test >= best_threshold] = 1
    f1 = f1_score(out_test_labels, y_pred)
    print('[Test phase] roc_auc score: %.3f, f1 score: %.3f ' % (roc_auc, f1))


if __name__ == "__main__":
    # set Japanese Font
    fp = FontProperties(fname=r'/System/Library/Fonts/ヒラギノ明朝 ProN.ttc', size=9)

    try:
        accs_normal1 = utils.merageVm2012Data(normal_position1_data_path + '/' + csv_file_regular_expression)
        print("Normal position 1 : ", accs_normal1.size)
        accs_bearing1 = utils.merageVm2012Data(bearing_position1_data_path + '/' + csv_file_regular_expression)
        print("Bearing position 1 : ", accs_bearing1.size)
        accs_gear1 = utils.merageVm2012Data(gear_position1_data_path + '/' + csv_file_regular_expression)
        print("Gear position 1 : ", accs_gear1.size)
        accs_norma2 = utils.merageVm2012Data(normal_position2_data_path + '/' + csv_file_regular_expression)
        print("Normal position 2 : ", accs_norma2.size)
        accs_bearing2 = utils.merageVm2012Data(bearing_position2_data_path + '/' + csv_file_regular_expression)
        print("Bearing position 2 : ", accs_bearing2.size)
        accs_gear2 = utils.merageVm2012Data(gear_position2_data_path + '/' + csv_file_regular_expression)
        print("Gear position 2 : ", accs_gear2.size)
    except Exception as e:
        print(e.__str__())

    # model for position 1
    print("Model 1")
    build_model(accs_normal1,accs_bearing1,accs_gear1)

    # model for position 2
    print("Model 2")
    build_model(accs_norma2, accs_bearing2, accs_gear2)